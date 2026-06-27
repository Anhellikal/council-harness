"""contract_parts orchestration: architecture -> implement -> review -> fix -> merge."""

from __future__ import annotations

import ast
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .agent import call_agent
from .contracts import (
    ContractValidationError,
    CouncilContract,
    MergeResult,
    PartAssignment,
    PartDraft,
    PartReview,
    build_assignments,
    contract_text,
    cross_review_pairs,
    parse_contract,
)
from .loop import CouncilResult, run_council
from .prompts import (
    architecture_iteration_prompt,
    architecture_prompt,
    architecture_retry_prompt,
    contract_retry_prompt,
    cross_review_prompt,
    implementation_prompt,
    integration_review_prompt,
    part_fix_prompt,
)


@dataclass
class PipelineResult:
    final_draft: str
    contract: CouncilContract
    assignments: list[PartAssignment]
    drafts: list[PartDraft]
    reviews: list[PartReview]
    merge_result: MergeResult
    integration_feedback: str
    syntax_errors: list[str]
    interface_warnings: list[str]
    rounds_run: int
    architecture_result: CouncilResult = field(repr=False)


_DEBUG_DIR = os.path.join(tempfile.gettempdir(), "council-pipeline-debug")


def _dump_debug(name: str, prompt: str, raw: str) -> str:
    """Write a failed implementation's prompt + raw output for diagnosis.

    Returns the path written. Survives the dashboard UI swallowing emit output.
    """
    os.makedirs(_DEBUG_DIR, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)
    path = os.path.join(_DEBUG_DIR, f"{safe}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("===== PROMPT =====\n")
        f.write(prompt)
        f.write("\n\n===== RAW OUTPUT =====\n")
        f.write(raw)
    return path


def _strip_block_fences(content: str) -> str:
    """Strip a single wrapping ```lang ... ``` fence from a file block's body.

    Agents commonly wrap each ### FILE: block's content in a markdown code
    fence; left in, the leading ```python makes every file a syntax error.
    Only strips a matched outer pair, leaving inner/nested fences intact.
    """
    lines = content.splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        inner = lines[1:]
        if inner and inner[-1].strip().startswith("```"):
            inner = inner[:-1]
        return "\n".join(inner).strip()
    return content


def _extract_file_blocks(text: str) -> dict[str, str]:
    marker = re.compile(r"^### FILE:\s*(.+)$", re.MULTILINE)
    matches = list(marker.finditer(text))
    files: dict[str, str] = {}
    for i, match in enumerate(matches):
        path = match.group(1).strip()
        start = match.end()
        if start < len(text) and text[start] == "\n":
            start += 1
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        files[path] = _strip_block_fences(text[start:end].rstrip())
    return files


def _render_interfaces(contract: CouncilContract) -> str:
    if not contract.interfaces:
        return "(none)"
    return "\n".join(f"- {sig}: {desc}" for sig, desc in contract.interfaces)


def _match_owned(parsed_keys: list[str], owned_files: list[str]) -> dict[str, str]:
    """Map each emitted file path to the canonical owned path it satisfies.

    Agents sometimes drift on directory prefixes (e.g. emit `dashboard/main.py`
    when the contract owns `main.py`, or vice versa). Match exact paths first,
    then fall back to an *unambiguous* basename match so prefix drift doesn't
    fail every file. Returns {emitted_key: canonical_owned_path}.
    """
    owned_set = set(owned_files)
    mapping: dict[str, str] = {}
    remaining_owned = list(owned_files)

    # Pass 1: exact path matches.
    for key in parsed_keys:
        if key in owned_set:
            mapping[key] = key
            if key in remaining_owned:
                remaining_owned.remove(key)

    # Pass 2: unambiguous basename matches for anything still unclaimed.
    owned_by_base: dict[str, list[str]] = {}
    for path in remaining_owned:
        owned_by_base.setdefault(os.path.basename(path), []).append(path)
    for key in parsed_keys:
        if key in mapping:
            continue
        candidates = owned_by_base.get(os.path.basename(key))
        if candidates and len(candidates) == 1:
            mapping[key] = candidates[0]
            owned_by_base[os.path.basename(key)] = []  # consume it

    return mapping


def _filter_owned_files(raw_text: str, owned_files: list[str]) -> tuple[str, list[str]]:
    parsed = _extract_file_blocks(raw_text)
    match = _match_owned(list(parsed.keys()), owned_files)
    accepted: dict[str, str] = {}
    for emitted_key, content in parsed.items():
        canonical = match.get(emitted_key)
        if canonical is not None:
            accepted[canonical] = content
    skipped = sorted(key for key in parsed if key not in match)
    merged_text = "\n\n".join(f"### FILE: {path}\n{content}" for path, content in accepted.items())
    return merged_text, skipped


def _parse_contract_with_retry(
    winner_agent: dict,
    raw_text: str,
    active_agent_ids: list[str],
    emit: Callable[[str], None],
) -> CouncilContract:
    try:
        return parse_contract(raw_text, active_agent_ids)
    except ContractValidationError as exc:
        emit(f"[Contract] validation failed — retrying once with [{winner_agent['id']}] ({exc})")
        retry_text, _ = call_agent(winner_agent, contract_retry_prompt(str(exc), active_agent_ids))
        try:
            return parse_contract(retry_text, active_agent_ids)
        except ContractValidationError as retry_exc:
            raise RuntimeError(f"Architecture round produced no valid contract: {retry_exc}") from retry_exc


def _run_parallel(
    items: list[Any],
    worker: Callable[[Any], Any],
) -> list[Any]:
    if not items:
        return []
    results: list[Any] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=len(items)) as ex:
        futures = {ex.submit(worker, item): idx for idx, item in enumerate(items)}
        for future in as_completed(futures):
            results[futures[future]] = future.result()
    return results


def _merge_drafts(drafts: list[PartDraft]) -> MergeResult:
    files: dict[str, str] = {}
    unresolved: list[str] = []
    for draft in drafts:
        parsed = _extract_file_blocks(draft.implementation)
        for owned_file in draft.owned_files:
            if owned_file not in parsed:
                unresolved.append(f"{draft.agent_id} missing owned file {owned_file}")
                continue
            files[owned_file] = parsed[owned_file]
    merged_text = "\n\n".join(f"### FILE: {path}\n{content}" for path, content in files.items())
    return MergeResult(files=files, merged_text=merged_text, unresolved=unresolved)


def _validate_syntax(merge_result: MergeResult) -> list[str]:
    errors = []
    for path, content in merge_result.files.items():
        try:
            ast.parse(content)
        except SyntaxError as exc:
            errors.append(f"{path}: SyntaxError line {exc.lineno}: {exc.msg}")
    return errors


def _interface_name(sig: str) -> str:
    """Extract a bare identifier from a contract interface signature.

    Tolerates markdown contamination (backticks, asterisks) and `def`/`class`
    prefixes that agents add, e.g. "`load_run(x) -> Y`" -> "load_run".
    """
    cleaned = sig.strip().strip("`*").strip()
    for prefix in ("def ", "class ", "async def "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    return cleaned.split("(")[0].strip().strip("`*").strip()


def _validate_interfaces(contract: CouncilContract, merge_result: MergeResult) -> list[str]:
    warnings = []
    top_level_names: set[str] = set()
    for content in merge_result.files.values():
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue
        # Functions AND classes — a dataclass interface (e.g. RunData) is a
        # legitimate top-level definition, not a missing function.
        top_level_names.update(
            node.name
            for node in tree.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        )
    for sig, _ in contract.interfaces:
        name = _interface_name(sig)
        if name and name not in top_level_names:
            warnings.append(
                f"{name}: defined in contract INTERFACES but not found as a "
                f"module-level function or class in any output file (may be a method or missing)"
            )
    return warnings


def _next_retry_agent(
    current_agent_id: str,
    agent_order: list[str],
    attempted_ids: set[str] | None = None,
) -> str | None:
    """Return the next available agent in council order, excluding attempted_ids."""
    if len(agent_order) < 2:
        return None
    attempted = set(attempted_ids or set())
    try:
        start = agent_order.index(current_agent_id)
    except ValueError:
        start = -1
    for offset in range(1, len(agent_order) + 1):
        candidate = agent_order[(start + offset) % len(agent_order)]
        if candidate not in attempted:
            return candidate
    return None


def run_pipeline(
    task: str,
    active_agents: list[dict],
    config: dict,
    emit: Callable[[str], None] = print,
    context_files: list[tuple[str, str]] | None = None,
    event_cb: Optional[Callable[[str, dict[str, Any]], None]] = None,
) -> PipelineResult:
    rounds_cfg = config.get("rounds", {})
    max_rounds: int = rounds_cfg.get("max", 5)

    def _phase(phase: str, label: str) -> None:
        emit(f"\n[{label}]")
        if event_cb:
            event_cb("pipeline_phase", {"phase": phase, "label": label})

    _phase("architecture", "Architecture")
    architecture_result = run_council(
        task=task,
        active_agents=active_agents,
        config=config,
        emit=emit,
        multifile=False,
        context_files=context_files,
        event_cb=event_cb,
        review_enabled=False,
        round1_prompt_fn=lambda task, multifile=False, context_files=None: architecture_prompt(
            task,
            active_agents,
            context_files=context_files,
        ),
        iteration_prompt_fn=lambda task, agent_id, round_num, max_rounds, current_drafts, multifile=False, context_files=None: architecture_iteration_prompt(
            task,
            agent_id,
            round_num,
            max_rounds,
            current_drafts,
        ),
        retry_prompt_fn=architecture_retry_prompt,
    )

    if not architecture_result.final_draft:
        raise RuntimeError("Architecture round did not produce a winning proposal.")

    _phase("contract", "Contract")
    winner_agent = next((agent for agent in active_agents if agent["id"] == architecture_result.winning_agent_id), active_agents[0])
    contract = _parse_contract_with_retry(
        winner_agent=winner_agent,
        raw_text=architecture_result.final_draft,
        active_agent_ids=[agent["id"] for agent in active_agents],
        emit=emit,
    )
    canonical_contract = contract_text(contract)
    assignments = build_assignments(contract)
    if event_cb:
        event_cb(
            "pipeline_assigned",
            {"parts": [{"agent": item.agent_id, "files": item.owned_files} for item in assignments]},
        )

    _phase("assignment", "Assignment")
    for assignment in assignments:
        emit(f"  [{assignment.agent_id}] owns {', '.join(assignment.owned_files)}")

    assignment_map = {item.agent_id: item for item in assignments}
    agent_map = {agent["id"]: agent for agent in active_agents}
    agent_order = [agent["id"] for agent in active_agents]

    _phase("implementation", "Implementation")

    def _implementation_attempt(assignment: PartAssignment, actor_id: str) -> PartDraft:
        prompt = implementation_prompt(task, canonical_contract, assignment.owned_files)
        try:
            raw, _ = call_agent(agent_map[actor_id], prompt)
            filtered, skipped = _filter_owned_files(raw, assignment.owned_files)
            if skipped:
                emit(f"  [{actor_id}] ignored unowned files for [{assignment.agent_id}]: {', '.join(skipped)}")
            missing_owned = [path for path in assignment.owned_files if path not in _extract_file_blocks(filtered)]
            if missing_owned:
                dbg = _dump_debug(f"impl-{assignment.agent_id}-by-{actor_id}", prompt, raw)
                emit(f"  [{actor_id}] implementation missing owned files for [{assignment.agent_id}]: {', '.join(missing_owned)}")
                emit(f"  [{actor_id}] raw output saved -> {dbg}")
                return PartDraft(
                    agent_id=assignment.agent_id,
                    owned_files=assignment.owned_files,
                    implementation=filtered,
                    failed=True,
                    implemented_by=actor_id,
                )
            if event_cb:
                event_cb(
                    "pipeline_part_done",
                    {"phase": "implement", "agent": actor_id, "owner": assignment.agent_id, "files": assignment.owned_files},
                )
            return PartDraft(
                agent_id=assignment.agent_id,
                owned_files=assignment.owned_files,
                implementation=filtered,
                implemented_by=actor_id,
            )
        except Exception as exc:
            emit(f"  [{actor_id}] implementation failed for [{assignment.agent_id}]: {exc}")
            return PartDraft(
                agent_id=assignment.agent_id,
                owned_files=assignment.owned_files,
                failed=True,
                implemented_by=actor_id,
            )

    def _implement(assignment: PartAssignment) -> PartDraft:
        return _implementation_attempt(assignment, assignment.agent_id)

    draft_results = _run_parallel(assignments, _implement)
    for idx, draft in enumerate(draft_results):
        if not draft.failed:
            continue
        retry_agent = _next_retry_agent(draft.implemented_by or draft.agent_id, agent_order, {draft.implemented_by or draft.agent_id})
        if retry_agent is None:
            continue
        emit(f"  [Retry] reassigning {', '.join(draft.owned_files)} from [{draft.agent_id}] to [{retry_agent}]")
        retried = _implementation_attempt(assignments[idx], retry_agent)
        if not retried.failed:
            emit(f"  [Retry] [{retry_agent}] completed files for [{draft.agent_id}]")
            draft_results[idx] = retried
        else:
            emit(f"  [Retry] [{retry_agent}] also failed for [{draft.agent_id}]")
    draft_by_agent = {draft.agent_id: draft for draft in draft_results}

    _phase("cross_review", "Cross-Review")
    review_pairs = cross_review_pairs(assignments)

    def _review(pair: tuple[str, str]) -> PartReview:
        reviewer_id, target_id = pair
        target_draft = draft_by_agent[target_id]
        try:
            feedback, _ = call_agent(
                agent_map[reviewer_id],
                cross_review_prompt(
                    task,
                    canonical_contract,
                    target_draft.implemented_by or target_id,
                    target_draft.owned_files,
                    target_draft.implementation,
                ),
            )
            if event_cb:
                event_cb(
                    "pipeline_part_done",
                    {"phase": "review", "agent": reviewer_id, "target": target_id, "files": target_draft.owned_files},
                )
            return PartReview(
                reviewer_id=reviewer_id,
                target_agent_id=target_id,
                target_files=target_draft.owned_files,
                feedback=feedback,
            )
        except Exception as exc:
            emit(f"  [{reviewer_id}] review for [{target_id}] failed: {exc}")
            return PartReview(
                reviewer_id=reviewer_id,
                target_agent_id=target_id,
                target_files=target_draft.owned_files,
                feedback=f"(review failed: {exc})",
                failed=True,
            )

    reviews = _run_parallel(review_pairs, _review)
    review_by_target = {review.target_agent_id: review for review in reviews}

    _phase("fix", "Fix")
    interfaces_text = _render_interfaces(contract)

    def _fix_attempt(draft: PartDraft, review: PartReview, actor_id: str) -> tuple[PartDraft, bool]:
        try:
            fixed, _ = call_agent(
                agent_map[actor_id],
                part_fix_prompt(
                    task,
                    interfaces_text,
                    draft.owned_files,
                    draft.implementation,
                    review.feedback,
                ),
            )
            filtered, skipped = _filter_owned_files(fixed, draft.owned_files)
            if skipped:
                emit(f"  [{actor_id}] ignored unowned files during fix for [{draft.agent_id}]: {', '.join(skipped)}")
            missing_owned = [path for path in draft.owned_files if path not in _extract_file_blocks(filtered)]
            if missing_owned:
                emit(f"  [{actor_id}] fix output missing owned files for [{draft.agent_id}]: {', '.join(missing_owned)}")
                return draft, False
            if event_cb:
                event_cb(
                    "pipeline_part_done",
                    {"phase": "fix", "agent": actor_id, "owner": draft.agent_id, "files": draft.owned_files},
                )
            return (
                PartDraft(
                    agent_id=draft.agent_id,
                    owned_files=draft.owned_files,
                    implementation=filtered or draft.implementation,
                    failed=draft.failed,
                    implemented_by=actor_id,
                ),
                True,
            )
        except Exception as exc:
            emit(f"  [{actor_id}] fix failed for [{draft.agent_id}]: {exc}")
            return draft, False

    def _fix(draft: PartDraft) -> PartDraft:
        review = review_by_target.get(draft.agent_id)
        if draft.failed or review is None or review.failed or "NO ISSUES FOUND" in review.feedback:
            return draft
        primary_actor = draft.implemented_by or draft.agent_id
        updated, ok = _fix_attempt(draft, review, primary_actor)
        if ok:
            return updated
        retry_agent = _next_retry_agent(primary_actor, agent_order, {primary_actor})
        if retry_agent is None:
            return draft
        emit(f"  [Retry] reassigning fix for {', '.join(draft.owned_files)} from [{primary_actor}] to [{retry_agent}]")
        retried, retry_ok = _fix_attempt(draft, review, retry_agent)
        if retry_ok:
            emit(f"  [Retry] [{retry_agent}] applied fixes for [{draft.agent_id}]")
            return retried
        emit(f"  [Retry] [{retry_agent}] also failed to fix [{draft.agent_id}]")
        return draft

    fixed_drafts = _run_parallel(draft_results, _fix)

    _phase("merge", "Merge")
    merge_result = _merge_drafts(fixed_drafts)
    for unresolved in merge_result.unresolved:
        emit(f"  unresolved: {unresolved}")
    failed_agents = [draft.agent_id for draft in fixed_drafts if draft.failed]
    if failed_agents or merge_result.unresolved:
        # Surface what each draft actually emitted vs. what it owned, so a
        # path/format mismatch is diagnosable from the console without artifacts.
        emit("  [Merge aborted] draft diagnostics:")
        for draft in fixed_drafts:
            emitted = sorted(_extract_file_blocks(draft.implementation or "").keys())
            emit(
                f"    [{draft.agent_id}] owns {draft.owned_files} | "
                f"emitted {emitted or '(nothing)'}"
                + (" | FAILED" if draft.failed else "")
            )
        failures = []
        if failed_agents:
            failures.append(f"failed drafts from {', '.join(failed_agents)}")
        if merge_result.unresolved:
            failures.append(f"missing owned files: {', '.join(merge_result.unresolved)}")
        raise RuntimeError("contract_parts merge aborted due to incomplete implementation: " + "; ".join(failures))

    _phase("validation", "Validation")
    syntax_errors = _validate_syntax(merge_result)
    interface_warnings = _validate_interfaces(contract, merge_result)
    for err in syntax_errors:
        emit(f"  [SYNTAX ERROR] {err}")
    for warn in interface_warnings:
        emit(f"  [INTERFACE WARNING] {warn}")
    if not syntax_errors and not interface_warnings:
        emit("  all files pass syntax and interface checks")

    _phase("integration", "Integration")
    integration_feedback = ""
    try:
        integration_feedback, _ = call_agent(
            winner_agent,
            integration_review_prompt(task, canonical_contract, merge_result.merged_text),
        )
        emit("  advisory integration review collected")
    except Exception as exc:
        integration_feedback = f"(integration review failed: {exc})"
        emit(f"  integration review failed: {exc}")

    if event_cb:
        event_cb(
            "result",
            {
                "status": "PIPELINE",
                "winner_id": architecture_result.winning_agent_id,
                "final_excerpt": merge_result.merged_text[:400],
            },
        )

    return PipelineResult(
        final_draft=merge_result.merged_text,
        contract=contract,
        assignments=assignments,
        drafts=fixed_drafts,
        reviews=reviews,
        merge_result=merge_result,
        integration_feedback=integration_feedback,
        syntax_errors=syntax_errors,
        interface_warnings=interface_warnings,
        rounds_run=architecture_result.rounds_run,
        architecture_result=architecture_result,
    )
