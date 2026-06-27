"""Council main loop: rounds, convergence tracking, tiebreaker."""

import collections
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .agent import call_agent, cli_can_write
from .prompts import round1_prompt, iteration_prompt, retry_prompt, writer_prompt, review_prompt, fix_prompt

MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RoundSummary:
    round_num: int
    events: list[str] = field(default_factory=list)
    distinct_count: int = 0


@dataclass
class CouncilResult:
    final_draft: str
    winning_agent_id: str
    rounds_run: int
    consensus_reached: bool
    transcript: list[RoundSummary]
    tiebreak_options: Optional[dict[str, str]] = None  # {agent_id: draft}
    reviewed_by: Optional[str] = None  # set when review+fix phase ran


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_response(
    text: str,
    active_ids: set[str],
    self_id: str,
) -> Optional[dict]:
    """
    Parse ACTION/TARGET/DRAFT from agent response.
    Returns a dict with key 'action' ('revise'|'adopt'), or None on failure.
    """
    text = text.strip()

    m = re.search(r"^ACTION:\s*(revise|adopt)", text, re.IGNORECASE | re.MULTILINE)
    if not m:
        return None

    action = m.group(1).lower()

    if action == "adopt":
        tm = re.search(r"^TARGET:\s*(\S+)", text, re.IGNORECASE | re.MULTILINE)
        if not tm:
            return None
        target = tm.group(1).strip().strip(".,;")
        # Strip surrounding brackets/quotes that models sometimes add
        target = target.strip("[]()\"'")
        if target == self_id:
            # Self-adoption is a no-op; caller will retry
            return None
        if target not in active_ids:
            return None
        return {"action": "adopt", "target": target}

    if action == "revise":
        dm = re.search(r"^DRAFT:\s*(.*)", text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if not dm:
            return None
        draft = dm.group(1).strip()
        if not draft:
            return None
        return {"action": "revise", "draft": draft}

    return None


# ---------------------------------------------------------------------------
# Convergence helpers
# ---------------------------------------------------------------------------

def _effective_threshold(
    config_threshold: int,
    config_total: int,
    active_count: int,
) -> int:
    """Scale consensus_threshold proportionally to the active council size."""
    if config_total <= 0:
        return active_count
    ratio = config_threshold / config_total
    effective = math.ceil(active_count * ratio)
    return max(1, min(effective, active_count))


def _adoption_draft(aid: str, target: str, round_adopts: dict[str, str], snapshot: dict[str, str]) -> str:
    """Return the draft aid should receive when adopting target.
    Detects adoption cycles (A→B→A) and collapses them: all cycle participants
    get the snapshot of the lexically smallest agent in the cycle."""
    visited: set[str] = set()
    cur = target
    while cur in round_adopts and cur not in visited:
        visited.add(cur)
        cur = round_adopts[cur]
        if cur == aid:
            # Cycle detected — all participants converge on min-id snapshot
            return snapshot.get(min(visited | {aid}), "")
    return snapshot.get(target, "")


def _merge_multifile(old_draft: str, new_draft: str) -> str:
    """Merge partial ### FILE: blocks from new_draft into old_draft.
    Files present in new_draft replace their counterparts; absent files are kept."""
    marker = re.compile(r"^### FILE:\s*(.+)$", re.MULTILINE)

    def _extract(text: str) -> dict[str, str]:
        matches = list(marker.finditer(text))
        files: dict[str, str] = {}
        for i, m in enumerate(matches):
            path = m.group(1).strip()
            start = m.end() + (1 if m.end() < len(text) and text[m.end()] == "\n" else 0)
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            files[path] = text[start:end].rstrip()
        return files

    old_files = _extract(old_draft)
    new_files = _extract(new_draft)
    if not new_files:
        return new_draft  # not multifile format — use as-is
    merged = {**old_files, **new_files}
    return "\n".join(f"### FILE: {p}\n{c}" for p, c in merged.items())


def _distinct_count(drafts: dict[str, str]) -> int:
    return len(set(drafts.values()))


def _convergence_winner(drafts: dict[str, str], threshold: int) -> Optional[str]:
    """Return winning draft text if any reaches threshold, else None."""
    counts = collections.Counter(drafts.values())
    top_text, top_count = counts.most_common(1)[0]
    return top_text if top_count >= threshold else None


def _find_winner_id(drafts: dict[str, str], winner_text: str) -> str:
    for aid, draft in drafts.items():
        if draft == winner_text:
            return aid
    return "unknown"


# ---------------------------------------------------------------------------
# Per-agent iteration (handles retries internally)
# ---------------------------------------------------------------------------

def _run_agent_iteration(
    agent: dict,
    initial_prompt: str,
    active_ids: set[str],
    emit: Callable[[str], None],
    retry_prompt_fn: Callable[[str], str] = retry_prompt,
) -> dict:
    """
    Call one agent for an iteration round, retrying on parse failure.
    Returns:
      {"action": "revise"|"adopt", ...}  — success
      {"action": "keep"}                 — agent responded but format was bad every time
      {"action": "drop"}                 — agent unreachable (network/timeout every attempt)
    """
    aid = agent["id"]
    raw = ""
    call_failures = 0

    for attempt in range(MAX_RETRIES):
        prompt = initial_prompt if attempt == 0 else retry_prompt_fn(raw)
        try:
            emit(f"  [{aid}] calling… (attempt {attempt + 1}/{MAX_RETRIES})")
            t = time.monotonic()
            raw, stats = call_agent(agent, prompt)
            elapsed = time.monotonic() - t
            parsed = _parse_response(raw, active_ids, aid)
            if parsed:
                parsed["_elapsed"] = elapsed
                parsed["_stats"] = stats
                return parsed
            emit(f"  [{aid}] parse failed ({elapsed:.1f}s) — response not in expected format, retrying…")
        except Exception as exc:
            call_failures += 1
            emit(f"  [{aid}] call error: {exc}")

    if call_failures == MAX_RETRIES:
        emit(f"  [{aid}] unreachable on all attempts — dropping from council")
        return {"action": "drop"}

    emit(f"  [{aid}] gave up parsing after {MAX_RETRIES} attempts — keeping current draft")
    return {"action": "keep"}


# ---------------------------------------------------------------------------
# Main council loop
# ---------------------------------------------------------------------------

def run_writer(
    task: str,
    agreed_draft: str,
    writer_cfg: dict,
    emit: Callable[[str], None] = print,
    multifile: bool = False,
    event_cb: Optional[Callable[[str, dict[str, Any]], None]] = None,
    output_dir: Optional[Any] = None,
) -> str:
    """
    Pass the agreed draft to the designated writer agent for final transcription.
    Returns the writer's output, or the original draft if the call fails.

    If the writer is a CLI agent with `can_write: true` and output_dir is given,
    the writer runs inside output_dir with write tools enabled and produces the
    files itself; the caller is responsible for skipping its own file write
    (see agent.cli_can_write).
    """
    if event_cb:
        event_cb("writer_started", {"agent_id": writer_cfg["id"]})
    emit(f"\n[Writer] Sending agreed draft to [{writer_cfg['id']}]…")
    workdir = str(output_dir) if (output_dir is not None and cli_can_write(writer_cfg)) else None
    if workdir:
        os.makedirs(workdir, exist_ok=True)  # subprocess cwd must exist
        emit(f"[Writer] Write-enabled: [{writer_cfg['id']}] will write files into {output_dir}/")
    try:
        result, stats = call_agent(
            writer_cfg,
            writer_prompt(task, agreed_draft, multifile=multifile, write_mode=bool(workdir)),
            workdir=workdir,
        )
        perf = f"  {stats['tps']} tok/s" if "tps" in stats else ""
        emit(f"[Writer] Done ({len(result):,} chars{perf}).")
        return result
    except Exception as exc:
        emit(f"[Writer] Call failed ({exc}) — using council draft unchanged.")
        return agreed_draft


def _run_review_phase(
    agents: list[dict],
    draft: str,
    multifile: bool,
    emit: Callable[[str], None],
    event_cb: Optional[Callable[[str, dict[str, Any]], None]] = None,
    winner_id: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """
    Review round: all agents find bugs in the agreed draft in parallel.
    Fix round: the winning agent applies the combined feedback; falls back to agents[0].
    Returns (fixed_draft, fixer_agent_id) — fixer_agent_id is None if no fix was applied.
    """
    if event_cb:
        event_cb("review_started", {})
    emit("\n[Review Round] Agents checking for bugs…")
    t0 = time.monotonic()

    def _review(agent: dict) -> tuple[str, str, bool]:
        try:
            text, _ = call_agent(agent, review_prompt(draft))
            return agent["id"], text, True
        except Exception as exc:
            return agent["id"], f"(review failed: {exc})", False

    reviews: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(agents)) as ex:
        for aid, feedback, ok in ex.map(lambda a: _review(a), agents):
            if ok:
                reviews[aid] = feedback
                preview = feedback[:120].replace("\n", " ")
                emit(f"  [{aid}] {preview}…")
            else:
                emit(f"  [{aid}] review failed — skipping")

    emit(f"  Review round complete  [{time.monotonic() - t0:.1f}s]")
    if event_cb:
        event_cb("review_complete", {})

    if not reviews:
        emit("  No reviews collected — skipping fix round")
        return draft, None

    combined = "\n\n".join(f"[{aid}]:\n{fb}" for aid, fb in reviews.items())
    emit("\n[Fix Round] Applying fixes…")
    t1 = time.monotonic()
    agent_map = {a["id"]: a for a in agents}
    fixer = agent_map.get(winner_id or "", agents[0]) if winner_id else agents[0]
    try:
        fixed, _ = call_agent(fixer, fix_prompt(draft, combined, multifile=multifile))
        if fixed.strip():
            emit(f"  [{fixer['id']}] fixed draft ({len(fixed):,} chars)  [{time.monotonic() - t1:.1f}s]")
            if event_cb:
                event_cb("fix_applied", {"agent_id": fixer["id"], "chars": len(fixed)})
            return fixed, fixer["id"]
        emit(f"  [{fixer['id']}] returned empty — keeping original draft")
        return draft, None
    except Exception as exc:
        emit(f"  Fix round failed ({exc}) — keeping original draft")
        return draft, None


def run_council(
    task: str,
    active_agents: list[dict],
    config: dict,
    emit: Callable[[str], None] = print,
    multifile: bool = False,
    context_files: list[tuple[str, str]] | None = None,
    event_cb: Optional[Callable[[str, dict[str, Any]], None]] = None,
    review_enabled: Optional[bool] = None,
    round1_prompt_fn: Optional[Callable[[str, bool, list[tuple[str, str]] | None], str]] = None,
    iteration_prompt_fn: Optional[
        Callable[[str, str, int, int, dict[str, str], bool, list[tuple[str, str]] | None], str]
    ] = None,
    retry_prompt_fn: Callable[[str], str] = retry_prompt,
) -> CouncilResult:
    """
    Run the full council loop.
    emit: progress callback — receives human-readable strings.
    """
    rounds_cfg = config.get("rounds", {})
    max_rounds: int = rounds_cfg.get("max", 5)
    config_threshold: int = rounds_cfg.get("consensus_threshold", 2)
    config_total: int = len(config.get("agents", active_agents))
    active_count: int = len(active_agents)
    if review_enabled is None:
        review_enabled = config.get("review_round", True)
    if round1_prompt_fn is None:
        round1_prompt_fn = round1_prompt
    if iteration_prompt_fn is None:
        iteration_prompt_fn = iteration_prompt

    threshold = _effective_threshold(config_threshold, config_total, active_count)
    active_ids = {a["id"] for a in active_agents}

    emit(f"Consensus threshold: {threshold}/{active_count} (scaled from {config_threshold}/{config_total})")
    if event_cb:
        event_cb(
            "council_start",
            {
                "threshold": threshold,
                "active_count": active_count,
                "active_ids": list(active_ids),
                "max_rounds": max_rounds,
            },
        )

    def _maybe_review(result: CouncilResult) -> CouncilResult:
        if review_enabled and result.final_draft and live_agents:
            fixed, fixer_id = _run_review_phase(live_agents, result.final_draft, multifile, emit, event_cb=event_cb, winner_id=result.winning_agent_id)
            result.final_draft = fixed
            result.reviewed_by = fixer_id
        return result

    transcript: list[RoundSummary] = []
    current_drafts: dict[str, str] = {}
    live_agents = list(active_agents)  # shrinks if agents drop out mid-run

    # -----------------------------------------------------------------------
    # Round 1 — independent proposals
    # -----------------------------------------------------------------------
    if event_cb:
        event_cb(
            "round_start",
            {
                "round_num": 1,
                "active_count": len(live_agents),
                "active_ids": [a["id"] for a in live_agents],
                "threshold": threshold,
            },
        )
    emit("\n[Round 1] Independent proposals...")
    summary = RoundSummary(round_num=1)
    t0 = time.monotonic()

    r1_times: dict[str, float] = {}

    def _timed_round1(agent: dict) -> tuple[str, "str | Exception", dict]:
        t = time.monotonic()
        prompt = round1_prompt_fn(task, multifile=multifile, context_files=context_files)
        try:
            result, stats = call_agent(agent, prompt)
            r1_times[agent["id"]] = time.monotonic() - t
            return agent["id"], result, stats
        except Exception as exc:
            r1_times[agent["id"]] = time.monotonic() - t
            return agent["id"], exc, {}

    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
    raw_results: dict[str, "str | Exception"] = {}
    r1_stats: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(len(active_agents), 1)) as ex:
        futures = [ex.submit(_timed_round1, a) for a in active_agents]
        for fut in _as_completed(futures):
            aid, res, stats = fut.result()
            raw_results[aid] = res
            r1_stats[aid] = stats

    dropped_r1 = []
    for agent in active_agents:
        aid = agent["id"]
        result = raw_results[aid]
        t_str = f"{r1_times.get(aid, 0):.1f}s"
        stats = r1_stats.get(aid, {})
        perf = ""
        if "tps" in stats:
            perf = f"  {stats['tps']} tok/s  ttft {stats.get('ttft_ms', '?')}ms"
        elif "tokens" in stats and r1_times.get(aid, 0):
            perf = f"  ~{round(stats['tokens'] / r1_times[aid], 1)} tok/s"

        if isinstance(result, Exception) or not result:
            reason = str(result) if isinstance(result, Exception) else "empty response (thinking stripped?)"
            emit(f"  [{aid}] FAILED ({reason}) — dropping from council")
            dropped_r1.append(agent)
            summary.events.append(f"{aid}: DROPPED (round 1 failure)")
            if event_cb:
                event_cb(
                    "agent_round1_result",
                    {"agent_id": aid, "result": "dropped", "reason": reason},
                )
        else:
            current_drafts[aid] = result
            preview = result[:72].replace("\n", " ")
            emit(f"  [{aid}] received {len(result):,} chars  [{t_str}{perf}] — {preview}…")
            summary.events.append(f"{aid}: proposed")
            if event_cb:
                event_cb(
                    "agent_round1_result",
                    {"agent_id": aid, "result": "proposed", "chars": len(result)},
                )

    for agent in dropped_r1:
        live_agents.remove(agent)

    if not live_agents:
        emit("\nAll agents failed in round 1. Aborting.")
        return CouncilResult(
            final_draft="",
            winning_agent_id="",
            rounds_run=1,
            consensus_reached=False,
            transcript=transcript,
        )

    if dropped_r1:
        threshold = _effective_threshold(config_threshold, config_total, len(live_agents))
        emit(f"  Council shrank to {len(live_agents)} agent(s) — threshold rescaled to {threshold}/{len(live_agents)}")

    summary.distinct_count = _distinct_count(current_drafts)
    transcript.append(summary)
    emit(f"  Distinct proposals: {summary.distinct_count}  [{time.monotonic() - t0:.1f}s]")
    if event_cb:
        event_cb(
            "round_complete",
            {
                "round_num": 1,
                "distinct_count": summary.distinct_count,
                "active_count": len(live_agents),
                "threshold": threshold,
            },
        )

    # Single-agent degenerate case (either configured that way, or everyone else dropped)
    if len(live_agents) == 1:
        sole = live_agents[0]["id"]
        emit("\nSingle-agent council — round 1 result is final.")
        return _maybe_review(CouncilResult(
            final_draft=current_drafts[sole],
            winning_agent_id=sole,
            rounds_run=1,
            consensus_reached=True,
            transcript=transcript,
        ))

    # Early consensus check after round 1
    if winner_text := _convergence_winner(current_drafts, threshold):
        emit("\nConsensus reached after round 1!")
        return _maybe_review(CouncilResult(
            final_draft=winner_text,
            winning_agent_id=_find_winner_id(current_drafts, winner_text),
            rounds_run=1,
            consensus_reached=True,
            transcript=transcript,
        ))

    # -----------------------------------------------------------------------
    # Rounds 2–N — iteration
    # -----------------------------------------------------------------------
    for round_num in range(2, max_rounds + 1):
        agent_names = ", ".join(a["id"] for a in live_agents)
        if event_cb:
            event_cb(
                "round_start",
                {
                    "round_num": round_num,
                    "active_count": len(live_agents),
                    "active_ids": [a["id"] for a in live_agents],
                    "threshold": threshold,
                },
            )
        emit(f"\n[Round {round_num}] Iteration… ({len(live_agents)} active: {agent_names})")
        summary = RoundSummary(round_num=round_num)
        t0 = time.monotonic()

        live_ids = {a["id"] for a in live_agents}

        # Snapshot current state so all agents see the same starting point
        snapshot = dict(current_drafts)

        def _make_prompt(agent: dict, rn: int = round_num) -> str:
            return iteration_prompt_fn(
                task=task,
                agent_id=agent["id"],
                round_num=rn,
                max_rounds=max_rounds,
                current_drafts=snapshot,
                multifile=multifile,
                context_files=context_files,
            )

        actions: dict[str, dict] = {}

        def _dispatch(agent: dict) -> tuple[str, dict]:
            prompt = _make_prompt(agent)
            return agent["id"], _run_agent_iteration(agent, prompt, live_ids, emit, retry_prompt_fn=retry_prompt_fn)

        with ThreadPoolExecutor(max_workers=len(live_agents)) as ex:
            futures = [ex.submit(_dispatch, a) for a in live_agents]
            for fut in as_completed(futures):
                aid, action = fut.result()
                actions[aid] = action

        # Build adoption map for cycle detection
        round_adopts = {
            aid: act["target"]
            for aid, act in actions.items()
            if act["action"] == "adopt"
        }

        # Apply all actions; handle drops first so rescaling happens before convergence check
        dropped_this_round = []
        for agent in list(live_agents):
            aid = agent["id"]
            action = actions[aid]

            elapsed = action.get("_elapsed", 0)
            stats = action.get("_stats", {})
            t_str = f"{elapsed:.1f}s"
            if "tps" in stats:
                t_str += f"  {stats['tps']} tok/s"
                if "ttft_ms" in stats:
                    t_str += f"  ttft {stats['ttft_ms']}ms"
            elif "tokens" in stats and elapsed:
                t_str += f"  ~{round(stats['tokens'] / elapsed, 1)} tok/s"

            if action["action"] == "drop":
                live_agents.remove(agent)
                del current_drafts[aid]
                dropped_this_round.append(aid)
                summary.events.append(f"{aid}: DROPPED (unreachable)")
                emit(f"  [{aid}] DROPPED — removed from council")
                if event_cb:
                    event_cb("agent_iteration", {"agent_id": aid, "action": "drop"})

            elif action["action"] == "revise":
                new = action["draft"]
                if multifile:
                    new = _merge_multifile(snapshot.get(aid, ""), new)
                current_drafts[aid] = new
                summary.events.append(f"{aid}: revised")
                emit(f"  [{aid}] REVISED ({len(new):,} chars)  [{t_str}]")
                if event_cb:
                    event_cb("agent_iteration", {"agent_id": aid, "action": "revise", "chars": len(new)})

            elif action["action"] == "adopt":
                target = action["target"]
                if target in snapshot:
                    current_drafts[aid] = _adoption_draft(aid, target, round_adopts, snapshot)
                    summary.events.append(f"{aid}: adopted {target}")
                    emit(f"  [{aid}] ADOPTED [{target}]  [{t_str}]")
                    if event_cb:
                        event_cb("agent_iteration", {"agent_id": aid, "action": "adopt", "target": target})
                else:
                    summary.events.append(f"{aid}: kept (adopted target was dropped)")
                    emit(f"  [{aid}] KEPT (target [{target}] was dropped)  [{t_str}]")
                    if event_cb:
                        event_cb(
                            "agent_iteration",
                            {"agent_id": aid, "action": "keep", "detail": f"Target {target} dropped"},
                        )

            else:  # keep
                summary.events.append(f"{aid}: kept draft (parse failure)")
                emit(f"  [{aid}] KEPT (parse failure)  [{t_str}]")
                if event_cb:
                    event_cb("agent_iteration", {"agent_id": aid, "action": "keep", "detail": "Parse failure"})

        if dropped_this_round:
            if not live_agents:
                emit("\nAll agents dropped. Aborting.")
                transcript.append(summary)
                return CouncilResult(
                    final_draft="",
                    winning_agent_id="",
                    rounds_run=round_num,
                    consensus_reached=False,
                    transcript=transcript,
                )
            threshold = _effective_threshold(config_threshold, config_total, len(live_agents))
            emit(f"  Council shrank to {len(live_agents)} agent(s) — threshold rescaled to {threshold}/{len(live_agents)}")

        summary.distinct_count = _distinct_count(current_drafts)
        transcript.append(summary)
        emit(f"  Distinct proposals: {summary.distinct_count}  [{time.monotonic() - t0:.1f}s]")
        if event_cb:
            event_cb(
                "round_complete",
                {
                    "round_num": round_num,
                    "distinct_count": summary.distinct_count,
                    "active_count": len(live_agents),
                    "threshold": threshold,
                },
            )

        # Convergence check
        if winner_text := _convergence_winner(current_drafts, threshold):
            emit(f"\nConsensus reached after round {round_num}!")
            return _maybe_review(CouncilResult(
                final_draft=winner_text,
                winning_agent_id=_find_winner_id(current_drafts, winner_text),
                rounds_run=round_num,
                consensus_reached=True,
                transcript=transcript,
            ))

    # -----------------------------------------------------------------------
    # Round limit reached — majority fallback
    # -----------------------------------------------------------------------
    emit(f"\nRound limit ({max_rounds}) reached. Applying majority fallback…")

    counts = collections.Counter(current_drafts.values())
    ranked = counts.most_common()

    if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
        winner_text = ranked[0][0]
        winner_id = _find_winner_id(current_drafts, winner_text)
        emit(f"Majority winner: [{winner_id}] ({ranked[0][1]}/{active_count} votes)")
        return _maybe_review(CouncilResult(
            final_draft=winner_text,
            winning_agent_id=winner_id,
            rounds_run=max_rounds,
            consensus_reached=False,
            transcript=transcript,
        ))

    # True tie — surface to human
    emit("True tie — no majority. Surfacing options to user.")
    unique_drafts: dict[str, str] = {}
    for aid, draft in current_drafts.items():
        if draft not in unique_drafts.values():
            unique_drafts[aid] = draft

    return CouncilResult(
        final_draft="",
        winning_agent_id="",
        rounds_run=max_rounds,
        consensus_reached=False,
        transcript=transcript,
        tiebreak_options=unique_drafts,
    )
