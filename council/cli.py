"""CLI entry point for the council harness."""

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from .agent import rollcall as _rollcall, cli_can_write
from .contracts import contract_text
from .dashboard import CouncilDashboard
from .loop import run_council, run_writer
from .pipeline import PipelineResult, run_pipeline

console = Console()
_UNSETTABLE_AGENT_FIELDS = ("api_key", "api_key_env", "timeout", "max_tokens", "num_ctx", "no_think")


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _save_config(path: Path, cfg: dict) -> None:
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copyfile(path, backup)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)


def _ensure_agent_list(cfg: dict) -> list[dict]:
    agents = cfg.get("agents")
    if agents is None:
        cfg["agents"] = []
        return cfg["agents"]
    if not isinstance(agents, list):
        raise click.ClickException("Config 'agents' must be a list.")
    return agents


def _find_agent(agents: list[dict], agent_id: str) -> dict | None:
    for agent in agents:
        if agent.get("id") == agent_id:
            return agent
    return None


def _clean_agent_values(agent: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in agent.items() if value is not None}


def _agent_updates(
    model: str | None,
    url: str | None,
    agent_type: str | None,
    api_key: str | None,
    api_key_env: str | None,
    timeout: int | None,
    max_tokens: int | None,
    num_ctx: int | None,
    no_think: bool | None,
) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    if model is not None:
        updates["model"] = model
    if url is not None:
        updates["url"] = url
    if agent_type is not None:
        updates["type"] = agent_type
    if api_key is not None:
        updates["api_key"] = api_key
    if api_key_env is not None:
        updates["api_key_env"] = api_key_env
    if timeout is not None:
        updates["timeout"] = timeout
    if max_tokens is not None:
        updates["max_tokens"] = max_tokens
    if num_ctx is not None:
        updates["num_ctx"] = num_ctx
    if no_think is not None:
        updates["no_think"] = no_think
    return updates


def _print_agent_config(agent: dict) -> None:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("Field", style="cyan", min_width=12)
    table.add_column("Value")
    for key in ("id", "model", "url", "type", "api_key", "api_key_env", "timeout", "max_tokens", "num_ctx", "no_think", "can_write"):
        if key in agent:
            table.add_row(key, str(agent[key]))
    console.print(table)


def _print_assignments_table(assignments: list) -> None:
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Agent", style="cyan", min_width=10)
    table.add_column("Owned Files")
    for assignment in assignments:
        table.add_row(str(assignment.agent_id), ", ".join(assignment.owned_files))
    console.print(table)


_STATUS_DISPLAY = {
    "ready":       "[green]READY[/green]",
    "no_model":    "[yellow]NO MODEL[/yellow]",
    "unreachable": "[red]UNREACHABLE[/red]",
}


def _print_rollcall_table(agents: list[dict], statuses: dict[str, str]) -> None:
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Agent", style="cyan", min_width=10)
    table.add_column("Model", min_width=16)
    table.add_column("URL", min_width=28)
    table.add_column("Type", min_width=8)
    table.add_column("Status", min_width=12)

    for a in agents:
        s = statuses.get(a["id"], "unreachable")
        table.add_row(a["id"], a.get("model", "—"), a.get("url", "—"), a.get("type", "ollama"), _STATUS_DISPLAY[s])

    console.print(table)


def _strip_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences (```lang ... ```)."""
    text = text.strip()
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _parse_files(draft: str) -> list[tuple[str, str]]:
    """Extract (relative_path, content) pairs from a ### FILE: block response.
    Falls back to ## filename.ext heading style if no ### FILE: markers found."""
    pattern = re.compile(r"^### FILE:\s*(.+)$", re.MULTILINE)
    matches = list(pattern.finditer(draft))
    if not matches:
        # Fallback: ## filename.ext (model used markdown headings instead)
        pattern = re.compile(r"^##\s+([\w./\-]+\.\w+)$", re.MULTILINE)
        matches = list(pattern.finditer(draft))
    if not matches:
        return []

    files = []
    for i, match in enumerate(matches):
        path = match.group(1).strip()
        content_start = match.end()
        if content_start < len(draft) and draft[content_start] == "\n":
            content_start += 1
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(draft)
        content = _strip_fences(draft[content_start:content_end].rstrip())
        files.append((path, content))

    return files


def _safe_path(rel_path: str, output_dir: Path) -> Path | None:
    """Return resolved dest path, or None if it would escape output_dir."""
    p = Path(rel_path)
    if p.is_absolute():
        return None
    parts = p.parts
    if any(part in ("", "..", ".") or part.startswith("/") for part in parts):
        return None
    dest = (output_dir / p).resolve()
    try:
        dest.relative_to(output_dir.resolve())
    except ValueError:
        return None
    return dest


def _write_files(files: list[tuple[str, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files:
        dest = _safe_path(rel_path, output_dir)
        if dest is None:
            console.print(f"[yellow]Skipping unsafe path: {rel_path!r}[/yellow]")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content)
        console.print(f"  [dim]wrote[/dim] {dest}")


def _save_transcript(path: str, task: str, events: list[str], final_draft: str) -> None:
    with open(path, "w") as f:
        f.write(f"TASK\n{'=' * 60}\n{task}\n\n")
        f.write(f"TRANSCRIPT\n{'=' * 60}\n")
        for line in events:
            f.write(line + "\n")
        if final_draft:
            f.write(f"\nFINAL IMPLEMENTATION\n{'=' * 60}\n{final_draft}\n")


def _artifact_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return name or "item"


def _write_pipeline_artifacts(output_dir: Path, result: PipelineResult) -> Path:
    artifact_dir = output_dir / ".council-harness"
    drafts_dir = artifact_dir / "drafts"
    reviews_dir = artifact_dir / "reviews"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    drafts_dir.mkdir(parents=True, exist_ok=True)
    reviews_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "mode": "contract_parts",
        "architecture_winner_id": result.architecture_result.winning_agent_id,
        "architecture_rounds": result.rounds_run,
        "assignments": [
            {"agent_id": assignment.agent_id, "owned_files": list(assignment.owned_files)}
            for assignment in result.assignments
        ],
        "drafts": [
            {
                "agent_id": draft.agent_id,
                "implemented_by": draft.implemented_by,
                "owned_files": list(draft.owned_files),
                "failed": draft.failed,
            }
            for draft in result.drafts
        ],
        "reviews": [
            {
                "reviewer_id": review.reviewer_id,
                "target_agent_id": review.target_agent_id,
                "target_files": list(review.target_files),
                "failed": review.failed,
            }
            for review in result.reviews
        ],
        "merge": {
            "files": sorted(result.merge_result.files),
            "unresolved": list(result.merge_result.unresolved),
        },
    }

    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (artifact_dir / "contract.json").write_text(result.contract.to_json())
    (artifact_dir / "contract.txt").write_text(contract_text(result.contract))
    (artifact_dir / "integration_review.txt").write_text(result.integration_feedback)
    (artifact_dir / "merged_output.txt").write_text(result.merge_result.merged_text)
    validation_lines = result.syntax_errors + result.interface_warnings
    (artifact_dir / "validation.txt").write_text("\n".join(validation_lines) if validation_lines else "OK")

    for draft in result.drafts:
        stem = _artifact_name(draft.agent_id)
        (drafts_dir / f"{stem}.json").write_text(json.dumps({
            "agent_id": draft.agent_id,
            "implemented_by": draft.implemented_by,
            "owned_files": list(draft.owned_files),
            "failed": draft.failed,
        }, indent=2))
        (drafts_dir / f"{stem}.txt").write_text(draft.implementation)

    for review in result.reviews:
        stem = _artifact_name(review.target_agent_id)
        (reviews_dir / f"{stem}.json").write_text(json.dumps({
            "reviewer_id": review.reviewer_id,
            "target_agent_id": review.target_agent_id,
            "target_files": list(review.target_files),
            "failed": review.failed,
        }, indent=2))
        (reviews_dir / f"{stem}.txt").write_text(review.feedback)

    return artifact_dir


_config_option = click.option(
    "--config", "-c",
    default="config.yaml",
    show_default=True,
    type=click.Path(exists=True, path_type=Path),
    help="Council config YAML.",
)


@click.group()
def main() -> None:
    """Council Harness — collaborative coding with local LLM agents."""


@main.group()
def agent() -> None:
    """Manage council agent configuration."""


@agent.command("list")
@_config_option
def agent_list(config: Path) -> None:
    """List configured agents."""
    cfg = _load_config(config) or {}
    agents = _ensure_agent_list(cfg)
    if not agents:
        console.print("[yellow]No agents configured.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Agent", style="cyan", min_width=10)
    table.add_column("Model", min_width=16)
    table.add_column("URL", min_width=28)
    table.add_column("Type", min_width=8)
    table.add_column("Auth", min_width=12)
    for entry in agents:
        auth = "api_key_env" if entry.get("api_key_env") else ("api_key" if entry.get("api_key") else "-")
        table.add_row(
            str(entry.get("id", "")),
            str(entry.get("model", "")),
            str(entry.get("url", "")),
            str(entry.get("type", "ollama")),
            auth,
        )
    console.print(table)


@agent.command("add")
@_config_option
@click.option("--id", "agent_id", required=True, help="Unique agent id.")
@click.option("--model", required=True, help="Model name passed to the endpoint.")
@click.option("--url", required=True, help="Base URL for the agent endpoint.")
@click.option(
    "--type",
    "agent_type",
    type=click.Choice(["ollama", "openai"], case_sensitive=False),
    default="ollama",
    show_default=True,
    help="Agent API type.",
)
@click.option("--api-key", default=None, help="Inline API key for OpenAI-compatible endpoints.")
@click.option("--api-key-env", default=None, help="Environment variable that stores the API key.")
@click.option("--timeout", type=int, default=None, help="Per-call timeout in seconds.")
@click.option("--max-tokens", type=int, default=None, help="Completion token limit.")
@click.option("--num-ctx", type=int, default=None, help="Context window for Ollama agents.")
@click.option("--no-think/--think", default=None, help="Disable or enable model thinking output.")
def agent_add(
    config: Path,
    agent_id: str,
    model: str,
    url: str,
    agent_type: str,
    api_key: str | None,
    api_key_env: str | None,
    timeout: int | None,
    max_tokens: int | None,
    num_ctx: int | None,
    no_think: bool | None,
) -> None:
    """Add a new agent to the config."""
    cfg = _load_config(config) or {}
    agents = _ensure_agent_list(cfg)
    if _find_agent(agents, agent_id):
        raise click.ClickException(f"Agent {agent_id!r} already exists.")

    new_agent = _clean_agent_values(
        {
            "id": agent_id,
            "model": model,
            "url": url,
            "type": agent_type.lower(),
            **_agent_updates(model=None, url=None, agent_type=None, api_key=api_key, api_key_env=api_key_env, timeout=timeout, max_tokens=max_tokens, num_ctx=num_ctx, no_think=no_think),
        }
    )
    agents.append(new_agent)
    _save_config(config, cfg)

    console.print(f"[green]Added agent[/green] [cyan]{agent_id}[/cyan] to {config}")
    _print_agent_config(new_agent)


@agent.command("update")
@_config_option
@click.argument("agent_id")
@click.option("--new-id", default=None, help="Rename the agent id.")
@click.option("--model", default=None, help="Update the model name.")
@click.option("--url", default=None, help="Update the base URL.")
@click.option(
    "--type",
    "agent_type",
    type=click.Choice(["ollama", "openai"], case_sensitive=False),
    default=None,
    help="Update the agent API type.",
)
@click.option("--api-key", default=None, help="Set an inline API key.")
@click.option("--api-key-env", default=None, help="Set the API key environment variable.")
@click.option("--timeout", type=int, default=None, help="Set per-call timeout in seconds.")
@click.option("--max-tokens", type=int, default=None, help="Set completion token limit.")
@click.option("--num-ctx", type=int, default=None, help="Set Ollama context window.")
@click.option("--no-think/--think", default=None, help="Disable or enable model thinking output.")
@click.option(
    "--unset",
    "unset_fields",
    multiple=True,
    type=click.Choice(_UNSETTABLE_AGENT_FIELDS, case_sensitive=False),
    help="Remove an optional field from the agent config.",
)
def agent_update(
    config: Path,
    agent_id: str,
    new_id: str | None,
    model: str | None,
    url: str | None,
    agent_type: str | None,
    api_key: str | None,
    api_key_env: str | None,
    timeout: int | None,
    max_tokens: int | None,
    num_ctx: int | None,
    no_think: bool | None,
    unset_fields: tuple[str, ...],
) -> None:
    """Update an existing agent in the config."""
    cfg = _load_config(config) or {}
    agents = _ensure_agent_list(cfg)
    agent_cfg = _find_agent(agents, agent_id)
    if not agent_cfg:
        raise click.ClickException(f"Agent {agent_id!r} was not found.")

    if new_id and new_id != agent_id and _find_agent(agents, new_id):
        raise click.ClickException(f"Agent {new_id!r} already exists.")

    updates = _agent_updates(
        model=model,
        url=url,
        agent_type=agent_type.lower() if agent_type else None,
        api_key=api_key,
        api_key_env=api_key_env,
        timeout=timeout,
        max_tokens=max_tokens,
        num_ctx=num_ctx,
        no_think=no_think,
    )

    changed = False
    if new_id:
        agent_cfg["id"] = new_id
        changed = True
    for key, value in updates.items():
        agent_cfg[key] = value
        changed = True
    for field_name in unset_fields:
        if field_name in agent_cfg:
            del agent_cfg[field_name]
            changed = True

    if not changed:
        raise click.ClickException("No changes requested. Provide at least one update option.")

    _save_config(config, cfg)

    console.print(f"[green]Updated agent[/green] [cyan]{agent_cfg['id']}[/cyan] in {config}")
    _print_agent_config(agent_cfg)


@main.command()
@_config_option
def rollcall(config: Path) -> None:
    """Ping all configured agents and report who is reachable."""
    cfg = _load_config(config)
    agents: list[dict] = cfg.get("agents", [])

    if not agents:
        console.print("[red]No agents defined in config.[/red]")
        sys.exit(1)

    console.print(Rule("[bold cyan]Rollcall[/bold cyan]"))
    console.print("Pinging agents…\n")

    active, missing, statuses = _rollcall(agents)
    _print_rollcall_table(agents, statuses)
    console.print()

    ready_names = ", ".join(a["id"] for a in active)
    console.print(f"[green]{len(active)}/{len(agents)} agent(s) ready[/green]: {ready_names}")
    if not active:
        sys.exit(1)


@main.command()
@click.argument("task")
@_config_option
@click.option(
    "--log", "-l",
    default=None,
    type=click.Path(path_type=Path),
    help="Save full transcript to this file.",
)
@click.option(
    "--consensus", "-k",
    default=None,
    type=int,
    help="Override consensus_threshold from config.",
)
@click.option(
    "--rounds", "-r",
    default=None,
    type=int,
    help="Override max rounds from config.",
)
@click.option(
    "--max-tokens", "-m",
    default=None,
    type=int,
    help="Override max_tokens for all agents.",
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(path_type=Path),
    help="Write the final implementation to this file (single-file tasks).",
)
@click.option(
    "--output-dir", "-d",
    default=None,
    type=click.Path(path_type=Path),
    help="Write multi-file output to this directory (agents use ### FILE: blocks).",
)
@click.option(
    "--file", "-f",
    "files",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Inject a file into agent context. Repeatable: -f A.java -f B.java",
)
@click.option(
    "--dashboard",
    is_flag=True,
    help="Show a live terminal dashboard for agent status, rounds, and convergence.",
)
@click.option(
    "--mode",
    type=click.Choice(["holistic", "contract_parts"], case_sensitive=False),
    default=None,
    help="Execution mode. Defaults to config mode or holistic.",
)
def run(task: str, config: Path, log: Path | None, consensus: int | None, rounds: int | None, max_tokens: int | None, output: Path | None, output_dir: Path | None, files: tuple[Path, ...], dashboard: bool, mode: str | None) -> None:
    """Run TASK through the council of local LLM agents."""
    cfg = _load_config(config)
    agents: list[dict] = cfg.get("agents", [])
    selected_mode = (mode or cfg.get("mode", "holistic")).lower()
    max_rounds = rounds if rounds is not None else cfg.get("rounds", {}).get("max", 5)
    ui = CouncilDashboard(console, task, agents, max_rounds=max_rounds) if dashboard else None

    if not agents:
        console.print("[red]No agents defined in config.[/red]")
        sys.exit(1)

    console.print(Rule("[bold cyan]Council Harness[/bold cyan]"))
    console.print(f"[dim]Task:[/dim] {task[:120]}{'…' if len(task) > 120 else ''}\n")

    console.print("[bold]Rollcall[/bold] — pinging agents…")
    active, missing, statuses = _rollcall(agents)
    if ui:
        ui.start()
        ui.handle_event("rollcall_complete", {"statuses": statuses})
    else:
        console.print()
        _print_rollcall_table(agents, statuses)
        console.print()

    if not active:
        if ui:
            ui.stop()
        console.print("[red]No agents responded. Exiting.[/red]")
        sys.exit(1)

    agent_names = ", ".join(a["id"] for a in active)
    console.print(f"[green]{len(active)} agent(s) active[/green] ({agent_names}) — starting council.\n")

    events: list[str] = []

    def emit(msg: str) -> None:
        if ui:
            ui.log(msg)
        else:
            console.print(msg, markup=False, highlight=False)
        events.append(msg)

    if consensus is not None:
        cfg.setdefault("rounds", {})["consensus_threshold"] = consensus
    if rounds is not None:
        cfg.setdefault("rounds", {})["max"] = rounds
    if max_tokens is not None:
        for a in agents:
            a["max_tokens"] = max_tokens

    context_files = [(f.name, f.read_text()) for f in files] if files else None
    if context_files:
        names = ", ".join(n for n, _ in context_files)
        if ui:
            emit(f"[context] {names}")
        else:
            console.print(f"[dim]Context files:[/dim] {names}\n")

    multifile = output_dir is not None
    try:
        if selected_mode == "contract_parts":
            result = run_pipeline(
                task=task,
                active_agents=active,
                config=cfg,
                emit=emit,
                context_files=context_files,
                event_cb=ui.handle_event if ui else None,
            )
        else:
            while True:
                result = run_council(
                    task=task,
                    active_agents=active,
                    config=cfg,
                    emit=emit,
                    multifile=multifile,
                    context_files=context_files,
                    event_cb=ui.handle_event if ui else None,
                )

                console.print()
                console.print(Rule("[bold]Result[/bold]"))

                if not result.tiebreak_options:
                    break

                if ui:
                    ui.stop()
                console.print(
                    "[yellow]True tie after round limit — no majority. "
                    "Please choose a proposal or restart:[/yellow]\n"
                )
                options = list(result.tiebreak_options.items())
                for i, (aid, draft) in enumerate(options, 1):
                    console.print(Panel(escape(draft), title=f"[{i}] {aid}", border_style="yellow"))

                restart = False
                while True:
                    raw = click.prompt(
                        f"Enter proposal number (1–{len(options)}) or 'r' to restart from round 1",
                        type=str,
                    ).strip()
                    if raw.lower() == "r":
                        console.print("[yellow]Restarting from round 1…[/yellow]\n")
                        restart = True
                        break
                    if raw.isdigit() and 1 <= int(raw) <= len(options):
                        chosen_id, chosen_draft = options[int(raw) - 1]
                        result.final_draft = chosen_draft
                        result.winning_agent_id = chosen_id
                        break
                    console.print(f"[red]Please enter a number between 1 and {len(options)}, or 'r' to restart.[/red]")

                if restart:
                    if ui:
                        ui.start()
                    continue
                break

        writer_wrote_files = False

        if isinstance(result, PipelineResult):
            writer_cfg = cfg.get("writer")
            if writer_cfg:
                result.final_draft = run_writer(
                    task=task,
                    agreed_draft=result.final_draft,
                    writer_cfg=writer_cfg,
                    emit=emit,
                    multifile=True,
                    event_cb=ui.handle_event if ui else None,
                    output_dir=output_dir,
                )
                writer_wrote_files = bool(output_dir) and cli_can_write(writer_cfg)
                events.append(f"[writer] final transcription by [{writer_cfg['id']}]")

            if ui:
                ui.handle_event(
                    "result",
                    {
                        "status": "CONTRACT_PARTS",
                        "winner_id": result.architecture_result.winning_agent_id,
                        "final_excerpt": _strip_fences(result.final_draft)[:400],
                    },
                )
                ui.stop()

            console.print("Status : [green]CONTRACT_PARTS[/green]")
            console.print(f"Architecture Winner : [cyan]{result.architecture_result.winning_agent_id}[/cyan]")
            console.print(f"Architecture Rounds : {result.rounds_run}")
            console.print()
            console.print(Panel(result.contract.to_json(), title="[bold blue]Contract JSON[/bold blue]", border_style="blue"))
            console.print()
            console.print(Panel(escape(contract_text(result.contract)), title="[bold cyan]Agreed Contract[/bold cyan]", border_style="cyan"))
            console.print()
            _print_assignments_table(result.assignments)
            console.print()
            console.print(Panel(escape(result.integration_feedback), title="[bold yellow]Integration Review[/bold yellow]", border_style="yellow"))
            console.print()
            if result.syntax_errors or result.interface_warnings:
                validation_lines = []
                for err in result.syntax_errors:
                    validation_lines.append(f"[red][SYNTAX][/red] {escape(err)}")
                for warn in result.interface_warnings:
                    validation_lines.append(f"[yellow][INTERFACE][/yellow] {escape(warn)}")
                console.print(Panel("\n".join(validation_lines), title="[bold red]Validation Issues[/bold red]", border_style="red"))
            else:
                console.print(Panel("[green]All files pass syntax and interface checks[/green]", title="Validation", border_style="green"))
            console.print()
            console.print(Panel(
                escape(result.final_draft),
                title="[bold green]Final Implementation[/bold green]",
                border_style="green",
            ))
        else:
            if cfg.get("review_round", True) and result.tiebreak_options and result.final_draft:
                from .loop import _run_review_phase
                fixed, fixer_id = _run_review_phase(
                    active,
                    result.final_draft,
                    multifile,
                    emit,
                    event_cb=ui.handle_event if ui else None,
                    winner_id=result.winning_agent_id,
                )
                result.final_draft = fixed
                result.reviewed_by = fixer_id

            writer_cfg = cfg.get("writer")
            if writer_cfg:
                result.final_draft = run_writer(
                    task=task,
                    agreed_draft=result.final_draft,
                    writer_cfg=writer_cfg,
                    emit=emit,
                    multifile=multifile,
                    event_cb=ui.handle_event if ui else None,
                    output_dir=output_dir,
                )
                writer_wrote_files = bool(output_dir) and cli_can_write(writer_cfg)
                events.append(f"[writer] final transcription by [{writer_cfg['id']}]")

            if ui:
                ui.handle_event(
                    "result",
                    {
                        "status": "CONSENSUS" if result.consensus_reached else "MAJORITY",
                        "winner_id": result.winning_agent_id,
                        "reviewed_by": result.reviewed_by,
                        "final_excerpt": _strip_fences(result.final_draft)[:400],
                    },
                )
                ui.stop()

            status_str = "[green]CONSENSUS[/green]" if result.consensus_reached else "[yellow]MAJORITY[/yellow]"
            console.print(f"Status : {status_str}")
            console.print(f"Winner : [cyan]{result.winning_agent_id}[/cyan]")
            if result.reviewed_by:
                console.print(f"Reviewed+fixed by : [cyan]{result.reviewed_by}[/cyan]")
            console.print(f"Rounds : {result.rounds_run}")
            console.print()
            console.print(Panel(
                escape(result.final_draft),
                title="[bold green]Final Implementation[/bold green]",
                border_style="green",
            ))

        is_pipeline = isinstance(result, PipelineResult)

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            wrote = writer_wrote_files and any(
                p.name != "council.log" for p in output_dir.iterdir()
            )
            if writer_wrote_files and wrote:
                console.print(f"\n[bold]Writer [{cfg['writer']['id']}] wrote files directly into {output_dir}/[/bold]")
            else:
                if writer_wrote_files:
                    console.print("\n[yellow]Writer produced no files — falling back to parsing the council draft.[/yellow]")
                files = _parse_files(result.final_draft)
                if files:
                    console.print(f"\n[bold]Writing {len(files)} file(s) to {output_dir}/[/bold]")
                    _write_files(files, output_dir)
                else:
                    console.print("\n[yellow]No ### FILE: blocks found in output — writing raw draft to output_dir/output.txt[/yellow]")
                    (output_dir / "output.txt").write_text(_strip_fences(result.final_draft))

            # Always write a council.log inside the output directory
            auto_log = output_dir / "council.log"
            _save_transcript(str(auto_log), task, events, result.final_draft)
            console.print(f"[dim]Council log → {auto_log}[/dim]")
            if is_pipeline:
                artifact_dir = _write_pipeline_artifacts(output_dir, result)
                console.print(f"[dim]Contract artifacts → {artifact_dir}[/dim]")

        elif output:
            if is_pipeline:
                files = _parse_files(result.final_draft)
                if len(files) > 1:
                    raise click.ClickException(
                        "--output-dir is required for contract_parts results with multiple files."
                    )
                if len(files) == 1:
                    _, content = files[0]
                    output.write_text(content)
                else:
                    output.write_text(_strip_fences(result.final_draft))
            else:
                output.write_text(_strip_fences(result.final_draft))
            console.print(f"\n[dim]Implementation written → {output}[/dim]")

        if log:
            _save_transcript(str(log), task, events, result.final_draft)
            console.print(f"\n[dim]Transcript saved → {log}[/dim]")
    finally:
        if ui:
            ui.stop()
