"""Live terminal dashboard for council runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class AgentView:
    model: str
    agent_type: str
    status: str = "pending"
    stage: str = "waiting"
    detail: str = ""
    executor: str = ""  # non-empty when a different agent is running this ownership slot
    updated_at: float = field(default_factory=time.monotonic)


@dataclass
class RoundView:
    round_num: int
    active_count: int
    distinct_count: int = 0
    threshold: str = ""
    status: str = "running"


class CouncilDashboard:
    """Rich-powered live dashboard for a council run."""

    def __init__(self, console, task: str, agents: list[dict], max_rounds: int) -> None:
        self.console = console
        self.task = task
        self.max_rounds = max_rounds
        self.phase = "Initializing"
        self.threshold = "?"
        self.current_round = 0
        self.consensus_state = "Pending"
        self.reviewed_by: Optional[str] = None
        self.writer_id: Optional[str] = None
        self.final_status = "Running"
        self.winner_id = ""
        self.final_excerpt = ""
        self.agent_order = [agent["id"] for agent in agents]
        self.agents = {
            agent["id"]: AgentView(
                model=agent.get("model", ""),
                agent_type=agent.get("type", "ollama"),
            )
            for agent in agents
        }
        self.rounds: list[RoundView] = []
        self.events: list[str] = []
        self._live: Live | None = None
        self.is_pipeline: bool = False
        self.pipeline_phases: list[str] = []
        self.pipeline_reassignments: int = 0
        self.pipeline_fix_retries: int = 0

    def start(self) -> None:
        self._live = Live(self.render(), console=self.console, refresh_per_second=8, transient=False)
        self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.stop()
            self._live = None

    def log(self, message: str) -> None:
        clean = " ".join(message.strip().split())
        if clean:
            self.events.append(clean)
            self.events = self.events[-8:]
            self.refresh()

    def handle_event(self, event: str, data: dict) -> None:
        if event == "rollcall_complete":
            statuses = data["statuses"]
            for aid in self.agent_order:
                status = statuses.get(aid, "unreachable")
                view = self.agents[aid]
                view.status = status
                view.stage = "rollcall"
                view.detail = status
                view.updated_at = time.monotonic()
            self.phase = "Rollcall complete"

        elif event == "council_start":
            self.phase = "Council running"
            self.threshold = f"{data['threshold']}/{data['active_count']}"
            self.consensus_state = "In progress"
            active_ids = set(data["active_ids"])
            for aid, view in self.agents.items():
                if aid in active_ids:
                    view.status = "active"
                    view.stage = "ready"
                    view.detail = "Awaiting round work"
                elif view.status == "ready":
                    view.status = "inactive"
                    view.stage = "waiting"
                    view.detail = "Not active this run"

        elif event == "round_start":
            self.phase = f"Round {data['round_num']}"
            self.current_round = data["round_num"]
            self.rounds.append(
                RoundView(
                    round_num=data["round_num"],
                    active_count=data["active_count"],
                    threshold=f"{data['threshold']}/{data['active_count']}",
                )
            )
            self.rounds = self.rounds[-6:]
            active_ids = set(data["active_ids"])
            for aid, view in self.agents.items():
                if aid in active_ids and view.status != "dropped":
                    view.stage = "thinking"
                    view.detail = f"Round {data['round_num']} in progress"

        elif event == "agent_round1_result":
            aid = data["agent_id"]
            view = self.agents[aid]
            if data["result"] == "proposed":
                view.status = "active"
                view.stage = "proposed"
                view.detail = f"{data['chars']} chars"
            else:
                view.status = "dropped"
                view.stage = "failed"
                view.detail = data["reason"]

        elif event == "agent_iteration":
            aid = data["agent_id"]
            view = self.agents[aid]
            action = data["action"]
            view.updated_at = time.monotonic()
            if action == "drop":
                view.status = "dropped"
                view.stage = "dropped"
                view.detail = "Unreachable"
            elif action == "revise":
                view.status = "active"
                view.stage = "revised"
                view.detail = f"{data.get('chars', 0)} chars"
            elif action == "adopt":
                view.status = "active"
                view.stage = "adopted"
                view.detail = f"Adopted {data.get('target', '?')}"
            else:
                view.status = "active"
                view.stage = "kept"
                view.detail = data.get("detail", "Kept current draft")

        elif event == "round_complete":
            if self.rounds:
                current = self.rounds[-1]
                current.status = "complete"
                current.distinct_count = data["distinct_count"]
                current.active_count = data["active_count"]
                current.threshold = f"{data['threshold']}/{data['active_count']}"
            self.consensus_state = f"{data['distinct_count']} distinct draft(s)"

        elif event == "review_started":
            self.phase = "Review round"

        elif event == "review_complete":
            self.phase = "Review complete"

        elif event == "fix_applied":
            self.phase = "Fix round"
            self.reviewed_by = data["agent_id"]

        elif event == "writer_started":
            self.phase = "Writer pass"
            self.writer_id = data["agent_id"]

        elif event == "result":
            self.phase = "Complete"
            self.final_status = data["status"]
            self.winner_id = data.get("winner_id", "")
            self.final_excerpt = data.get("final_excerpt", "")
            self.consensus_state = data["status"]
            if data.get("reviewed_by"):
                self.reviewed_by = data["reviewed_by"]

        elif event == "pipeline_phase":
            self.is_pipeline = True
            label = data.get("label", data.get("phase", "Pipeline"))
            self.phase = label
            self.consensus_state = data.get("phase", "pipeline")
            self.pipeline_phases.append(label)

        elif event == "pipeline_assigned":
            for item in data.get("parts", []):
                aid = item.get("agent")
                if aid in self.agents:
                    self.agents[aid].status = "active"
                    self.agents[aid].stage = "assigned"
                    self.agents[aid].detail = ", ".join(item.get("files", []))
                    self.agents[aid].executor = ""

        elif event == "pipeline_part_done":
            actor = data.get("agent", "")
            phase = data.get("phase", "done")
            files = data.get("files", [])
            if phase in ("implement", "fix"):
                owner = data.get("owner", actor)
                if owner in self.agents:
                    view = self.agents[owner]
                    view.status = "active"
                    view.stage = phase
                    if files:
                        view.detail = ", ".join(files)
                    if actor != owner:
                        view.executor = actor
                        if phase == "implement":
                            self.pipeline_reassignments += 1
                        else:
                            self.pipeline_fix_retries += 1
                    else:
                        view.executor = ""
            elif phase == "review":
                target = data.get("target", actor)
                if target in self.agents:
                    view = self.agents[target]
                    view.status = "active"
                    view.stage = "reviewing"
                    view.detail = f"by {actor}"

        self.refresh()

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self.render())

    def render(self):
        top = Table.grid(expand=True)
        top.add_column(ratio=2)
        top.add_column(ratio=1)
        top.add_row(self._summary_panel(), self._status_panel())

        bottom = Table.grid(expand=True)
        bottom.add_column(ratio=3)
        bottom.add_column(ratio=2)
        bottom.add_row(self._agents_panel(), self._rounds_panel())

        return Group(top, bottom, self._events_panel(), self._final_panel())

    def _summary_panel(self) -> Panel:
        body = Table.grid(padding=(0, 1))
        body.add_column(style="bold cyan", width=10)
        body.add_column()
        body.add_row("Task", self.task[:120] + ("..." if len(self.task) > 120 else ""))
        body.add_row("Phase", self.phase)
        body.add_row("Round", f"{self.current_round}/{self.max_rounds}" if self.current_round else f"0/{self.max_rounds}")
        body.add_row("Threshold", self.threshold)
        return Panel(body, title="Council Dashboard", border_style="cyan")

    def _status_panel(self) -> Panel:
        body = Table.grid(padding=(0, 1))
        body.add_column(style="bold green", width=12)
        body.add_column()
        body.add_row("Convergence", self.consensus_state)
        body.add_row("Winner", self.winner_id or "-")
        body.add_row("Reviewed By", self.reviewed_by or "-")
        body.add_row("Writer", self.writer_id or "-")
        return Panel(body, title="Run Status", border_style="green")

    def _agents_panel(self) -> Panel:
        table = Table(expand=True, box=None, pad_edge=False)
        table.add_column("Agent", style="cyan", min_width=10)
        table.add_column("State", min_width=10)
        table.add_column("Stage", min_width=10)
        if self.is_pipeline:
            table.add_column("Executor", min_width=10)
        table.add_column("Detail")
        for aid in self.agent_order:
            view = self.agents[aid]
            row: list[str] = [aid, view.status, view.stage]
            if self.is_pipeline:
                row.append(f"[yellow]{view.executor}[/yellow]" if view.executor else "-")
            row.append(view.detail or f"{view.model} ({view.agent_type})")
            table.add_row(*row)
        return Panel(table, title="Agents", border_style="blue")

    def _rounds_panel(self) -> Panel:
        if self.is_pipeline:
            return self._pipeline_panel()
        table = Table(expand=True, box=None, pad_edge=False)
        table.add_column("Round", min_width=6)
        table.add_column("Active", min_width=6)
        table.add_column("Distinct", min_width=8)
        table.add_column("Threshold", min_width=9)
        table.add_column("State", min_width=8)
        if not self.rounds:
            table.add_row("-", "-", "-", "-", "waiting")
        else:
            for round_view in self.rounds:
                table.add_row(
                    str(round_view.round_num),
                    str(round_view.active_count),
                    str(round_view.distinct_count or "-"),
                    round_view.threshold or "-",
                    round_view.status,
                )
        return Panel(table, title="Rounds", border_style="magenta")

    def _pipeline_panel(self) -> Panel:
        body = Table.grid(padding=(0, 1))
        body.add_column(style="bold magenta", width=14)
        body.add_column()
        body.add_row("Reassigned", str(self.pipeline_reassignments) if self.pipeline_reassignments else "-")
        body.add_row("Fix Retries", str(self.pipeline_fix_retries) if self.pipeline_fix_retries else "-")
        if self.pipeline_phases:
            body.add_row("", "")
            for i, ph in enumerate(self.pipeline_phases):
                marker = "[green]▶[/green]" if i == len(self.pipeline_phases) - 1 else "[dim]✓[/dim]"
                body.add_row("" if i else "Phases", f"{marker} {ph}")
        return Panel(body, title="Pipeline", border_style="magenta")

    def _events_panel(self) -> Panel:
        if not self.events:
            text = Text("No events yet.", style="dim")
        else:
            text = Text("\n".join(self.events))
        return Panel(text, title="Recent Events", border_style="yellow")

    def _final_panel(self) -> Panel:
        excerpt = self.final_excerpt or "Final result will appear here."
        return Panel(excerpt, title=f"Final Result ({self.final_status})", border_style="green")
