"""Rich terminal dashboard for real-time agent monitoring."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .events import EventType, PipelineEvent, Stage

# Stage display order
STAGES = [Stage.DECOMPOSE, Stage.RESEARCH, Stage.BASE_RATE, Stage.INSIDE_VIEW, Stage.SYNTHESIZE]
STAGE_LABELS = {
    Stage.DECOMPOSE: "Decompose",
    Stage.RESEARCH: "Research",
    Stage.BASE_RATE: "Base Rate",
    Stage.INSIDE_VIEW: "Inside View",
    Stage.SYNTHESIZE: "Synthesize",
}

# Status icons
DONE = "[green]■[/green]"
ACTIVE = "[yellow]▸[/yellow]"
PENDING = "[dim]◻[/dim]"


class _AgentState:
    def __init__(self, agent_id: int) -> None:
        self.agent_id = agent_id
        self.current_stage: Stage | None = None
        self.completed_stages: set[Stage] = set()
        self.search_query: str = ""
        self.probability: float | None = None

    def stage_icon(self, stage: Stage) -> str:
        if stage in self.completed_stages:
            return DONE
        if stage == self.current_stage:
            return ACTIVE
        return PENDING


class _SupervisorState:
    def __init__(self) -> None:
        self.active = False
        self.done = False
        self.agent_probs: list[float] = []
        self.disagreements: list[str] = []
        self.search_queries: list[str] = []
        self.reconciled: float | None = None


@dataclass
class _QuestionState:
    question_id: int
    title: str
    agents: dict[int, _AgentState] = field(default_factory=dict)
    supervisor: _SupervisorState = field(default_factory=_SupervisorState)
    calibrated_prob: float | None = None
    done: bool = False
    raw_prob: float | None = None


class RichHandler:
    """Event handler that renders a live Rich dashboard with multi-question support."""

    def __init__(self, deadline: float | None = None) -> None:
        self._console = Console()
        self._live: Live | None = None
        self._questions: dict[int, _QuestionState] = {}
        self._active_question_id: int | None = None
        self._log: deque[str] = deque(maxlen=8)
        self._batch_progress: str = ""
        self._deadline: float | None = deadline
        self._phase: str = ""
        self._batches_completed: int = 0
        # Running Brier accumulator
        self._brier_sum: float = 0.0
        self._brier_count: int = 0
        # Cost tracking by provider
        self._costs: dict[str, float] = {}

    def start(self) -> None:
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    async def handle(self, event: PipelineEvent) -> None:
        self._process(event)
        if self._live:
            self._live.update(self._render())

    def _get_question(self, event: PipelineEvent) -> _QuestionState | None:
        """Get or create a QuestionState for the event's question_id."""
        qid = event.question_id
        if qid is None:
            return None
        return self._questions.get(qid)

    def _process(self, event: PipelineEvent) -> None:
        et = event.event_type

        if et == EventType.QUESTION_START:
            qid = event.question_id
            if qid is not None:
                self._questions[qid] = _QuestionState(
                    question_id=qid, title=event.question_title,
                )
                self._active_question_id = qid
            self._log.append(f"[bold]Starting:[/bold] {event.question_title[:70]}")

        elif et == EventType.QUESTION_DONE:
            qs = self._get_question(event)
            if qs:
                qs.done = True
                qs.raw_prob = event.data.get("raw_probability")
                cal = event.data.get("calibrated_probability")
                if cal is not None:
                    qs.calibrated_prob = cal
            raw = event.data.get("raw_probability")
            cal = event.data.get("calibrated_probability")
            # Track running Brier on raw probability
            brier = event.data.get("brier_raw")
            if brier is not None:
                self._brier_sum += brier
                self._brier_count += 1
            prob_str = f"raw={raw:.2%}" if raw is not None else ""
            if cal is not None:
                prob_str += f", cal={cal:.2%}"
            self._log.append(f"[bold green]Done:[/bold green] {event.question_title[:40]} {prob_str}")

        elif et == EventType.AGENT_STAGE_START:
            qs = self._get_question(event)
            aid = event.agent_id
            if qs and aid is not None:
                if aid not in qs.agents:
                    qs.agents[aid] = _AgentState(aid)
                qs.agents[aid].current_stage = event.stage
                label = STAGE_LABELS.get(event.stage, str(event.stage))
                self._log.append(f"Agent {aid} → {label}")

        elif et == EventType.AGENT_STAGE_DONE:
            qs = self._get_question(event)
            aid = event.agent_id
            if qs and aid is not None and aid in qs.agents:
                agent = qs.agents[aid]
                if event.stage:
                    agent.completed_stages.add(event.stage)
                agent.current_stage = None
                agent.search_query = ""
                if event.stage == Stage.SYNTHESIZE:
                    prob = event.data.get("final_probability")
                    if prob is not None:
                        agent.probability = prob

        elif et == EventType.SEARCH_START:
            qs = self._get_question(event)
            aid = event.agent_id
            query = event.data.get("query", "")
            if qs and aid is not None and aid in qs.agents:
                qs.agents[aid].search_query = query
            self._log.append(f"[dim]🔍 {query[:60]}[/dim]")

        elif et == EventType.SEARCH_DONE:
            qs = self._get_question(event)
            aid = event.agent_id
            if qs and aid is not None and aid in qs.agents:
                qs.agents[aid].search_query = ""

        elif et == EventType.SUPERVISOR_START:
            qs = self._get_question(event)
            if qs:
                qs.supervisor.active = True
                qs.supervisor.agent_probs = event.data.get("agent_probabilities", [])
            self._log.append("[bold cyan]Supervisor[/bold cyan] reconciling...")

        elif et == EventType.SUPERVISOR_SEARCH:
            qs = self._get_question(event)
            query = event.data.get("query", "")
            if qs:
                qs.supervisor.search_queries.append(query)
            self._log.append(f"[cyan]Supervisor 🔍 {query[:50]}[/cyan]")

        elif et == EventType.SUPERVISOR_DONE:
            qs = self._get_question(event)
            if qs:
                qs.supervisor.done = True
                qs.supervisor.active = False
                qs.supervisor.reconciled = event.data.get("reconciled_probability")
                qs.supervisor.disagreements = event.data.get("disagreements", [])
            prob = event.data.get("reconciled_probability")
            self._log.append(
                f"[bold cyan]Supervisor done:[/bold cyan] {prob:.2%}" if prob is not None else "Supervisor done"
            )

        elif et == EventType.CALIBRATION_DONE:
            qs = self._get_question(event)
            if qs:
                qs.calibrated_prob = event.data.get("calibrated_probability")

        elif et == EventType.API_COST:
            cost = event.data.get("cost_usd", 0.0)
            provider = event.data.get("provider", "unknown")
            self._costs[provider] = self._costs.get(provider, 0.0) + cost

        elif et == EventType.PHASE_CHANGE:
            self._phase = event.data.get("phase", "")
            self._batches_completed = event.data.get("batches_completed", 0)

        elif et == EventType.BATCH_PROGRESS:
            current = event.data.get("current", 0)
            total = event.data.get("total", 0)
            batch_id = event.data.get("batch_id", "?")
            self._batch_progress = f"Batch {batch_id}  [{current}/{total}]"

    def _format_remaining(self) -> str:
        if self._deadline is None:
            return ""
        remaining = max(0, self._deadline - time.monotonic())
        h, remainder = divmod(int(remaining), 3600)
        m, s = divmod(remainder, 60)
        if h > 0:
            return f"{h}h{m:02d}m"
        return f"{m}m{s:02d}s"

    def _phase_label(self) -> str:
        labels = {
            "initial": "Initial Batch",
            "eval": "Eval + A/B Testing",
            "random_batches": f"Random Batches ({self._batches_completed} done)",
        }
        return labels.get(self._phase, "")

    def _render(self) -> Group:
        parts = []

        # Header: phase + timer + batch progress + Brier + cost
        header_text = Text()

        # Phase and timer on first line
        phase = self._phase_label()
        if phase:
            header_text.append(phase, style="bold magenta")
            header_text.append("  ")
        remaining = self._format_remaining()
        if remaining:
            header_text.append(f"{remaining} remaining", style="bold")
            header_text.append("  ")

        # Batch progress
        if self._batch_progress:
            header_text.append(self._batch_progress, style="bold")
            header_text.append("  ")

        # Count completed
        n_done = sum(1 for qs in self._questions.values() if qs.done)
        n_total = len(self._questions)
        if n_total > 0:
            header_text.append(f"Completed: {n_done}/{n_total}", style="green" if n_done == n_total else "yellow")

        # Running Brier average
        if self._brier_count > 0:
            avg_brier = self._brier_sum / self._brier_count
            header_text.append(f"\nAvg Brier: {avg_brier:.4f} ({self._brier_count} questions)", style="bold")

        # Cost display
        total_cost = sum(self._costs.values())
        if total_cost > 0:
            cost_parts = [f"{name.capitalize()}: ${amt:.3f}" for name, amt in sorted(self._costs.items()) if amt > 0]
            header_text.append(f"\n$  {' | '.join(cost_parts)}  |  Total: ${total_cost:.3f}", style="bold yellow")
        if header_text.plain:
            parts.append(Panel(header_text, style="blue", padding=(0, 1)))

        # Active question: full 3-panel view
        active_qs = self._questions.get(self._active_question_id) if self._active_question_id else None
        if active_qs:
            parts.append(Text(f"  {active_qs.title[:80]}", style="bold italic"))

            agent_panels = []
            for aid in sorted(active_qs.agents.keys()):
                agent_panels.append(self._render_agent(active_qs.agents[aid]))
            if agent_panels:
                parts.append(Columns(agent_panels, equal=True, expand=True))

            sup = active_qs.supervisor
            if sup.active or sup.done:
                parts.append(self._render_supervisor(active_qs))

        # Other in-flight questions: compact status lines
        other_inflight = [
            qs for qid, qs in self._questions.items()
            if qid != self._active_question_id and not qs.done
        ]
        if other_inflight:
            lines = []
            for qs in other_inflight:
                agents_done = sum(1 for a in qs.agents.values() if Stage.SYNTHESIZE in a.completed_stages)
                sup_status = "done" if qs.supervisor.done else "active" if qs.supervisor.active else "pending"
                lines.append(f"Q{qs.question_id}: {agents_done}/{len(qs.agents)} agents | sup: {sup_status}  {qs.title[:40]}")
            parts.append(Panel(Text("\n".join(lines)), title="In-flight", border_style="yellow", padding=(0, 1)))

        # Completed questions: one-liner each
        completed = [qs for qid, qs in self._questions.items() if qs.done and qid != self._active_question_id]
        if completed:
            lines = []
            for qs in completed:
                prob = qs.calibrated_prob or qs.raw_prob
                prob_str = f"{prob:.2%}" if prob is not None else "?"
                lines.append(f"[green]■[/green] Q{qs.question_id}: {prob_str}  {qs.title[:50]}")
            parts.append(Panel(Text.from_markup("\n".join(lines)), title="Completed", border_style="green", padding=(0, 1)))

        # Activity log
        if self._log:
            log_text = Text()
            for i, entry in enumerate(self._log):
                if i > 0:
                    log_text.append("\n")
                log_text.append_text(Text.from_markup(entry))
            parts.append(Panel(log_text, title="Activity", border_style="dim", padding=(0, 1)))

        return Group(*parts) if parts else Group(Text("Waiting for events...", style="dim"))

    def _render_agent(self, agent: _AgentState) -> Panel:
        table = Table(show_header=False, show_edge=False, padding=(0, 1), expand=True)
        table.add_column("icon", width=2)
        table.add_column("stage")
        table.add_column("info", justify="right")

        for stage in STAGES:
            icon = agent.stage_icon(stage)
            label = STAGE_LABELS[stage]
            info = ""
            if stage == agent.current_stage and agent.search_query:
                info = f"[dim]{agent.search_query[:30]}[/dim]"
            elif stage == Stage.SYNTHESIZE and stage in agent.completed_stages and agent.probability is not None:
                info = f"[bold]{agent.probability:.2%}[/bold]"
            table.add_row(icon, label, info)

        title = f"Agent {agent.agent_id}"
        if agent.probability is not None and Stage.SYNTHESIZE in agent.completed_stages:
            title += f"  {agent.probability:.2%}"

        style = "green" if Stage.SYNTHESIZE in agent.completed_stages else "yellow" if agent.current_stage else "dim"
        return Panel(table, title=title, border_style=style, padding=(0, 0))

    def _render_supervisor(self, qs: _QuestionState) -> Panel:
        sup = qs.supervisor
        parts = []

        if sup.agent_probs:
            probs_str = "  ".join(f"A{i}: {p:.2%}" for i, p in enumerate(sup.agent_probs))
            parts.append(f"Agent probabilities: {probs_str}")

        if sup.disagreements:
            parts.append("Disagreements: " + "; ".join(d[:60] for d in sup.disagreements[:3]))

        if sup.search_queries:
            parts.append("Searches: " + ", ".join(q[:40] for q in sup.search_queries))

        if sup.reconciled is not None:
            parts.append(f"[bold]Reconciled: {sup.reconciled:.2%}[/bold]")

        if qs.calibrated_prob is not None:
            parts.append(f"[bold]Calibrated: {qs.calibrated_prob:.2%}[/bold]")

        status = "done" if sup.done else "active"
        icon = DONE if sup.done else ACTIVE
        content = Text.from_markup(f"{icon} Supervisor ({status})\n" + "\n".join(parts))

        style = "cyan" if sup.active else "green" if sup.done else "dim"
        return Panel(content, border_style=style, padding=(0, 1))
