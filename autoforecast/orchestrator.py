"""Orchestrator: backtest, eval, and continuous run modes."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .agent import run_agent
from .calibrate import apply_platt, load_params
from .events import EventType, NullHandler, PipelineEvent
from .supervisor import supervise, supervise_cluster
from .types import (
    AgentTrace,
    BatchResult,
    ClusterResearchOutput,
    ForecastResult,
    Question,
    RunSummary,
    load_questions,
    load_testing_questions,
    make_ask_question,
    sample_batch,
    sample_random_batch,
    PROJECT_ROOT,
)
from .utils import load_memory

DEFAULT_CONCURRENCY = int(os.environ.get("FORECAST_CONCURRENCY", "5"))


def _safe_tag(tag: str) -> str:
    """Sanitize a cluster tag for use as a filename component."""
    import re as _re
    return _re.sub(r'[^\w\-]', '_', tag)


async def _run_agents_only(
    question: Question,
    memory: str,
    handler=None,
    live_mode: bool = False,
    shared_research: ClusterResearchOutput | None = None,
    timeout_seconds: int = 300,
) -> list[AgentTrace]:
    """Run 3 agents for a question without supervisor or Platt.

    Used for clustered questions where the cluster supervisor handles
    reconciliation across all questions at once.
    """
    _handler = handler or NullHandler()

    await _handler.handle(PipelineEvent(
        event_type=EventType.QUESTION_START,
        question_id=question.question_id, question_title=question.title,
    ))

    agent_tasks = [
        asyncio.ensure_future(run_agent(question, agent_id=i, memory=memory, handler=_handler, live_mode=live_mode, shared_research=shared_research))
        for i in range(3)
    ]
    done, pending = await asyncio.wait(agent_tasks, timeout=timeout_seconds)

    for task in pending:
        task.cancel()

    traces = []
    for task in done:
        try:
            traces.append(task.result())
        except Exception as e:
            print(f"    ⚠ Agent failed: {type(e).__name__}: {e}")

    if not traces:
        raise RuntimeError(f"All agents failed or timed out for {question.title[:50]}")

    if pending:
        print(f"    ⚠ {len(pending)} agent(s) timed out, proceeding with {len(traces)}")

    return traces


async def _forecast_one(
    question: Question,
    memory: str,
    platt_params=None,
    timeout_seconds: int = 300,
    handler=None,
    live_mode: bool = False,
    shared_research: ClusterResearchOutput | None = None,
) -> ForecastResult:
    """Forecast a single question: 3 parallel agents → supervisor → Platt.

    timeout_seconds caps wall time for agents (default 5 min).
    If some agents time out, supervisor works with whatever traces completed.
    """
    _handler = handler or NullHandler()

    await _handler.handle(PipelineEvent(
        event_type=EventType.QUESTION_START,
        question_id=question.question_id, question_title=question.title,
    ))

    # Run 3 agents in parallel; collect whatever finishes within the timeout
    agent_tasks = [
        asyncio.ensure_future(run_agent(question, agent_id=i, memory=memory, handler=_handler, live_mode=live_mode, shared_research=shared_research))
        for i in range(3)
    ]
    done, pending = await asyncio.wait(agent_tasks, timeout=timeout_seconds)

    # Cancel stragglers
    for task in pending:
        task.cancel()

    # Collect completed traces (skip any that raised)
    traces = []
    for task in done:
        try:
            traces.append(task.result())
        except Exception as e:
            print(f"    ⚠ Agent failed: {type(e).__name__}: {e}")

    if not traces:
        raise RuntimeError("All agents failed or timed out")

    if pending:
        print(f"    ⚠ {len(pending)} agent(s) timed out, proceeding with {len(traces)}")

    # Supervisor reconciliation with whatever traces we have
    supervisor_output = await supervise(question, traces, memory=memory, handler=_handler, live_mode=live_mode)

    raw_prob = supervisor_output.reconciled_probability

    # Apply Platt scaling if params exist
    calibrated = None
    if platt_params:
        calibrated = apply_platt(raw_prob, platt_params)
        await _handler.handle(PipelineEvent(
            event_type=EventType.CALIBRATION_DONE,
            question_id=question.question_id, question_title=question.title,
            data={"calibrated_probability": calibrated, "raw_probability": raw_prob},
        ))

    # Compute Brier scores (skip for live/unresolved questions)
    brier_raw = None
    brier_cal = None
    if not live_mode and question.outcome is not None:
        outcome = question.outcome
        brier_raw = (raw_prob - outcome) ** 2
        brier_cal = (calibrated - outcome) ** 2 if calibrated is not None else None

    result = ForecastResult(
        question=question,
        agent_traces=list(traces),
        supervisor=supervisor_output,
        raw_probability=raw_prob,
        calibrated_probability=calibrated,
        brier_raw=brier_raw,
        brier_calibrated=brier_cal,
    )

    await _handler.handle(PipelineEvent(
        event_type=EventType.QUESTION_DONE,
        question_id=question.question_id, question_title=question.title,
        data={"raw_probability": raw_prob, "calibrated_probability": calibrated, "brier_raw": brier_raw},
    ))

    return result


async def backtest_batch(
    questions: list[Question],
    batch_id: int,
    memory: str | None = None,
    deadline: float | None = None,
    handler=None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[ForecastResult]:
    """Run backtest on a batch of questions with bounded concurrency."""
    _handler = handler or NullHandler()
    memory = memory if memory is not None else load_memory()
    platt_params = load_params()

    traces_dir = PROJECT_ROOT / "logs" / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(concurrency)

    await _handler.handle(PipelineEvent(
        event_type=EventType.BATCH_PROGRESS,
        question_id=questions[0].question_id if questions else 0,
        question_title="",
        data={"current": 0, "total": len(questions), "batch_id": batch_id},
    ))

    async def _run_one(question: Question) -> ForecastResult | None:
        async with sem:
            if deadline and time.monotonic() >= deadline:
                return None
            try:
                result = await _forecast_one(question, memory, platt_params, handler=_handler)
            except Exception as e:
                print(f"    ✗ FAILED ({question.title[:50]}): {e}")
                return None

            # Save trace
            trace_path = traces_dir / f"{batch_id}_{question.question_id}.json"
            with open(trace_path, "w") as f:
                json.dump(result.model_dump(), f, indent=2, default=str)

            return result

    raw_results = await asyncio.gather(*[_run_one(q) for q in questions])
    return [r for r in raw_results if r is not None]


async def run_backtest(start_batch: int = 0, num_batches: int = 1, batch_size: int = 12, concurrency: int = DEFAULT_CONCURRENCY) -> None:
    """Main entry point: run backtest on specified batches."""
    from .ui import RichHandler

    questions = load_questions()
    memory = load_memory()

    with RichHandler() as handler:
        for batch_num in range(start_batch, start_batch + num_batches):
            batch = sample_batch(questions, batch_num, batch_size=batch_size)
            results = await backtest_batch(batch, batch_num, memory, handler=handler, concurrency=concurrency)

            # Summary
            raw_briers = [r.brier_raw for r in results if r.brier_raw is not None]
            avg_brier = sum(raw_briers) / len(raw_briers) if raw_briers else float("nan")

    # Print summary after Live stops so it's visible
    print(f"\nBatch avg Brier (raw): {avg_brier:.4f}")


async def run_eval(batch_id: int, results: list[ForecastResult] | None = None, ab_size: int | None = None) -> BatchResult | None:
    """Run eval + postmortem + autoresearcher on a completed batch."""
    from .eval import evaluate_batch
    from .postmortem import run_postmortems, update_memory, consolidate_memory
    from .autoresearcher import run_autoresearcher

    # Load traces from disk if not provided directly
    if results is None:
        traces_dir = PROJECT_ROOT / "logs" / "traces"
        results = []
        for trace_file in sorted(traces_dir.glob(f"{batch_id}_*.json")):
            with open(trace_file) as f:
                data = json.load(f)
            results.append(ForecastResult.model_validate(data))

    if not results:
        print(f"No traces found for batch {batch_id}")
        return None

    # Collect all historical raw probs and outcomes for Platt fitting.
    # Index current batch by question_id to avoid re-reading those trace files.
    current_batch_ids = {r.question.question_id for r in results}
    all_raw_probs = [r.raw_probability for r in results]
    all_outcomes = [r.question.outcome for r in results]

    traces_dir = PROJECT_ROOT / "logs" / "traces"
    for trace_file in sorted(traces_dir.glob("*.json")):
        with open(trace_file) as f:
            data = json.load(f)
        r = ForecastResult.model_validate(data)
        if r.question.question_id not in current_batch_ids:
            all_raw_probs.append(r.raw_probability)
            all_outcomes.append(r.question.outcome)

    print(f"\n=== Eval for Batch {batch_id} ({len(results)} questions) ===")

    # Evaluate
    batch_result = await evaluate_batch(results, batch_id, all_raw_probs, all_outcomes)
    print(f"Pipeline Brier: {batch_result.brier_pipeline:.4f}")
    print(f"Community Brier: {batch_result.brier_community:.4f}")
    print(f"Naive LLM Brier: {batch_result.brier_naive:.4f}")
    if batch_result.contamination_flags:
        print(f"Contamination flags: {batch_result.contamination_flags}")

    # Postmortems
    print("\nRunning postmortems...")
    postmortems = await run_postmortems(results)
    update_memory(postmortems)
    await consolidate_memory()
    print(f"Postmortems complete. {len(postmortems)} lessons extracted.")

    # Autoresearcher
    print("\nRunning autoresearcher...")
    changelog_entry = await run_autoresearcher(postmortems, batch_id, results, ab_size=ab_size)
    if changelog_entry:
        status = "ACCEPTED" if changelog_entry.accepted else "REJECTED"
        print(f"Autoresearcher: {status} — {changelog_entry.change_description}")
    else:
        print("Autoresearcher: No changes proposed.")

    return batch_result


async def run_continuous(duration_seconds: int = 7200, start_batch: int = 0, batch_size: int = 12, concurrency: int = DEFAULT_CONCURRENCY, ab_size: int | None = None) -> None:
    """Run backtest → eval loop continuously for the specified duration.

    Phase 1 (initial): Run one deterministic batch → eval (triggers A/B testing).
    Phase 2 (random_batches): Loop random batches until timer runs out.
    """
    from .ui import RichHandler

    t_start = time.monotonic()
    deadline = t_start + duration_seconds
    start_time = datetime.now(timezone.utc).isoformat()

    questions = load_questions()
    memory = load_memory()

    completed_batches: list[int] = []
    forecasted_qids: set[int] = set()
    last_batch_result: BatchResult | None = None

    with RichHandler(deadline=deadline) as handler:
        # --- Phase 1: Initial batch + eval (triggers autoresearcher A/B) ---
        await handler.handle(PipelineEvent(
            event_type=EventType.PHASE_CHANGE,
            data={"phase": "initial", "batches_completed": 0},
        ))

        batch = sample_batch(questions, start_batch, batch_size=batch_size)

        try:
            results = await backtest_batch(batch, start_batch, memory, deadline=deadline, handler=handler, concurrency=concurrency)
        except KeyboardInterrupt:
            results = []

        if results:
            forecasted_qids.update(r.question.question_id for r in results)

            await handler.handle(PipelineEvent(
                event_type=EventType.PHASE_CHANGE,
                data={"phase": "eval", "batches_completed": 0},
            ))

            try:
                batch_result = await run_eval(start_batch, results=results, ab_size=ab_size)
            except KeyboardInterrupt:
                batch_result = None

            if batch_result:
                last_batch_result = batch_result
            completed_batches.append(start_batch)
            memory = load_memory()

        # --- Phase 2: Random batches until deadline ---
        await handler.handle(PipelineEvent(
            event_type=EventType.PHASE_CHANGE,
            data={"phase": "random_batches", "batches_completed": len(completed_batches)},
        ))

        batch_counter = start_batch + 1
        while time.monotonic() < deadline:
            batch = sample_random_batch(questions, exclude_ids=forecasted_qids)

            try:
                results = await backtest_batch(
                    batch, batch_counter, memory, deadline=deadline, handler=handler, concurrency=concurrency,
                )
            except KeyboardInterrupt:
                break

            if not results:
                break

            forecasted_qids.update(r.question.question_id for r in results)

            try:
                batch_result = await run_eval(batch_counter, results=results, ab_size=ab_size)
            except KeyboardInterrupt:
                completed_batches.append(batch_counter)
                break

            if batch_result:
                last_batch_result = batch_result

            completed_batches.append(batch_counter)
            batch_counter += 1
            memory = load_memory()

            await handler.handle(PipelineEvent(
                event_type=EventType.PHASE_CHANGE,
                data={"phase": "random_batches", "batches_completed": len(completed_batches)},
            ))

    # Save run summary
    end_time = datetime.now(timezone.utc).isoformat()
    total_duration = time.monotonic() - t_start

    # Generate plot
    plot_path = None
    try:
        from .plot import plot_brier_scores
        p = plot_brier_scores()
        plot_path = str(p)
        print(f"\nPlot saved to {p}")
    except Exception as e:
        print(f"\nCould not generate plot: {e}")

    summary = RunSummary(
        start_time=start_time,
        end_time=end_time,
        duration_seconds=total_duration,
        batches_completed=completed_batches,
        final_brier_pipeline=last_batch_result.brier_pipeline if last_batch_result else None,
        final_brier_community=last_batch_result.brier_community if last_batch_result else None,
        final_brier_naive=last_batch_result.brier_naive if last_batch_result else None,
        plot_path=plot_path,
    )

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = logs_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary.model_dump(), f, indent=2)

    print(f"\nRun complete: {len(completed_batches)} batches in {total_duration/60:.1f}m")
    if last_batch_result:
        print(f"Final Brier — Pipeline: {last_batch_result.brier_pipeline:.4f}, "
              f"Community: {last_batch_result.brier_community:.4f}, "
              f"Naive: {last_batch_result.brier_naive:.4f}")
    print(f"Summary saved to {summary_path}")


async def run_test(
    concurrency: int = DEFAULT_CONCURRENCY,
    max_questions: int | None = None,
) -> list[ForecastResult]:
    """Run pipeline on unresolved testing questions (live mode).

    Saves each result incrementally to logs/test_traces/ and resumes
    from existing results on restart.
    """
    import webbrowser
    from .ui import RichHandler

    from .cluster import cluster_questions, select_with_clusters
    from .cluster_research import run_cluster_research

    questions = load_testing_questions()

    memory = load_memory()
    platt_params = load_params()

    traces_dir = PROJECT_ROOT / "logs" / "test_traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: skip already-completed question IDs
    existing_results: list[ForecastResult] = []
    existing_ids: set[str] = set()
    for trace_file in traces_dir.glob("*.json"):
        if trace_file.name.startswith("cluster_"):
            continue
        with open(trace_file) as f:
            data = json.load(f)
        result = ForecastResult.model_validate(data)
        existing_results.append(result)
        existing_ids.add(result.question.id)

    remaining = [q for q in questions if q.id not in existing_ids]

    # Cluster questions before --max slicing to keep clusters whole
    clusters, singletons = cluster_questions(remaining)

    if max_questions:
        remaining = select_with_clusters(clusters, singletons, max_questions)
        # Re-cluster the selected subset
        clusters, singletons = cluster_questions(remaining)

    if existing_ids:
        print(f"Resuming: {len(existing_ids)} already done, {len(remaining)} remaining")
    if clusters:
        cluster_sizes = {tag: len(qs) for tag, qs in clusters.items()}
        print(f"Clusters: {cluster_sizes}, Singletons: {len(singletons)}")

    sem = asyncio.Semaphore(concurrency)
    new_results: list[ForecastResult] = []

    # Phase 1: Cluster research (load cached or run new)
    cluster_research_map: dict[str, ClusterResearchOutput] = {}

    # Load existing cluster research from disk
    for tag in list(clusters.keys()):
        cached = traces_dir / f"cluster_{_safe_tag(tag)}.json"
        if cached.exists():
            with open(cached) as f:
                cluster_research_map[tag] = ClusterResearchOutput.model_validate(json.load(f))

    with RichHandler() as handler:
        # Run cluster research for any clusters not yet cached
        uncached_clusters = {tag: qs for tag, qs in clusters.items() if tag not in cluster_research_map}
        if uncached_clusters:
            print(f"Running cluster research for {len(uncached_clusters)} clusters...")

            async def _research_cluster(tag: str, qs: list[Question]) -> tuple[str, ClusterResearchOutput]:
                async with sem:
                    result = await run_cluster_research(tag, qs, memory=memory, handler=handler, live_mode=True)
                    # Save cluster trace
                    path = traces_dir / f"cluster_{_safe_tag(tag)}.json"
                    with open(path, "w") as f:
                        json.dump(result.model_dump(), f, indent=2, default=str)
                    return tag, result

            cluster_results = await asyncio.gather(*[
                _research_cluster(t, qs) for t, qs in uncached_clusters.items()
            ])
            cluster_research_map.update(dict(cluster_results))

        # Build question → shared_research mapping
        q_to_research: dict[str, ClusterResearchOutput] = {}
        for tag, qs in clusters.items():
            for q in qs:
                q_to_research[q.id] = cluster_research_map[tag]

        # Phase 2a: Run agents only for clustered questions (no supervisor yet)
        cluster_agent_traces: dict[str, list[AgentTrace]] = {}  # question.id → traces

        async def _run_agents_for_cluster_q(question: Question) -> tuple[str, list[AgentTrace]] | None:
            async with sem:
                try:
                    traces = await _run_agents_only(
                        question, memory, handler=handler, live_mode=True,
                        shared_research=q_to_research.get(question.id),
                    )
                    return question.id, traces
                except Exception as e:
                    print(f"    ✗ FAILED agents ({question.title[:50]}): {e}")
                    return None

        clustered_questions = [q for qs in clusters.values() for q in qs]
        if clustered_questions:
            agent_results_raw = await asyncio.gather(*[
                _run_agents_for_cluster_q(q) for q in clustered_questions
            ])
            for item in agent_results_raw:
                if item is not None:
                    cluster_agent_traces[item[0]] = item[1]

        # Phase 2b: Cluster supervisor for each cluster
        for tag, qs in clusters.items():
            # Only include questions whose agents succeeded
            qs_with_traces = [q for q in qs if q.id in cluster_agent_traces]
            if not qs_with_traces:
                continue

            traces_for_cluster = {q.id: cluster_agent_traces[q.id] for q in qs_with_traces}

            try:
                cluster_sv = await supervise_cluster(
                    tag, qs_with_traces, traces_for_cluster,
                    shared_research=cluster_research_map.get(tag),
                    memory=memory, handler=handler, live_mode=True,
                )
            except Exception as e:
                print(f"    ✗ Cluster supervisor failed ({tag}): {e}")
                # Fallback: run per-question supervisor for this cluster
                for q in qs_with_traces:
                    try:
                        sv_output = await supervise(q, traces_for_cluster[q.id], memory=memory, handler=handler, live_mode=True)
                        raw_prob = sv_output.reconciled_probability
                        calibrated = apply_platt(raw_prob, platt_params) if platt_params else None
                        result = ForecastResult(
                            question=q, agent_traces=traces_for_cluster[q.id],
                            supervisor=sv_output, raw_probability=raw_prob,
                            calibrated_probability=calibrated,
                        )
                        trace_path = traces_dir / f"{q.id}.json"
                        with open(trace_path, "w") as f:
                            json.dump(result.model_dump(), f, indent=2, default=str)
                        new_results.append(result)
                    except Exception as e2:
                        print(f"    ✗ Fallback supervisor failed ({q.title[:50]}): {e2}")
                continue

            # Phase 2c: Platt + assemble ForecastResult per question
            print(f"  Cluster {tag}: probability sum = {cluster_sv.probability_sum:.3f}")
            for q in qs_with_traces:
                sv_output = cluster_sv.question_results.get(q.id)
                if sv_output is None:
                    print(f"    ⚠ No cluster supervisor output for {q.id}")
                    continue

                raw_prob = sv_output.reconciled_probability
                calibrated = apply_platt(raw_prob, platt_params) if platt_params else None

                result = ForecastResult(
                    question=q,
                    agent_traces=cluster_agent_traces[q.id],
                    supervisor=sv_output,
                    raw_probability=raw_prob,
                    calibrated_probability=calibrated,
                )

                await handler.handle(PipelineEvent(
                    event_type=EventType.QUESTION_DONE,
                    question_id=q.question_id, question_title=q.title,
                    data={"raw_probability": raw_prob, "calibrated_probability": calibrated},
                ))

                trace_path = traces_dir / f"{q.id}.json"
                with open(trace_path, "w") as f:
                    json.dump(result.model_dump(), f, indent=2, default=str)

                new_results.append(result)

        # Phase 2d: Singletons use normal _forecast_one()
        async def _run_singleton(question: Question) -> ForecastResult | None:
            async with sem:
                try:
                    result = await _forecast_one(
                        question, memory, platt_params, handler=handler, live_mode=True,
                        shared_research=q_to_research.get(question.id),
                    )
                except Exception as e:
                    print(f"    ✗ FAILED ({question.title[:50]}): {e}")
                    return None

                trace_path = traces_dir / f"{question.id}.json"
                with open(trace_path, "w") as f:
                    json.dump(result.model_dump(), f, indent=2, default=str)

                return result

        if singletons:
            raw = await asyncio.gather(*[_run_singleton(q) for q in singletons])
            new_results.extend(r for r in raw if r is not None)

    all_results = existing_results + new_results

    # Write lightweight summary
    summary = []
    for r in all_results:
        summary.append({
            "question_id": r.question.id,
            "title": r.question.title,
            "source": r.question.source,
            "url": r.question.url,
            "market_price": r.question.community_prediction_final,
            "raw_probability": r.raw_probability,
            "calibrated_probability": r.calibrated_probability,
        })

    logs_dir = PROJECT_ROOT / "logs"
    summary_path = logs_dir / "test_results.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{len(all_results)} results saved to {summary_path}")

    # Generate interactive plot and open in browser
    try:
        from .test_plot import generate_test_plot
        plot_path = generate_test_plot()
        print(f"Plot saved to {plot_path}")
        webbrowser.open(f"file://{plot_path}")
    except Exception as e:
        print(f"Could not generate plot: {e}")

    return all_results


def _print_ask_summary(result: ForecastResult) -> None:
    """Print a polished Rich summary panel after --ask completes."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    # Final probability: prefer calibrated
    final_prob = result.calibrated_probability if result.calibrated_probability is not None else result.raw_probability
    prob_pct = f"{final_prob:.1%}"

    # Header
    header = Text()
    header.append(result.question.title, style="bold")
    header.append("\n\n")
    header.append("Final Probability: ", style="dim")
    header.append(prob_pct, style="bold green" if final_prob >= 0.5 else "bold red")

    # Agent breakdown table
    agent_table = Table(title="Agent Breakdown", show_header=True, header_style="bold cyan", expand=True)
    agent_table.add_column("Agent", justify="center")
    agent_table.add_column("Model", justify="center")
    agent_table.add_column("Probability", justify="center")

    for trace in result.agent_traces:
        agent_table.add_row(
            f"Agent {trace.agent_id}",
            trace.model_id,
            f"{trace.raw_probability:.1%}",
        )

    # Supervisor section
    sv = result.supervisor
    disagreements = "\n".join(f"  • {d}" for d in sv.disagreements[:3]) if sv.disagreements else "  None"
    reconciliation = sv.reconciliation_reasoning[:300]
    if len(sv.reconciliation_reasoning) > 300:
        reconciliation += "…"

    supervisor_text = (
        f"[bold]Supervisor[/bold]\n"
        f"Reconciled: {sv.reconciled_probability:.1%}\n"
        f"Disagreements:\n{disagreements}\n\n"
        f"Reasoning: {reconciliation}"
    )

    # Calibration
    calibration_text = ""
    if result.calibrated_probability is not None:
        calibration_text = f"\n\n[bold]Calibration[/bold]: {result.raw_probability:.1%} → {result.calibrated_probability:.1%} (Platt scaling)"

    # Assemble panel body
    body_parts = [supervisor_text, calibration_text]

    console.print()
    console.print(Panel(header, title="Forecast Result", border_style="bright_blue"))
    console.print(agent_table)
    console.print(Panel("\n".join(body_parts), border_style="dim"))


async def run_ask(title: str) -> None:
    """Run the full forecasting pipeline on a single ad-hoc question."""
    from .ui import RichHandler

    question = make_ask_question(title)
    memory = load_memory()
    platt_params = load_params()

    with RichHandler() as handler:
        result = await _forecast_one(
            question, memory, platt_params,
            handler=handler, live_mode=True,
        )

    _print_ask_summary(result)


def main():
    """CLI entry point: python -m autoforecast <start_batch> <num_batches> [--eval <batch_id>] [--run [hours]] [--test] [--ask [question]] [--batch-size N] [--concurrency N] [--plot]"""
    batch_size = 12
    if "--batch-size" in sys.argv:
        idx = sys.argv.index("--batch-size")
        batch_size = int(sys.argv[idx + 1])

    concurrency = DEFAULT_CONCURRENCY
    if "--concurrency" in sys.argv:
        idx = sys.argv.index("--concurrency")
        concurrency = int(sys.argv[idx + 1])

    ab_size: int | None = None
    if "--ab-size" in sys.argv:
        idx = sys.argv.index("--ab-size")
        ab_size = int(sys.argv[idx + 1])

    max_questions: int | None = None
    if "--max" in sys.argv:
        idx = sys.argv.index("--max")
        max_questions = int(sys.argv[idx + 1])

    if "--ask" in sys.argv:
        idx = sys.argv.index("--ask")
        # Use next arg as title if it exists and isn't a flag
        if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-"):
            title = sys.argv[idx + 1]
        else:
            from rich.prompt import Prompt
            title = Prompt.ask("Enter your forecasting question")
        asyncio.run(run_ask(title))
    elif "--test" in sys.argv:
        asyncio.run(run_test(concurrency=concurrency, max_questions=max_questions))
    elif "--plot" in sys.argv:
        from .plot import plot_brier_scores, plot_domain_breakdown
        p1 = plot_brier_scores()
        print(f"Brier scores plot: {p1}")
        p2 = plot_domain_breakdown()
        print(f"Domain breakdown plot: {p2}")
    elif "--run" in sys.argv:
        idx = sys.argv.index("--run")
        hours = float(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-") else 2.0
        start_batch = 0
        if "--start" in sys.argv:
            start_idx = sys.argv.index("--start")
            start_batch = int(sys.argv[start_idx + 1])
        asyncio.run(run_continuous(duration_seconds=int(hours * 3600), start_batch=start_batch, batch_size=batch_size, concurrency=concurrency, ab_size=ab_size))
    elif "--eval" in sys.argv:
        idx = sys.argv.index("--eval")
        batch_id = int(sys.argv[idx + 1])
        asyncio.run(run_eval(batch_id, ab_size=ab_size))
    else:
        start_batch = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        num_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        asyncio.run(run_backtest(start_batch, num_batches, batch_size=batch_size, concurrency=concurrency))


if __name__ == "__main__":
    main()
