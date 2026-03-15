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
from .supervisor import supervise
from .types import (
    BatchResult,
    ForecastResult,
    Question,
    RunSummary,
    load_questions,
    sample_batch,
    PROJECT_ROOT,
)

DEFAULT_CONCURRENCY = int(os.environ.get("FORECAST_CONCURRENCY", "5"))


def _load_memory() -> str:
    path = PROJECT_ROOT / "memory.md"
    if path.exists():
        return path.read_text()
    return ""


async def _forecast_one(
    question: Question,
    memory: str,
    platt_params=None,
    timeout_seconds: int = 300,
    handler=None,
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
        asyncio.ensure_future(run_agent(question, agent_id=i, memory=memory, handler=_handler))
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
    supervisor_output = await supervise(question, traces, memory=memory, handler=_handler)

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

    # Compute Brier scores
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
        data={"raw_probability": raw_prob, "calibrated_probability": calibrated},
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
    memory = memory if memory is not None else _load_memory()
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
    memory = _load_memory()

    with RichHandler() as handler:
        for batch_num in range(start_batch, start_batch + num_batches):
            batch = sample_batch(questions, batch_num, batch_size=batch_size)
            results = await backtest_batch(batch, batch_num, memory, handler=handler, concurrency=concurrency)

            # Summary
            raw_briers = [r.brier_raw for r in results if r.brier_raw is not None]
            avg_brier = sum(raw_briers) / len(raw_briers) if raw_briers else float("nan")

    # Print summary after Live stops so it's visible
    print(f"\nBatch avg Brier (raw): {avg_brier:.4f}")


async def run_eval(batch_id: int, results: list[ForecastResult] | None = None) -> BatchResult | None:
    """Run eval + postmortem + autoresearcher on a completed batch."""
    from .eval import evaluate_batch
    from .postmortem import run_postmortems, update_memory
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

    # Collect all historical raw probs and outcomes for Platt fitting
    traces_dir = PROJECT_ROOT / "logs" / "traces"
    all_raw_probs = []
    all_outcomes = []
    for trace_file in sorted(traces_dir.glob("*.json")):
        with open(trace_file) as f:
            data = json.load(f)
        r = ForecastResult.model_validate(data)
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
    print(f"Postmortems complete. {len(postmortems)} lessons extracted.")

    # Autoresearcher
    print("\nRunning autoresearcher...")
    changelog_entry = await run_autoresearcher(postmortems, batch_id, results)
    if changelog_entry:
        status = "ACCEPTED" if changelog_entry.accepted else "REJECTED"
        print(f"Autoresearcher: {status} — {changelog_entry.change_description}")
    else:
        print("Autoresearcher: No changes proposed.")

    return batch_result


async def run_continuous(duration_seconds: int = 7200, start_batch: int = 0, batch_size: int = 12, concurrency: int = DEFAULT_CONCURRENCY) -> None:
    """Run backtest → eval loop continuously for the specified duration."""
    from .ui import RichHandler

    t_start = time.monotonic()
    deadline = t_start + duration_seconds
    start_time = datetime.now(timezone.utc).isoformat()

    questions = load_questions()
    max_batch = len(questions) // batch_size - 1
    memory = _load_memory()

    completed_batches: list[int] = []
    last_batch_result: BatchResult | None = None

    with RichHandler() as handler:
        for batch_id in range(start_batch, max_batch + 1):
            if time.monotonic() >= deadline:
                break

            batch = sample_batch(questions, batch_id, batch_size=batch_size)

            try:
                results = await backtest_batch(batch, batch_id, memory, deadline=deadline, handler=handler, concurrency=concurrency)
            except KeyboardInterrupt:
                break

            if not results:
                break

            try:
                batch_result = await run_eval(batch_id, results=results)
            except KeyboardInterrupt:
                completed_batches.append(batch_id)
                break

            if batch_result:
                last_batch_result = batch_result

            completed_batches.append(batch_id)

            # Reload memory (postmortem may have updated it)
            memory = _load_memory()

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


def main():
    """CLI entry point: python -m autoforecast <start_batch> <num_batches> [--eval <batch_id>] [--run [hours]] [--batch-size N] [--concurrency N] [--plot]"""
    batch_size = 12
    if "--batch-size" in sys.argv:
        idx = sys.argv.index("--batch-size")
        batch_size = int(sys.argv[idx + 1])

    concurrency = DEFAULT_CONCURRENCY
    if "--concurrency" in sys.argv:
        idx = sys.argv.index("--concurrency")
        concurrency = int(sys.argv[idx + 1])

    if "--plot" in sys.argv:
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
        asyncio.run(run_continuous(duration_seconds=int(hours * 3600), start_batch=start_batch, batch_size=batch_size, concurrency=concurrency))
    elif "--eval" in sys.argv:
        idx = sys.argv.index("--eval")
        batch_id = int(sys.argv[idx + 1])
        asyncio.run(run_eval(batch_id))
    else:
        start_batch = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        num_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        asyncio.run(run_backtest(start_batch, num_batches, batch_size=batch_size, concurrency=concurrency))


if __name__ == "__main__":
    main()
