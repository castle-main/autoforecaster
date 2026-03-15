"""Tests for parallelization changes."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from autoforecast.events import EventType, NullHandler, PipelineEvent, Stage
from autoforecast.types import ForecastResult, sample_batch
from autoforecast.ui import RichHandler, _QuestionState

from tests.conftest import make_forecast_result, make_question


# ---------------------------------------------------------------------------
# backtest_batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backtest_batch_runs_concurrently():
    """6 questions with concurrency=3 should complete in ~2 rounds, not 6."""
    questions = [make_question(question_id=i) for i in range(6)]

    async def slow_forecast(question, memory, platt_params, timeout_seconds=300, handler=None):
        await asyncio.sleep(0.1)
        return make_forecast_result(question_id=question.question_id)

    with patch("autoforecast.orchestrator._forecast_one", side_effect=slow_forecast), \
         patch("autoforecast.orchestrator.load_params", return_value=None):
        t0 = time.monotonic()
        results = await _import_backtest_batch()(questions, batch_id=0, memory="", concurrency=3)
        elapsed = time.monotonic() - t0

    assert len(results) == 6
    # 6 items at concurrency=3 → 2 rounds of 0.1s ≈ 0.2s, should be well under 0.5s
    assert elapsed < 0.5


@pytest.mark.asyncio
async def test_backtest_batch_respects_semaphore():
    """Max concurrent calls should not exceed the concurrency limit."""
    questions = [make_question(question_id=i) for i in range(6)]
    concurrent = 0
    max_concurrent = 0

    async def tracked_forecast(question, memory, platt_params, timeout_seconds=300, handler=None):
        nonlocal concurrent, max_concurrent
        concurrent += 1
        max_concurrent = max(max_concurrent, concurrent)
        await asyncio.sleep(0.05)
        concurrent -= 1
        return make_forecast_result(question_id=question.question_id)

    with patch("autoforecast.orchestrator._forecast_one", side_effect=tracked_forecast), \
         patch("autoforecast.orchestrator.load_params", return_value=None):
        await _import_backtest_batch()(questions, batch_id=0, memory="", concurrency=2)

    assert max_concurrent <= 2


@pytest.mark.asyncio
async def test_backtest_batch_deadline_skips():
    """If deadline is already past, no questions should be processed."""
    questions = [make_question(question_id=i) for i in range(3)]

    async def should_not_be_called(*args, **kwargs):
        raise AssertionError("Should not be called")

    with patch("autoforecast.orchestrator._forecast_one", side_effect=should_not_be_called), \
         patch("autoforecast.orchestrator.load_params", return_value=None):
        # Deadline in the past
        results = await _import_backtest_batch()(
            questions, batch_id=0, memory="", deadline=time.monotonic() - 1,
        )

    assert len(results) == 0


@pytest.mark.asyncio
async def test_backtest_batch_failure_isolation():
    """A failure in one question should not affect others."""
    questions = [make_question(question_id=i) for i in range(4)]

    async def failing_forecast(question, memory, platt_params, timeout_seconds=300, handler=None):
        if question.question_id == 2:
            raise RuntimeError("Agent exploded")
        return make_forecast_result(question_id=question.question_id)

    with patch("autoforecast.orchestrator._forecast_one", side_effect=failing_forecast), \
         patch("autoforecast.orchestrator.load_params", return_value=None):
        results = await _import_backtest_batch()(questions, batch_id=0, memory="", concurrency=4)

    assert len(results) == 3
    result_ids = {r.question.question_id for r in results}
    assert 2 not in result_ids


@pytest.mark.asyncio
async def test_backtest_batch_saves_traces(tmp_path):
    """Trace JSON files should be written for each successful result."""
    questions = [make_question(question_id=i) for i in range(3)]

    async def mock_forecast(question, memory, platt_params, timeout_seconds=300, handler=None):
        return make_forecast_result(question_id=question.question_id)

    with patch("autoforecast.orchestrator._forecast_one", side_effect=mock_forecast), \
         patch("autoforecast.orchestrator.load_params", return_value=None), \
         patch("autoforecast.orchestrator.PROJECT_ROOT", tmp_path):
        results = await _import_backtest_batch()(questions, batch_id=7, memory="", concurrency=3)

    traces_dir = tmp_path / "logs" / "traces"
    assert traces_dir.exists()
    trace_files = list(traces_dir.glob("7_*.json"))
    assert len(trace_files) == 3
    # Verify JSON is valid
    for tf in trace_files:
        data = json.loads(tf.read_text())
        assert "raw_probability" in data


# ---------------------------------------------------------------------------
# eval: naive baselines
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_naive_baselines_parallel():
    """Naive baselines should run concurrently."""
    questions = [make_question(question_id=i) for i in range(5)]

    async def slow_baseline(client, question):
        await asyncio.sleep(0.1)
        return 0.5

    with patch("autoforecast.eval._compute_naive_baseline", side_effect=slow_baseline):
        from autoforecast.eval import _compute_naive_baselines
        t0 = time.monotonic()
        results = await _compute_naive_baselines(None, questions)
        elapsed = time.monotonic() - t0

    assert len(results) == 5
    # All 5 in parallel → ~0.1s, not 0.5s
    assert elapsed < 0.3


# ---------------------------------------------------------------------------
# postmortem: parallel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_postmortems_parallel():
    """All postmortems should be produced (order doesn't matter)."""
    from autoforecast.postmortem import _run_one_postmortem
    from autoforecast.types import PostmortemOutput, ProcessClassification, Domain

    results = [make_forecast_result(question_id=i) for i in range(4)]

    async def mock_one(result, client, system_prompt):
        return PostmortemOutput(
            question_id=result.question.question_id,
            question_title=result.question.title,
            outcome=result.question.outcome,
            forecast_probability=result.raw_probability,
            brier_score=0.09,
            process_classification=ProcessClassification.GOOD_PROCESS_GOOD_OUTCOME,
            process_reasoning="test",
            lessons=["lesson"],
            domain=Domain.TECHNOLOGY_POLICY,
        )

    with patch("autoforecast.postmortem._run_one_postmortem", side_effect=mock_one), \
         patch("autoforecast.postmortem.load_prompt", return_value="prompt"):
        from autoforecast.postmortem import run_postmortems
        postmortems = await run_postmortems(results)

    assert len(postmortems) == 4


# ---------------------------------------------------------------------------
# UI: multi-question support
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ui_multi_question_no_clobber():
    """Interleaved events for different questions should all be tracked."""
    handler = RichHandler()
    for qid in [10, 20, 30]:
        await handler.handle(PipelineEvent(
            event_type=EventType.QUESTION_START,
            question_id=qid, question_title=f"Q{qid}",
        ))
        await handler.handle(PipelineEvent(
            event_type=EventType.AGENT_STAGE_START,
            question_id=qid, agent_id=0, stage=Stage.DECOMPOSE,
        ))

    assert len(handler._questions) == 3
    assert all(qid in handler._questions for qid in [10, 20, 30])


@pytest.mark.asyncio
async def test_ui_active_question_switches():
    """The most recently started question should become active."""
    handler = RichHandler()
    await handler.handle(PipelineEvent(
        event_type=EventType.QUESTION_START, question_id=1, question_title="Q1",
    ))
    assert handler._active_question_id == 1

    await handler.handle(PipelineEvent(
        event_type=EventType.QUESTION_START, question_id=2, question_title="Q2",
    ))
    assert handler._active_question_id == 2


@pytest.mark.asyncio
async def test_ui_render_no_crash():
    """Rendering with mixed events should not raise."""
    handler = RichHandler()
    events = [
        PipelineEvent(event_type=EventType.BATCH_PROGRESS, question_id=1, data={"current": 0, "total": 3, "batch_id": 0}),
        PipelineEvent(event_type=EventType.QUESTION_START, question_id=1, question_title="Q1"),
        PipelineEvent(event_type=EventType.AGENT_STAGE_START, question_id=1, agent_id=0, stage=Stage.DECOMPOSE),
        PipelineEvent(event_type=EventType.AGENT_STAGE_DONE, question_id=1, agent_id=0, stage=Stage.DECOMPOSE),
        PipelineEvent(event_type=EventType.QUESTION_START, question_id=2, question_title="Q2"),
        PipelineEvent(event_type=EventType.SUPERVISOR_START, question_id=1, data={"agent_probabilities": [0.5, 0.6, 0.7]}),
        PipelineEvent(event_type=EventType.SUPERVISOR_DONE, question_id=1, data={"reconciled_probability": 0.6}),
        PipelineEvent(event_type=EventType.QUESTION_DONE, question_id=1, data={"raw_probability": 0.6}),
    ]
    for ev in events:
        await handler.handle(ev)

    # Should not raise
    result = handler._render()
    assert result is not None


# ---------------------------------------------------------------------------
# batch size default
# ---------------------------------------------------------------------------


def test_sample_batch_size_12():
    """Default batch size should be 12."""
    questions = [make_question(question_id=i) for i in range(20)]
    batch = sample_batch(questions, 0)
    assert len(batch) == 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_backtest_batch():
    from autoforecast.orchestrator import backtest_batch
    return backtest_batch
