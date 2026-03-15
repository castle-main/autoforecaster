"""Shared fixtures for parallel tests."""

from __future__ import annotations

import pytest

from autoforecast.types import (
    AgentTrace,
    BaseRateOutput,
    DecomposeOutput,
    Domain,
    ForecastResult,
    InsideViewOutput,
    Question,
    ResearchOutput,
    SearchResult,
    SearchRound,
    SearchTrace,
    SupervisorOutput,
    SynthesisOutput,
)


def make_question(question_id: int = 1, **overrides) -> Question:
    defaults = dict(
        post_id=question_id + 40000,
        question_id=question_id,
        id=f"metaculus_{question_id + 40000}",
        source="metaculus",
        url=f"https://www.metaculus.com/questions/{question_id + 40000}/",
        title=f"Test question {question_id}",
        description="",
        resolution_criteria="",
        created_date="2026-01-01",
        close_date="2026-02-01",
        resolved_date="2026-02-01",
        outcome=1,
        community_prediction_final=0.6,
        num_forecasters=50,
        tags=["test"],
        domain=Domain.TECHNOLOGY_POLICY,
    )
    defaults.update(overrides)
    return Question(**defaults)


def make_agent_trace(agent_id: int = 0, question_id: int = 1, prob: float = 0.7) -> AgentTrace:
    return AgentTrace(
        agent_id=agent_id,
        question_id=question_id,
        decompose=DecomposeOutput(
            sub_questions=["sub1"], initial_search_queries=["q1"], reasoning="test",
        ),
        research=ResearchOutput(
            key_findings=["finding1"],
            evidence_for=["ev1"],
            evidence_against=["ev2"],
            search_trace=SearchTrace(
                rounds=[SearchRound(
                    round_number=1, query="q1",
                    result=SearchResult(query="q1", content="content"),
                    reasoning="test",
                )],
                total_citations=1,
            ),
            information_gaps=["gap1"],
        ),
        base_rate=BaseRateOutput(
            reference_classes=["ref1"], base_rate_estimate=0.5, reasoning="test",
        ),
        inside_view=InsideViewOutput(
            factors_for=["f1"], factors_against=["f2"],
            inside_view_estimate=prob, reasoning="test",
        ),
        synthesis=SynthesisOutput(
            base_rate_weight=0.3, adjustment_reasoning="test",
            final_probability=prob, confidence_reasoning="test",
        ),
        raw_probability=prob,
    )


def make_forecast_result(question_id: int = 1, prob: float = 0.7) -> ForecastResult:
    q = make_question(question_id=question_id)
    traces = [make_agent_trace(agent_id=i, question_id=question_id, prob=prob) for i in range(3)]
    return ForecastResult(
        question=q,
        agent_traces=traces,
        supervisor=SupervisorOutput(
            agent_probabilities=[prob] * 3,
            disagreements=[],
            reconciliation_reasoning="test",
            reconciled_probability=prob,
        ),
        raw_probability=prob,
        calibrated_probability=None,
        brier_raw=(prob - q.outcome) ** 2,
    )
