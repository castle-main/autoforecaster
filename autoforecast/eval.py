"""Evaluation: Brier scores, baselines, domain breakdowns."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path

import anthropic
import numpy as np

from .calibrate import fit_platt, save_params, apply_platt
from .types import (
    BatchResult,
    ForecastResult,
    PlattParams,
    Question,
    PROJECT_ROOT,
)

MODEL = "claude-opus-4-6"


def _brier(prob: float, outcome: int) -> float:
    return (prob - outcome) ** 2


async def _compute_naive_baseline(client: anthropic.AsyncAnthropic, question: Question) -> float:
    """Get a naive LLM probability estimate — no pipeline, no search."""
    response = await client.messages.create(
        model=MODEL,
        max_tokens=256,
        temperature=0.0,
        system="You are a forecaster. Given a yes/no question, provide your probability estimate that it resolves Yes. Respond with ONLY a number between 0 and 1, nothing else.",
        messages=[{"role": "user", "content": f"Question: {question.title}\nClose date: {question.close_date}"}],
    )

    text = response.content[0].text.strip()
    # Extract number from response
    import re
    match = re.search(r'(0\.\d+|1\.0|0|1)', text)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not parse naive baseline probability from: {text}")


async def _compute_naive_baselines(
    client: anthropic.AsyncAnthropic,
    questions: list[Question],
) -> list[float]:
    """Get naive baseline probabilities for a list of questions (parallel)."""
    return list(await asyncio.gather(*[_compute_naive_baseline(client, q) for q in questions]))


def _flag_contamination(
    naive_briers: list[float],
    community_briers: list[float],
    question_ids: list[int],
    threshold: float = 0.15,
) -> list[int]:
    """Flag questions where naive LLM beats community by a suspicious margin."""
    flagged = []
    for i, (nb, cb) in enumerate(zip(naive_briers, community_briers)):
        if cb - nb > threshold:
            flagged.append(question_ids[i])
    return flagged


async def evaluate_batch(
    results: list[ForecastResult],
    batch_id: int,
    all_raw_probs: list[float],
    all_outcomes: list[int],
) -> BatchResult:
    """Score a batch: Brier for pipeline, community, naive. Refit Platt. Domain breakdown."""
    client = anthropic.AsyncAnthropic(timeout=120.0)
    questions = [r.question for r in results]

    # Pipeline Brier (use calibrated if available, else raw)
    pipeline_probs = [r.calibrated_probability or r.raw_probability for r in results]
    pipeline_briers = [_brier(p, r.question.outcome) for p, r in zip(pipeline_probs, results)]
    brier_pipeline = float(np.mean(pipeline_briers))

    # Community Brier
    community_briers = [_brier(q.community_prediction_final, q.outcome) for q in questions]
    brier_community = float(np.mean(community_briers))

    # Naive LLM Brier
    naive_probs = await _compute_naive_baselines(client, questions)
    naive_briers = [_brier(p, q.outcome) for p, q in zip(naive_probs, questions)]
    brier_naive = float(np.mean(naive_briers))

    # Domain breakdown
    domain_scores: dict[str, list[float]] = defaultdict(list)
    for brier, r in zip(pipeline_briers, results):
        domain_scores[r.question.domain.value].append(brier)
    brier_by_domain = {d: float(np.mean(scores)) for d, scores in domain_scores.items()}

    # Per-agent Brier (average each agent's raw probability vs outcome)
    agent_scores: dict[int, list[float]] = defaultdict(list)
    for r in results:
        for trace in r.agent_traces:
            agent_scores[trace.agent_id].append(_brier(trace.raw_probability, r.question.outcome))
    brier_by_agent = {aid: float(np.mean(scores)) for aid, scores in agent_scores.items()}

    # Contamination flags (naive LLM vs community)
    contamination_flags = _flag_contamination(
        naive_briers, community_briers,
        [q.question_id for q in questions],
    )

    # Search contamination rate: fraction of search rounds where outcome leakage was detected
    total_search_rounds = 0
    contaminated_rounds = 0
    for r in results:
        for trace in r.agent_traces:
            for search_round in trace.research.search_trace.rounds:
                total_search_rounds += 1
                if search_round.result.was_contaminated:
                    contaminated_rounds += 1
        for search_result in r.supervisor.targeted_searches:
            total_search_rounds += 1
            if search_result.was_contaminated:
                contaminated_rounds += 1
    contamination_rate = contaminated_rounds / total_search_rounds if total_search_rounds > 0 else 0.0

    # Refit Platt on all available data
    platt_params = None
    if len(all_raw_probs) >= 10:
        platt_params = fit_platt(all_raw_probs, all_outcomes)
        save_params(platt_params)

    # Save batch scores
    scores_dir = PROJECT_ROOT / "logs" / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)
    batch_result = BatchResult(
        batch_id=batch_id,
        n_questions=len(results),
        brier_pipeline=brier_pipeline,
        brier_community=brier_community,
        brier_naive=brier_naive,
        brier_by_domain=brier_by_domain,
        brier_by_agent=brier_by_agent,
        contamination_flags=contamination_flags,
        contamination_rate=contamination_rate,
        platt_params=platt_params,
    )

    with open(scores_dir / f"batch_{batch_id}.json", "w") as f:
        json.dump(batch_result.model_dump(), f, indent=2)

    return batch_result
