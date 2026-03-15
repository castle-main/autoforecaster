"""Postmortem: classify process quality and extract lessons."""

from __future__ import annotations

import asyncio
import re

import anthropic

from .agent import _clean_schema
from .types import (
    ForecastResult,
    PostmortemOutput,
    ProcessClassification,
    PROJECT_ROOT,
)

MODEL = "claude-opus-4-6"


def _load_prompt() -> str:
    return (PROJECT_ROOT / "prompts" / "postmortem.md").read_text()


def _format_result_for_postmortem(result: ForecastResult) -> str:
    """Format a forecast result for postmortem analysis."""
    prob = result.calibrated_probability or result.raw_probability
    brier = (prob - result.question.outcome) ** 2

    traces_summary = []
    for trace in result.agent_traces:
        traces_summary.append(
            f"### Agent {trace.agent_id} ({trace.raw_probability:.2%})\n"
            f"Sub-questions: {', '.join(trace.decompose.sub_questions)}\n"
            f"Key findings: {', '.join(trace.research.key_findings[:3])}\n"
            f"Base rate: {trace.base_rate.base_rate_estimate:.2%} — {trace.base_rate.reasoning[:200]}\n"
            f"Inside view: {trace.inside_view.inside_view_estimate:.2%} — {trace.inside_view.reasoning[:200]}\n"
            f"Final: {trace.synthesis.final_probability:.2%} — {trace.synthesis.confidence_reasoning[:200]}"
        )

    return (
        f"# Forecast Postmortem\n\n"
        f"**Question:** {result.question.title}\n"
        f"**Domain:** {result.question.domain.value}\n"
        f"**Close date:** {result.question.close_date}\n"
        f"**Outcome:** {'Yes' if result.question.outcome == 1 else 'No'}\n"
        f"**Pipeline forecast:** {prob:.2%}\n"
        f"**Brier score:** {brier:.4f}\n"
        f"**Community prediction:** {result.question.community_prediction_final:.2%}\n\n"
        f"## Supervisor\n"
        f"Reconciled: {result.supervisor.reconciled_probability:.2%}\n"
        f"Disagreements: {', '.join(result.supervisor.disagreements)}\n"
        f"Reasoning: {result.supervisor.reconciliation_reasoning}\n\n"
        f"## Agent Traces\n\n" + "\n\n".join(traces_summary)
    )


async def _run_one_postmortem(
    result: ForecastResult,
    client: anthropic.AsyncAnthropic,
    system_prompt: str,
) -> PostmortemOutput:
    """Run postmortem on a single forecast result."""
    prob = result.calibrated_probability or result.raw_probability
    brier = (prob - result.question.outcome) ** 2

    user_msg = _format_result_for_postmortem(result)

    schema = PostmortemOutput.model_json_schema()
    tool_schema = _clean_schema(schema)

    tool = {
        "name": "provide_output",
        "description": "Provide the PostmortemOutput",
        "input_schema": tool_schema,
    }

    response = await client.messages.create(
        model=MODEL,
        max_tokens=2048,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
        tools=[tool],
        tool_choice={"type": "tool", "name": "provide_output"},
    )

    for block in response.content:
        if block.type == "tool_use":
            pm = PostmortemOutput.model_validate(block.input)
            # Override computed fields to ensure consistency
            pm.question_id = result.question.question_id
            pm.question_title = result.question.title
            pm.outcome = result.question.outcome
            pm.forecast_probability = prob
            pm.brier_score = brier
            pm.domain = result.question.domain
            return pm

    raise RuntimeError(f"No tool_use block for postmortem of question {result.question.question_id}")


async def run_postmortems(
    results: list[ForecastResult],
) -> list[PostmortemOutput]:
    """Run postmortem on each forecast result (parallel)."""
    client = anthropic.AsyncAnthropic(timeout=120.0)
    system_prompt = _load_prompt()

    return list(await asyncio.gather(*[
        _run_one_postmortem(result, client, system_prompt)
        for result in results
    ]))


def update_memory(postmortems: list[PostmortemOutput]) -> None:
    """Append lessons from postmortems to memory.md."""
    memory_path = PROJECT_ROOT / "memory.md"

    existing = memory_path.read_text() if memory_path.exists() else "# Forecasting Lessons\n\n"

    new_lessons = []
    for pm in postmortems:
        if not pm.lessons:
            continue
        lessons_text = "\n".join(f"  - {lesson}" for lesson in pm.lessons)
        # Tag with source question_id for A/B filtering
        new_lessons.append(
            f"<!-- source: {pm.question_id} -->\n"
            f"- **{pm.question_title}** ({pm.domain.value}, {pm.process_classification.value}):\n"
            f"{lessons_text}"
        )

    if new_lessons:
        existing += "\n" + "\n\n".join(new_lessons) + "\n"
        memory_path.write_text(existing)


def filter_memory_for_ab(memory: str, exclude_ids: set[int]) -> str:
    """Filter memory to exclude lessons from specific question IDs.

    Used during A/B testing to prevent hindsight contamination.
    """
    if not memory:
        return memory

    lines = memory.split("\n")
    filtered_lines = []
    skip_block = False

    for line in lines:
        # Check for source tags
        source_match = re.search(r'<!-- source: (\d+) -->', line)
        if source_match:
            qid = int(source_match.group(1))
            if qid in exclude_ids:
                skip_block = True
                continue
            else:
                skip_block = False

        if skip_block:
            # Skip until next source tag or empty line that isn't part of a lesson
            if line.strip() == "" or (line.startswith("- **") and "<!-- source:" not in line):
                skip_block = False
            else:
                continue

        filtered_lines.append(line)

    return "\n".join(filtered_lines)
