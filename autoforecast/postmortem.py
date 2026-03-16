"""Postmortem: classify process quality and extract lessons."""

from __future__ import annotations

import asyncio

import anthropic

from .types import (
    ForecastResult,
    PostmortemOutput,
    ProcessClassification,
    PROJECT_ROOT,
    clean_schema,
)
from .utils import load_prompt

MODEL = "claude-opus-4-6"


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
    tool_schema = clean_schema(schema)

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
    system_prompt = load_prompt("postmortem")

    return list(await asyncio.gather(*[
        _run_one_postmortem(result, client, system_prompt)
        for result in results
    ]))


def update_memory(postmortems: list[PostmortemOutput]) -> None:
    """Append lessons from postmortems to memory.md, grouped by domain."""
    memory_path = PROJECT_ROOT / "memory.md"

    existing = memory_path.read_text() if memory_path.exists() else "# Forecasting Lessons\n\n"

    # Group lessons by domain
    domain_lessons: dict[str, list[str]] = {}
    for pm in postmortems:
        if not pm.lessons:
            continue
        domain = pm.domain.value
        if domain not in domain_lessons:
            domain_lessons[domain] = []
        domain_lessons[domain].extend(pm.lessons)

    if domain_lessons:
        new_blocks = []
        for domain, lessons in sorted(domain_lessons.items()):
            lines = "\n".join(f"- {lesson}" for lesson in lessons)
            new_blocks.append(f"<!-- domain: {domain} -->\n{lines}")
        existing += "\n" + "\n\n".join(new_blocks) + "\n"
        memory_path.write_text(existing)


async def consolidate_memory() -> None:
    """Consolidate memory.md using an LLM to deduplicate and merge lessons."""
    memory_path = PROJECT_ROOT / "memory.md"

    if not memory_path.exists():
        return

    content = memory_path.read_text()
    # Skip consolidation if memory is still small
    if content.count("\n") < 30:
        return

    client = anthropic.AsyncAnthropic(timeout=120.0)

    response = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=4096,
        temperature=0.0,
        system=(
            "You consolidate a forecasting lessons file. Your job:\n"
            "1. Deduplicate semantically similar lessons\n"
            "2. Merge related lessons within the same domain\n"
            "3. Preserve <!-- domain: X --> tags for organizational grouping\n"
            "4. Keep ONLY generalizable forecasting methodology — remove any event-specific "
            "facts, outcomes, names, dates, or details that leaked through\n"
            "5. Output clean consolidated markdown starting with '# Forecasting Lessons'\n\n"
            "Return ONLY the consolidated markdown, no commentary."
        ),
        messages=[{"role": "user", "content": content}],
    )

    consolidated = response.content[0].text
    memory_path.write_text(consolidated)
