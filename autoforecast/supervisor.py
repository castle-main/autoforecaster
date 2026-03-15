"""Supervisor: reconcile 3 agent traces via targeted search."""

from __future__ import annotations

import asyncio
import json
import re

import anthropic

from .events import EventType, NullHandler, PipelineEvent, Stage, track_api_cost
from .search import execute_search
from .types import (
    AgentTrace,
    Question,
    SearchResult,
    SupervisorOutput,
    clean_schema,
)
from .utils import load_prompt

MODEL = "claude-opus-4-6"
MAX_SUPERVISOR_SEARCH_ROUNDS = 3


def _format_traces(traces: list[AgentTrace]) -> str:
    """Format agent traces for supervisor review."""
    parts = []
    for trace in traces:
        parts.append(
            f"## Agent {trace.agent_id} (probability: {trace.raw_probability:.2%})\n\n"
            f"**Decomposition:**\n"
            f"Sub-questions: {', '.join(trace.decompose.sub_questions)}\n\n"
            f"**Key Findings:**\n"
            + "\n".join(f"- {f}" for f in trace.research.key_findings) +
            f"\n\n**Base Rate:** {trace.base_rate.base_rate_estimate:.2%}\n"
            f"Reference classes: {', '.join(trace.base_rate.reference_classes)}\n"
            f"Reasoning: {trace.base_rate.reasoning}\n\n"
            f"**Inside View:** {trace.inside_view.inside_view_estimate:.2%}\n"
            f"Factors for: {', '.join(trace.inside_view.factors_for)}\n"
            f"Factors against: {', '.join(trace.inside_view.factors_against)}\n"
            f"Reasoning: {trace.inside_view.reasoning}\n\n"
            f"**Synthesis:** {trace.synthesis.final_probability:.2%}\n"
            f"Reasoning: {trace.synthesis.adjustment_reasoning}\n"
            f"Confidence: {trace.synthesis.confidence_reasoning}\n"
        )
    return "\n---\n\n".join(parts)


async def supervise(
    question: Question,
    traces: list[AgentTrace],
    memory: str | None = None,
    handler=None,
) -> SupervisorOutput:
    """Reconcile 3 agent traces into a single probability."""
    _handler = handler or NullHandler()
    client = anthropic.AsyncAnthropic(timeout=120.0)
    system_prompt = load_prompt("supervisor")
    if memory:
        system_prompt += f"\n\n## Accumulated Forecasting Lessons\n{memory}"

    traces_text = _format_traces(traces)
    agent_probs = [t.raw_probability for t in traces]

    await _handler.handle(PipelineEvent(
        event_type=EventType.SUPERVISOR_START, question_id=question.question_id,
        question_title=question.title, stage=Stage.SUPERVISOR,
        data={"agent_probabilities": agent_probs},
    ))

    # Step 1: Identify disagreements and decide on targeted searches
    identify_msg = (
        f"Question: {question.title}\n"
        f"Close date: {question.close_date}\n\n"
        f"{traces_text}\n\n"
        f"First, identify the key disagreements between agents. "
        f"Then decide if targeted searches would help resolve them. "
        f"If yes, provide up to {MAX_SUPERVISOR_SEARCH_ROUNDS} search queries as a JSON list: "
        f'[{{"query": "...", "reason": "..."}}]. '
        f"If no searches needed, provide an empty list: []"
    )

    search_decision = await client.messages.create(
        model=MODEL,
        max_tokens=2048,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": identify_msg}],
    )

    if search_decision.usage:
        await track_api_cost(_handler, "anthropic", MODEL, search_decision.usage.input_tokens, search_decision.usage.output_tokens)

    search_decision_text = search_decision.content[0].text

    # Extract search queries and run them in parallel
    targeted_searches: list[SearchResult] = []
    try:
        match = re.search(r'\[.*?\]', search_decision_text, re.DOTALL)
        if match:
            queries = json.loads(match.group())
            valid_queries = [q for q in queries[:MAX_SUPERVISOR_SEARCH_ROUNDS] if isinstance(q, dict) and q.get("query")]
            for q in valid_queries:
                await _handler.handle(PipelineEvent(
                    event_type=EventType.SUPERVISOR_SEARCH,
                    question_id=question.question_id, stage=Stage.SUPERVISOR,
                    data={"query": q["query"]},
                ))
            results = await asyncio.gather(*[
                execute_search(q["query"], question.close_date, handler=_handler, question_title=question.title, model="sonar-pro")
                for q in valid_queries
            ])
            targeted_searches.extend(results)
    except (json.JSONDecodeError, KeyError):
        pass  # No valid search queries found — proceed without

    # Step 2: Reconcile with all evidence
    search_results_text = ""
    if targeted_searches:
        search_results_text = "\n\n## Targeted Search Results\n" + "\n\n".join([
            f"**Query: {s.query}**\n{s.content}" for s in targeted_searches
        ])

    reconcile_msg = (
        f"Question: {question.title}\n"
        f"Close date: {question.close_date}\n\n"
        f"{traces_text}\n"
        f"{search_results_text}\n\n"
        f"Now produce your reconciled probability estimate."
    )

    schema = SupervisorOutput.model_json_schema()
    tool_schema = clean_schema(schema)

    tool = {
        "name": "provide_output",
        "description": "Provide the SupervisorOutput",
        "input_schema": tool_schema,
    }

    response = await client.messages.create(
        model=MODEL,
        max_tokens=4096,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": reconcile_msg}],
        tools=[tool],
        tool_choice={"type": "tool", "name": "provide_output"},
    )

    if response.usage:
        await track_api_cost(_handler, "anthropic", MODEL, response.usage.input_tokens, response.usage.output_tokens)

    for block in response.content:
        if block.type == "tool_use":
            output = SupervisorOutput.model_validate(block.input)
            # Ensure agent_probabilities is correct
            output.agent_probabilities = agent_probs
            # Inject targeted searches
            output.targeted_searches = targeted_searches
            await _handler.handle(PipelineEvent(
                event_type=EventType.SUPERVISOR_DONE,
                question_id=question.question_id, stage=Stage.SUPERVISOR,
                data={
                    "reconciled_probability": output.reconciled_probability,
                    "disagreements": output.disagreements,
                },
            ))
            return output

    raise RuntimeError("No tool_use block in supervisor response")
