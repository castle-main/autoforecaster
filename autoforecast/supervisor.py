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
    ClusterResearchOutput,
    ClusterSupervisorOutput,
    Question,
    SearchResult,
    SupervisorOutput,
    clean_schema,
)
from .utils import load_prompt

MODEL = "claude-opus-4-6"
MAX_SUPERVISOR_SEARCH_ROUNDS = 1


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
            f"**Synthesis:** {trace.synthesis.final_probability:.2%} "
            f"(confidence: {trace.synthesis.confidence_level}, "
            f"80% CI: {trace.synthesis.confidence_interval_lower:.2%}–{trace.synthesis.confidence_interval_upper:.2%})\n"
            f"Confidence justification: {trace.synthesis.confidence_justification}\n"
            f"Reasoning: {trace.synthesis.adjustment_reasoning}\n"
            f"Confidence: {trace.synthesis.confidence_reasoning}\n"
        )
    return "\n---\n\n".join(parts)


async def supervise(
    question: Question,
    traces: list[AgentTrace],
    memory: str | None = None,
    handler=None,
    live_mode: bool = False,
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
        f"You have 1 search query you can use. Before deciding, ask yourself: "
        f"what specific claim would this search prove or disprove, and would the answer "
        f"change my reconciled probability? "
        f"If yes, provide exactly 1 search query as a JSON list: "
        f'[{{"query": "...", "reason": "why this search would change the reconciliation"}}]. '
        f"If the agents' evidence is sufficient, provide an empty list: []"
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
                execute_search(q["query"], question.close_date, handler=_handler, question_title=question.title, model="sonar-pro", live_mode=live_mode)
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


def _format_cluster_traces(
    questions: list[Question],
    traces_by_question: dict[str, list[AgentTrace]],
) -> str:
    """Format all questions' agent traces for cluster supervisor review."""
    parts = []
    for q in questions:
        traces = traces_by_question.get(q.id, [])
        parts.append(
            f"# Question: {q.title}\n"
            f"ID: {q.id}\n"
            f"Close date: {q.close_date}\n\n"
            + _format_traces(traces)
        )
    return "\n\n===\n\n".join(parts)


# Tool schema for cluster reconciliation output
_CLUSTER_RECONCILIATION_SCHEMA = {
    "type": "object",
    "properties": {
        "question_probabilities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question_id": {"type": "string"},
                    "probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "reasoning": {"type": "string"},
                },
                "required": ["question_id", "probability", "reasoning"],
            },
            "description": "Reconciled probability for each question in the cluster",
        },
        "coherence_reasoning": {
            "type": "string",
            "description": "How the sum <= 1.0 constraint was enforced",
        },
        "disagreements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Key disagreements identified across the cluster",
        },
    },
    "required": ["question_probabilities", "coherence_reasoning", "disagreements"],
}


async def supervise_cluster(
    cluster_tag: str,
    questions: list[Question],
    traces_by_question: dict[str, list[AgentTrace]],
    shared_research: ClusterResearchOutput | None = None,
    memory: str | None = None,
    handler=None,
    live_mode: bool = False,
) -> ClusterSupervisorOutput:
    """Reconcile agent traces across a cluster of mutually exclusive questions."""
    _handler = handler or NullHandler()
    client = anthropic.AsyncAnthropic(timeout=120.0)
    system_prompt = load_prompt("cluster_supervisor")
    if memory:
        system_prompt += f"\n\n## Accumulated Forecasting Lessons\n{memory}"

    cluster_traces_text = _format_cluster_traces(questions, traces_by_question)

    # Collect all agent probs for event reporting
    all_agent_probs = {}
    for q in questions:
        all_agent_probs[q.id] = [t.raw_probability for t in traces_by_question.get(q.id, [])]

    await _handler.handle(PipelineEvent(
        event_type=EventType.SUPERVISOR_START,
        question_id=questions[0].question_id if questions else 0,
        question_title=f"Cluster: {cluster_tag}",
        stage=Stage.SUPERVISOR,
        data={"cluster_tag": cluster_tag, "n_questions": len(questions), "agent_probabilities_by_question": all_agent_probs},
    ))

    # Shared research context
    shared_research_text = ""
    if shared_research:
        shared_research_text = (
            "\n\n## Shared Cluster Research\n"
            + "\n".join(f"- {f}" for f in shared_research.shared_findings)
        )

    # Step 1: Identify disagreements and decide on targeted searches
    identify_msg = (
        f"Cluster: {cluster_tag}\n"
        f"Number of questions: {len(questions)}\n"
        f"These questions represent mutually exclusive outcomes of the same event.\n\n"
        f"{cluster_traces_text}\n"
        f"{shared_research_text}\n\n"
        f"First, identify the key disagreements between agents — both within each question "
        f"and across questions. Pay special attention to whether the sum of agent estimates "
        f"across questions exceeds 1.0.\n"
        f"You have up to {len(questions)} search queries (1 per question in the cluster). "
        f"Before searching, ask yourself for each query: "
        f"what specific claim would this search prove or disprove, and would the answer "
        f"change my reconciled probability for that question? "
        f"Provide search queries as a JSON list: "
        f'[{{"query": "...", "reason": "why this search would change the reconciliation"}}]. '
        f"If the agents' evidence is sufficient, provide an empty list: []"
    )

    search_decision = await client.messages.create(
        model=MODEL,
        max_tokens=4096,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": identify_msg}],
    )

    if search_decision.usage:
        await track_api_cost(_handler, "anthropic", MODEL, search_decision.usage.input_tokens, search_decision.usage.output_tokens)

    search_decision_text = search_decision.content[0].text

    # Extract search queries and run them in parallel
    targeted_searches: list[SearchResult] = []
    # Use the first question's close_date for search date gating
    close_date = questions[0].close_date if questions else None
    # Scale search budget by number of questions in the cluster
    max_searches = len(questions)
    try:
        match = re.search(r'\[.*?\]', search_decision_text, re.DOTALL)
        if match:
            queries = json.loads(match.group())
            valid_queries = [q for q in queries[:max_searches] if isinstance(q, dict) and q.get("query")]
            for q in valid_queries:
                await _handler.handle(PipelineEvent(
                    event_type=EventType.SUPERVISOR_SEARCH,
                    question_id=questions[0].question_id if questions else 0,
                    stage=Stage.SUPERVISOR,
                    data={"query": q["query"], "cluster_tag": cluster_tag},
                ))
            # Use first question's title as representative for contamination checking
            q_title = questions[0].title if questions else ""
            results = await asyncio.gather(*[
                execute_search(q["query"], close_date, handler=_handler, question_title=q_title, model="sonar-pro", live_mode=live_mode)
                for q in valid_queries
            ])
            targeted_searches.extend(results)
    except (json.JSONDecodeError, KeyError):
        pass

    # Step 2: Reconcile with coherence constraint
    search_results_text = ""
    if targeted_searches:
        search_results_text = "\n\n## Targeted Search Results\n" + "\n\n".join([
            f"**Query: {s.query}**\n{s.content}" for s in targeted_searches
        ])

    reconcile_msg = (
        f"Cluster: {cluster_tag}\n"
        f"These questions represent mutually exclusive outcomes of the same event. "
        f"Their probabilities MUST sum to ≤ 1.0. Allocate probability budget across outcomes.\n\n"
        f"{cluster_traces_text}\n"
        f"{shared_research_text}\n"
        f"{search_results_text}\n\n"
        f"Now produce your reconciled probabilities for all {len(questions)} questions. "
        f"Remember: the sum of all probabilities must be ≤ 1.0."
    )

    tool = {
        "name": "provide_cluster_output",
        "description": "Provide reconciled probabilities for all questions in the cluster",
        "input_schema": _CLUSTER_RECONCILIATION_SCHEMA,
    }

    response = await client.messages.create(
        model=MODEL,
        max_tokens=8192,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": reconcile_msg}],
        tools=[tool],
        tool_choice={"type": "tool", "name": "provide_cluster_output"},
    )

    if response.usage:
        await track_api_cost(_handler, "anthropic", MODEL, response.usage.input_tokens, response.usage.output_tokens)

    for block in response.content:
        if block.type == "tool_use":
            data = block.input

            # Build per-question SupervisorOutput objects
            question_results: dict[str, SupervisorOutput] = {}
            prob_sum = 0.0

            for qp in data["question_probabilities"]:
                qid = qp["question_id"]
                prob = qp["probability"]
                prob_sum += prob

                agent_probs = all_agent_probs.get(qid, [])
                question_results[qid] = SupervisorOutput(
                    agent_probabilities=agent_probs,
                    disagreements=data.get("disagreements", []),
                    targeted_searches=targeted_searches,
                    reconciliation_reasoning=qp["reasoning"],
                    reconciled_probability=prob,
                )

            # Post-hoc enforcement: normalize if sum > 1.0
            if prob_sum > 1.0:
                scale = 1.0 / prob_sum
                for qid, sv in question_results.items():
                    sv.reconciled_probability = sv.reconciled_probability * scale
                prob_sum = sum(sv.reconciled_probability for sv in question_results.values())

            output = ClusterSupervisorOutput(
                cluster_tag=cluster_tag,
                question_results=question_results,
                probability_sum=prob_sum,
                coherence_reasoning=data.get("coherence_reasoning", ""),
            )

            await _handler.handle(PipelineEvent(
                event_type=EventType.SUPERVISOR_DONE,
                question_id=questions[0].question_id if questions else 0,
                question_title=f"Cluster: {cluster_tag}",
                stage=Stage.SUPERVISOR,
                data={
                    "cluster_tag": cluster_tag,
                    "probability_sum": output.probability_sum,
                    "probabilities": {qid: sv.reconciled_probability for qid, sv in question_results.items()},
                },
            ))

            return output

    raise RuntimeError("No tool_use block in cluster supervisor response")
