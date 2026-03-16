"""Shared research for clusters of related questions."""

from __future__ import annotations

import asyncio
import json
import os

import anthropic

from .events import EventType, NullHandler, PipelineEvent, Stage, track_api_cost
from .search import execute_search
from .types import (
    ClusterResearchOutput,
    Question,
    SearchRound,
    SearchTrace,
    clean_schema,
)
from .utils import extract_json, load_prompt

MODEL = "claude-opus-4-6"
FAST_MODEL = "claude-haiku-4-5-20251001"
MAX_SEARCH_ROUNDS = 3


async def run_cluster_research(
    cluster_tag: str,
    questions: list[Question],
    memory: str | None = None,
    handler=None,
    live_mode: bool = False,
) -> ClusterResearchOutput:
    """Run shared research for a cluster of related questions.

    Generates search queries covering the shared topic, runs agentic search,
    and synthesizes findings into shared context for per-question agents.
    """
    _handler = handler or NullHandler()
    client = anthropic.AsyncAnthropic(timeout=120.0)

    # Build the cluster research prompt
    cluster_prompt = load_prompt("cluster_research")
    question_list = "\n".join(f"- {q.title}" for q in questions)
    memory_section = f"\n\n## Accumulated Forecasting Lessons\n{memory}" if memory else ""

    user_msg = (
        f"Cluster topic: {cluster_tag}\n\n"
        f"Questions in this cluster ({len(questions)}):\n{question_list}\n"
        f"{memory_section}"
    )

    # Step 1: Generate initial search queries via LLM
    query_gen_response = await client.messages.create(
        model=MODEL,
        max_tokens=2048,
        temperature=1.0,
        system=cluster_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )

    if query_gen_response.usage:
        await track_api_cost(_handler, "anthropic", MODEL, query_gen_response.usage.input_tokens, query_gen_response.usage.output_tokens)

    # Parse search queries from response
    response_text = query_gen_response.content[0].text
    try:
        parsed = extract_json(response_text)
        initial_queries = parsed.get("queries", [])[:4]  # Cap at 4 initial queries
    except (json.JSONDecodeError, AttributeError):
        # Fallback: use the cluster tag as a single query
        initial_queries = [f"{cluster_tag} latest developments 2026"]

    if not initial_queries:
        initial_queries = [f"{cluster_tag} latest developments 2026"]

    # Step 2: Run agentic search (same pattern as agent.py _run_agentic_search)
    # Fire all initial queries in parallel
    initial_results = await asyncio.gather(*[
        execute_search(query, questions[0].close_date, handler=_handler, question_title=cluster_tag, live_mode=live_mode)
        for query in initial_queries
    ])

    rounds: list[SearchRound] = [
        SearchRound(
            round_number=i + 1,
            query=query,
            result=result,
            reasoning="Initial query from cluster research",
        )
        for i, (query, result) in enumerate(zip(initial_queries, initial_results))
    ]

    # Follow-up rounds driven by Haiku
    if len(rounds) < MAX_SEARCH_ROUNDS:
        for round_num in range(len(rounds) + 1, MAX_SEARCH_ROUNDS + 1):
            search_summary = "\n\n".join([
                f"**Search {r.round_number}: {r.query}**\n{r.result.content[:1000]}"
                for r in rounds
            ])

            follow_up_response = await client.messages.create(
                model=FAST_MODEL,
                max_tokens=1024,
                temperature=1.0,
                system=(
                    "You are a research assistant. Given the cluster topic, questions, and search results so far, "
                    "decide if more searching would help cover the shared research needs. "
                    "If yes, respond with a JSON object: {\"search\": true, \"query\": \"your query\", \"reasoning\": \"why\"}. "
                    "If no, respond with: {\"search\": false, \"reasoning\": \"why not\"}."
                ),
                messages=[{"role": "user", "content": (
                    f"Cluster topic: {cluster_tag}\n\n"
                    f"Questions:\n{question_list}\n\n"
                    f"Search results so far:\n{search_summary}\n\n"
                    "Should we search for more information? Respond with JSON only."
                )}],
            )

            if follow_up_response.usage:
                await track_api_cost(_handler, "anthropic", FAST_MODEL, follow_up_response.usage.input_tokens, follow_up_response.usage.output_tokens)

            text = follow_up_response.content[0].text
            try:
                decision = extract_json(text)
            except json.JSONDecodeError:
                break

            if not decision.get("search", False):
                break

            result = await execute_search(
                decision["query"], questions[0].close_date,
                handler=_handler, question_title=cluster_tag,
                model="sonar-pro", live_mode=live_mode,
            )
            rounds.append(SearchRound(
                round_number=round_num,
                query=decision["query"],
                result=result,
                reasoning=decision.get("reasoning", ""),
            ))

    total_citations = sum(len(r.result.citations) for r in rounds)
    search_trace = SearchTrace(rounds=rounds, total_citations=total_citations)

    # Step 3: Synthesize shared findings via LLM
    all_search_content = "\n\n".join([
        f"**Search {r.round_number}: {r.query}**\n{r.result.content}"
        for r in rounds
    ])

    synthesis_response = await client.messages.create(
        model=MODEL,
        max_tokens=4096,
        temperature=0.0,
        system=(
            "You are a research synthesizer. Given search results about a cluster of related forecasting questions, "
            "extract the key shared findings that are relevant to ALL questions in the cluster. "
            "Return a JSON object with a single key \"findings\" containing a list of concise finding strings. "
            "Each finding should be a self-contained factual statement. Aim for 5-15 findings."
        ),
        messages=[{"role": "user", "content": (
            f"Cluster topic: {cluster_tag}\n\n"
            f"Questions:\n{question_list}\n\n"
            f"Search results:\n{all_search_content}"
        )}],
    )

    if synthesis_response.usage:
        await track_api_cost(_handler, "anthropic", MODEL, synthesis_response.usage.input_tokens, synthesis_response.usage.output_tokens)

    try:
        parsed = extract_json(synthesis_response.content[0].text)
        shared_findings = parsed.get("findings", [])
    except (json.JSONDecodeError, AttributeError):
        # Fallback: use raw search content as a single finding
        shared_findings = [all_search_content[:2000]]

    return ClusterResearchOutput(
        cluster_tag=cluster_tag,
        shared_findings=shared_findings,
        search_trace=search_trace,
    )
