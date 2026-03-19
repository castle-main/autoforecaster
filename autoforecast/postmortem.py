"""Postmortem: classify process quality and extract lessons."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

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
HAIKU_MODEL = "claude-haiku-4-5-20251001"


def _format_result_for_postmortem(result: ForecastResult) -> str:
    """Format a forecast result for postmortem analysis."""
    prob = result.calibrated_probability or result.raw_probability
    community = result.question.community_prediction_final
    divergence = prob - community
    resolved = result.question.outcome is not None

    if resolved:
        brier = (prob - result.question.outcome) ** 2
        outcome_str = "Yes" if result.question.outcome == 1 else "No"
    else:
        brier = abs(divergence)
        outcome_str = f"Unresolved (community target: {community:.2%})"

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

    if resolved:
        score_line = f"**Brier score:** {brier:.4f}\n"
    else:
        score_line = f"**Absolute divergence from community:** {brier:.4f}\n"

    return (
        f"# Forecast Postmortem\n\n"
        f"**Question:** {result.question.title}\n"
        f"**Domain:** {result.question.domain.value}\n"
        f"**Close date:** {result.question.close_date}\n"
        f"**Outcome:** {outcome_str}\n"
        f"**Pipeline forecast:** {prob:.2%}\n"
        f"{score_line}"
        f"**Community prediction:** {community:.2%}\n"
        f"**Divergence (pipeline - community):** {divergence:+.4f}\n\n"
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
    community = result.question.community_prediction_final
    divergence = prob - community
    resolved = result.question.outcome is not None

    if resolved:
        brier = (prob - result.question.outcome) ** 2
    else:
        brier = abs(divergence)

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
            pm.question_id = result.question.question_id
            pm.question_title = result.question.title
            pm.outcome = result.question.outcome
            pm.forecast_probability = prob
            pm.brier_score = brier if resolved else None
            pm.community_prediction = community
            pm.divergence = divergence
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


# --- Batched postmortem for large trace sets ---


async def _summarize_one_trace(
    result: ForecastResult,
    client: anthropic.AsyncAnthropic,
) -> dict:
    """Phase 1: Use Haiku to produce a compact summary of one trace."""
    prob = result.calibrated_probability or result.raw_probability
    community = result.question.community_prediction_final
    divergence = prob - community

    # Build a concise representation of the full trace
    agent_summaries = []
    for trace in result.agent_traces:
        agent_summaries.append(
            f"Agent {trace.agent_id}: base_rate={trace.base_rate.base_rate_estimate:.2%}, "
            f"inside_view={trace.inside_view.inside_view_estimate:.2%}, "
            f"final={trace.synthesis.final_probability:.2%}, "
            f"confidence={trace.synthesis.confidence_level}, "
            f"top_findings={'; '.join(trace.research.key_findings[:2])}"
        )

    full_trace = (
        f"Question: {result.question.title}\n"
        f"Domain: {result.question.domain.value}\n"
        f"Pipeline: {prob:.2%} | Community: {community:.2%} | Divergence: {divergence:+.4f}\n"
        f"Supervisor reconciled: {result.supervisor.reconciled_probability:.2%}\n"
        f"Disagreements: {'; '.join(result.supervisor.disagreements[:3])}\n"
        f"Reasoning: {result.supervisor.reconciliation_reasoning[:300]}\n"
        + "\n".join(agent_summaries)
    )

    response = await client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=256,
        temperature=0.0,
        system=(
            "Summarize this forecast trace in ~100 words. Include: raw probability, "
            "community probability, divergence direction/magnitude, key reasoning points, "
            "what evidence was found or missed, and any notable agent disagreements. "
            "Be factual and concise."
        ),
        messages=[{"role": "user", "content": full_trace}],
    )

    return {
        "question_id": result.question.question_id,
        "question_title": result.question.title,
        "domain": result.question.domain.value,
        "url": result.question.url,
        "raw_probability": prob,
        "community_prediction": community,
        "divergence": divergence,
        "outcome": result.question.outcome,
        "summary": response.content[0].text,
    }


async def _run_batch_postmortem(
    summaries: list[dict],
    client: anthropic.AsyncAnthropic,
    batch_num: int,
) -> dict:
    """Phase 2: Use Opus to analyze a batch of summaries."""
    summaries_text = []
    for i, s in enumerate(summaries, 1):
        resolved_str = "Resolved" if s["outcome"] is not None else "Unresolved"
        summaries_text.append(
            f"### Question {i}: {s['question_title']}\n"
            f"Domain: {s['domain']} | {resolved_str}\n"
            f"Pipeline: {s['raw_probability']:.2%} | Community: {s['community_prediction']:.2%} | "
            f"Divergence: {s['divergence']:+.4f}\n"
            f"{s['summary']}\n"
        )

    user_msg = (
        f"# Batch {batch_num} Postmortem ({len(summaries)} questions)\n\n"
        + "\n".join(summaries_text)
        + "\n\n## Instructions\n\n"
        "For this batch, provide:\n"
        "1. A process classification for each question (good_process_good_outcome, "
        "good_process_bad_outcome, bad_process_good_outcome, bad_process_bad_outcome). "
        "For unresolved questions, use community divergence > 0.15 as threshold for 'bad outcome'.\n"
        "2. Batch-level pattern analysis: what systematic errors appear across multiple questions?\n"
        "3. 3-5 actionable methodology lessons for the batch (not per question). "
        "Lessons must be general forecasting methodology — no event-specific facts.\n\n"
        "Return your analysis as JSON with keys:\n"
        '- "per_question": list of {question_id, process_classification, process_reasoning}\n'
        '- "patterns": list of strings describing systematic patterns\n'
        '- "lessons": list of 3-5 actionable methodology lessons\n'
    )

    response = await client.messages.create(
        model=MODEL,
        max_tokens=4096,
        temperature=0.0,
        system=(
            "You are a senior superforecaster conducting batch postmortem analysis. "
            "Analyze the batch of forecast summaries and identify systematic patterns. "
            "Focus on methodology improvements, not event-specific facts. "
            "Return your response as valid JSON only, no markdown fencing."
        ),
        messages=[{"role": "user", "content": user_msg}],
    )

    import re
    text = response.content[0].text
    # Extract JSON from response
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


def _group_into_batches(
    summaries: list[dict],
    n_batches: int = 5,
) -> list[list[dict]]:
    """Group summaries into batches, keeping cluster members together."""
    # Identify clusters by URL pattern
    clusters: dict[str, list[dict]] = {}
    standalones: list[dict] = []

    for s in summaries:
        url = s["url"]
        # Extract cluster key from URL
        # Polymarket: https://polymarket.com/event/XXX/...
        # Kalshi: similar patterns with shared prefixes
        cluster_key = None
        if "polymarket.com/event/" in url:
            parts = url.split("polymarket.com/event/")[1].split("/")
            if parts:
                cluster_key = f"polymarket_{parts[0]}"
        elif "kalshi.com" in url:
            # Kalshi questions with same prefix are clustered
            qid = s.get("question_title", "")
            # Use the question ID prefix (e.g., EVSHARE-30JAN)
            import re
            match = re.match(r'([A-Z]+-\d+[A-Z]*)', url.split("/")[-1] if "/" in url else "")
            if not match:
                # Try from the file-level id
                pass

        if cluster_key:
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            clusters[cluster_key].append(s)
        else:
            standalones.append(s)

    # Merge all items into roughly equal batches
    all_groups = list(clusters.values()) + [[s] for s in standalones]
    # Sort groups by size descending for better packing
    all_groups.sort(key=len, reverse=True)

    batches: list[list[dict]] = [[] for _ in range(n_batches)]
    for group in all_groups:
        # Add to smallest batch
        smallest = min(range(n_batches), key=lambda i: len(batches[i]))
        batches[smallest].extend(group)

    # Remove empty batches
    return [b for b in batches if b]


async def run_batched_postmortems(
    results: list[ForecastResult],
    n_batches: int = 5,
) -> tuple[list[PostmortemOutput], list[dict]]:
    """Run batched postmortems: Haiku summarization → Opus batch analysis.

    Returns (postmortem_outputs, batch_findings).
    """
    client = anthropic.AsyncAnthropic(timeout=180.0)

    # Phase 1: Haiku summaries (parallel)
    print(f"Phase 1: Summarizing {len(results)} traces with Haiku...")
    summaries = list(await asyncio.gather(*[
        _summarize_one_trace(result, client)
        for result in results
    ]))
    print(f"Phase 1 complete: {len(summaries)} summaries produced.")

    # Phase 2: Group and run Opus batch analysis
    batches = _group_into_batches(summaries, n_batches)
    print(f"Phase 2: Running {len(batches)} batch postmortems with Opus...")
    print(f"  Batch sizes: {[len(b) for b in batches]}")

    batch_results = list(await asyncio.gather(*[
        _run_batch_postmortem(batch, client, i + 1)
        for i, batch in enumerate(batches)
    ]))
    print(f"Phase 2 complete.")

    # Convert batch results to PostmortemOutput objects
    # Build lookup from question_id to summary
    summary_by_id = {s["question_id"]: s for s in summaries}

    postmortems = []
    batch_findings = []

    for batch_result, batch_summaries in zip(batch_results, batches):
        # Collect batch-level findings
        batch_findings.append({
            "patterns": batch_result.get("patterns", []),
            "lessons": batch_result.get("lessons", []),
        })

        per_question = batch_result.get("per_question", [])
        pq_by_id = {pq["question_id"]: pq for pq in per_question}

        for s in batch_summaries:
            qid = s["question_id"]
            pq = pq_by_id.get(qid, {})

            # Map classification string to enum
            classification_str = pq.get("process_classification", "bad_process_bad_outcome")
            try:
                classification = ProcessClassification(classification_str)
            except ValueError:
                classification = ProcessClassification.BAD_PROCESS_BAD_OUTCOME

            from .types import Domain
            pm = PostmortemOutput(
                question_id=qid,
                question_title=s["question_title"],
                outcome=s["outcome"],
                forecast_probability=s["raw_probability"],
                brier_score=None if s["outcome"] is None else (s["raw_probability"] - s["outcome"]) ** 2,
                community_prediction=s["community_prediction"],
                divergence=s["divergence"],
                process_classification=classification,
                process_reasoning=pq.get("process_reasoning", ""),
                lessons=batch_result.get("lessons", []),
                domain=Domain(s["domain"]),
            )
            postmortems.append(pm)

    return postmortems, batch_findings


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

    # Deduplicate within each domain
    for domain in domain_lessons:
        domain_lessons[domain] = list(dict.fromkeys(domain_lessons[domain]))

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
