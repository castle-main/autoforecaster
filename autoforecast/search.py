"""Search wrapper using Perplexity API with date gating and contamination scanning."""

from __future__ import annotations

import os
from datetime import datetime

import anthropic
import httpx

from .events import EventType, NullHandler, PipelineEvent, compute_cost
from .types import SearchResult

PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
DECONTAMINATION_MODEL = "claude-haiku-4-5-20251001"

# Hardcoded in search.py intentionally — NOT in prompts/ — so the autoresearcher
# cannot weaken or remove contamination checks.
_DECONTAMINATION_PROMPT = """\
You are a contamination scanner for a forecasting backtesting system.

Given a forecasting question, its close date, and search result content, your job is to \
identify any sentences that reveal what actually happened AFTER the close date — i.e., \
the resolution or outcome of the question.

Replace ONLY those sentences with: [REDACTED: outcome information removed]

DO NOT remove:
- Background context or history before the close date
- Analysis, opinions, or predictions made before the close date
- General factual information that doesn't reveal the outcome

Preserve the rest of the content exactly as-is. Return the full content with only the \
contaminated sentences replaced."""


async def _decontaminate_content(
    content: str,
    question_title: str,
    close_date: str,
    handler=None,
) -> tuple[str, bool]:
    """Scan search content for outcome leakage and redact contaminated passages.

    Returns (cleaned_content, was_contaminated).
    """
    client = anthropic.AsyncAnthropic(timeout=60.0)

    response = await client.messages.create(
        model=DECONTAMINATION_MODEL,
        max_tokens=4096,
        temperature=0.0,
        system=_DECONTAMINATION_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Forecasting question: {question_title}\n"
                f"Close date: {close_date}\n\n"
                f"Search content to scan:\n{content}"
            ),
        }],
    )

    # Track cost
    _handler = handler or NullHandler()
    if response.usage:
        cost = compute_cost(DECONTAMINATION_MODEL, response.usage.input_tokens, response.usage.output_tokens)
        await _handler.handle(PipelineEvent(
            event_type=EventType.API_COST,
            data={"provider": "anthropic", "model": DECONTAMINATION_MODEL, "cost_usd": cost,
                  "input_tokens": response.usage.input_tokens,
                  "output_tokens": response.usage.output_tokens},
        ))

    cleaned = response.content[0].text
    was_contaminated = "[REDACTED: outcome information removed]" in cleaned

    if was_contaminated:
        await _handler.handle(PipelineEvent(
            event_type=EventType.SEARCH_DONE,
            data={"contamination_detected": True, "question_title": question_title},
        ))

    return cleaned, was_contaminated


async def execute_search(
    query: str,
    close_date: str,
    api_key: str | None = None,
    handler=None,
    question_title: str | None = None,
) -> SearchResult:
    """Execute a single search query via Perplexity API with date gating.

    Appends date constraint to query, post-hoc filters citations
    with known post-close dates, and runs LLM contamination scan.
    """
    api_key = api_key or os.environ["PERPLEXITY_API_KEY"]

    # Layer 1: Append date constraint to push filtering into the search index
    date_constrained_query = f"{query} before:{close_date}"

    payload = {
        "model": "sonar-reasoning-pro",
        "messages": [
            {
                "role": "system",
                "content": (
                    f"You are a research assistant. The current date is {close_date}. "
                    f"ONLY use sources published on or before {close_date}. "
                    f"Do NOT reference any events, data, or developments after {close_date}. "
                    f"If a source was published after {close_date}, ignore it completely. "
                    f"Answer as if you are living on {close_date} and have no knowledge of the future."
                ),
            },
            {"role": "user", "content": date_constrained_query},
        ],
        "temperature": 0.0,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # sonar-reasoning-pro can take a while; retry on transient timeouts
    async with httpx.AsyncClient(timeout=180.0) as client:
        for attempt in range(3):
            try:
                response = await client.post(PERPLEXITY_URL, json=payload, headers=headers)
                response.raise_for_status()
                break
            except httpx.ReadTimeout:
                if attempt == 2:
                    raise
                continue

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    citations = data.get("citations", [])

    # Track Perplexity cost
    _handler = handler or NullHandler()
    usage = data.get("usage", {})
    if usage:
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        cost = compute_cost("sonar-reasoning-pro", input_tokens, output_tokens)
        await _handler.handle(PipelineEvent(
            event_type=EventType.API_COST,
            data={"provider": "perplexity", "model": "sonar-reasoning-pro", "cost_usd": cost,
                  "input_tokens": input_tokens, "output_tokens": output_tokens},
        ))

    # Post-hoc filter: remove citations with dates after close_date
    close_dt = datetime.fromisoformat(close_date)
    valid_citations = []
    filtered_citations = []
    for url in citations:
        # Best-effort: if URL contains a recognizable date after close, filter it
        if _citation_appears_after_date(url, close_dt):
            filtered_citations.append(url)
        else:
            valid_citations.append(url)

    # Layer 2: LLM contamination scan and redaction
    was_contaminated = False
    if question_title:
        content, was_contaminated = await _decontaminate_content(
            content, question_title, close_date, handler=_handler,
        )

    return SearchResult(
        query=query,
        content=content,
        citations=valid_citations,
        filtered_citations=filtered_citations,
        was_contaminated=was_contaminated,
    )


def _citation_appears_after_date(url: str, close_dt: datetime) -> bool:
    """Best-effort check if a URL contains a date after the close date.

    Looks for common date patterns in URLs like /2026/03/15/ or /2026-03-15.
    """
    import re

    # Match YYYY/MM/DD or YYYY-MM-DD patterns in URLs
    patterns = [
        r'(\d{4})/(\d{2})/(\d{2})',
        r'(\d{4})-(\d{2})-(\d{2})',
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            try:
                year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
                url_dt = datetime(year, month, day)
                if url_dt > close_dt:
                    return True
            except ValueError:
                continue

    return False
