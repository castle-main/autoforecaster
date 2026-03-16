"""Cluster questions by same-event options for deduplicated research."""

from __future__ import annotations

from collections import defaultdict

from .types import Question


def _event_key(q: Question) -> str:
    """Extract event key from question ID/URL.

    Kalshi: strip last hyphen-delimited segment (candidate code) from ID.
    Polymarket: use shared event page URL.
    """
    if q.id.startswith("kalshi_"):
        # kalshi_KXFOO-BAR-CANDIDATE -> kalshi_KXFOO-BAR
        parts = q.id.rsplit("-", 1)
        return parts[0] if len(parts) > 1 else q.id
    elif q.id.startswith("polymarket_"):
        return q.url
    return q.id  # Fallback: no clustering (unique key per question)


def cluster_questions(
    questions: list[Question],
) -> tuple[dict[str, list[Question]], list[Question]]:
    """Group questions that are different options for the same event.

    Returns (event_key_to_questions, singletons). Only groups with 2+
    questions become clusters; singletons are questions with unique events.
    """
    groups: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        groups[_event_key(q)].append(q)

    clusters: dict[str, list[Question]] = {}
    singletons: list[Question] = []

    for key, qs in groups.items():
        if len(qs) >= 2:
            clusters[key] = qs
        else:
            singletons.extend(qs)

    return clusters, singletons


def select_with_clusters(
    clusters: dict[str, list[Question]],
    singletons: list[Question],
    max_questions: int,
) -> list[Question]:
    """Select up to max_questions, keeping clusters whole.

    Adds smallest clusters first until budget is spent, then fills with singletons.
    """
    selected: list[Question] = []

    # Sort clusters by size (smallest first) to fit more clusters within budget
    sorted_clusters = sorted(clusters.items(), key=lambda item: len(item[1]))

    for _key, qs in sorted_clusters:
        if len(selected) + len(qs) <= max_questions:
            selected.extend(qs)

    # Fill remaining budget with singletons
    remaining_budget = max_questions - len(selected)
    selected.extend(singletons[:remaining_budget])

    return selected
