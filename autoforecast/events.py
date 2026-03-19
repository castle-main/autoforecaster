"""Lightweight event system for pipeline monitoring."""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, Field


class EventType(str, Enum):
    QUESTION_START = "question_start"
    QUESTION_DONE = "question_done"
    AGENT_STAGE_START = "agent_stage_start"
    AGENT_STAGE_DONE = "agent_stage_done"
    SEARCH_START = "search_start"
    SEARCH_DONE = "search_done"
    SUPERVISOR_START = "supervisor_start"
    SUPERVISOR_SEARCH = "supervisor_search"
    SUPERVISOR_DONE = "supervisor_done"
    CALIBRATION_DONE = "calibration_done"
    BATCH_PROGRESS = "batch_progress"
    PHASE_CHANGE = "phase_change"  # data: {"phase": "initial"|"eval"|"random_batches", "batches_completed": int}
    API_COST = "api_cost"  # data: {"provider": "anthropic"|"perplexity", "cost_usd": float, "input_tokens": int, "output_tokens": int}


class Stage(str, Enum):
    DECOMPOSE = "decompose"
    RESEARCH = "research"
    BASE_RATE = "base_rate"
    INSIDE_VIEW = "inside_view"
    SYNTHESIZE = "synthesize"
    SUPERVISOR = "supervisor"


class PipelineEvent(BaseModel):
    event_type: EventType
    question_id: int | None = None
    question_title: str = ""
    agent_id: int | None = None
    stage: Stage | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class EventHandler(Protocol):
    async def handle(self, event: PipelineEvent) -> None: ...


class NullHandler:
    """No-op handler. Zero overhead when UI is off."""

    async def handle(self, event: PipelineEvent) -> None:
        pass


# Pricing per million tokens (USD)
PRICING = {
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    "sonar-reasoning-pro": {"input": 2.0, "output": 8.0},
    "gpt-5.4": {"input": 10.0, "output": 30.0},
    "gemini-3.1-pro-preview": {"input": 1.25, "output": 10.0},
}


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute USD cost from token counts."""
    prices = PRICING.get(model, {"input": 15.0, "output": 75.0})
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000


async def track_api_cost(
    handler, provider: str, model: str, input_tokens: int, output_tokens: int,
) -> None:
    """Emit an API_COST event if handler is present."""
    if not handler:
        return
    cost = compute_cost(model, input_tokens, output_tokens)
    await handler.handle(PipelineEvent(
        event_type=EventType.API_COST,
        data={"provider": provider, "model": model, "cost_usd": cost,
              "input_tokens": input_tokens, "output_tokens": output_tokens},
    ))

