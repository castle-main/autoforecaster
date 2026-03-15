"""Pydantic models for all pipeline stages and data utilities."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).parent.parent


class Domain(str, Enum):
    CONFLICT_SECURITY = "conflict_security"
    CLIMATE_ENERGY = "climate_energy"
    DIPLOMACY = "diplomacy"
    ELECTIONS = "elections"
    OTHER_POLICY = "other_policy"
    REGULATION_POLICY = "regulation_policy"
    TECHNOLOGY_POLICY = "technology_policy"
    TRADE_ECONOMICS = "trade_economics"


class Question(BaseModel):
    post_id: int
    question_id: int
    id: str
    source: str
    url: str
    title: str
    description: str
    resolution_criteria: str
    created_date: str
    close_date: str
    resolved_date: str
    outcome: int  # 0 or 1
    community_prediction_final: float
    num_forecasters: int
    tags: list[str]
    domain: Domain


class SearchResult(BaseModel):
    query: str
    content: str
    citations: list[str] = Field(default_factory=list)
    filtered_citations: list[str] = Field(default_factory=list, description="Citations removed due to date filtering")
    was_contaminated: bool = Field(default=False, description="Whether outcome-leaking content was detected and redacted")


class SearchRound(BaseModel):
    round_number: int
    query: str
    result: SearchResult
    reasoning: str = Field(description="Why this query was chosen")


class SearchTrace(BaseModel):
    rounds: list[SearchRound]
    total_citations: int = 0


class DecomposeOutput(BaseModel):
    sub_questions: list[str] = Field(description="Key sub-questions to investigate")
    initial_search_queries: list[str] = Field(description="Search queries to start research")
    reasoning: str = Field(description="Why these sub-questions matter")


class ResearchOutput(BaseModel):
    key_findings: list[str] = Field(description="Most important findings from research")
    evidence_for: list[str] = Field(description="Evidence supporting Yes resolution")
    evidence_against: list[str] = Field(description="Evidence supporting No resolution")
    search_trace: SearchTrace
    information_gaps: list[str] = Field(description="What we still don't know")


class BaseRateOutput(BaseModel):
    reference_classes: list[str] = Field(description="Reference classes considered")
    base_rate_estimate: float = Field(ge=0.0, le=1.0, description="Base rate probability")
    reasoning: str = Field(description="How base rate was derived")


class InsideViewOutput(BaseModel):
    factors_for: list[str] = Field(description="Factors favoring Yes")
    factors_against: list[str] = Field(description="Factors favoring No")
    inside_view_estimate: float = Field(ge=0.0, le=1.0)
    reasoning: str


class SynthesisOutput(BaseModel):
    base_rate_weight: float = Field(description="Weight given to base rate vs inside view")
    adjustment_reasoning: str = Field(description="How base rate and inside view were combined")
    final_probability: float = Field(ge=0.0, le=1.0)
    confidence_reasoning: str = Field(description="Why this probability and not higher/lower")


class AgentTrace(BaseModel):
    agent_id: int
    question_id: int
    model_id: str = "claude-opus-4-6"
    decompose: DecomposeOutput
    research: ResearchOutput
    base_rate: BaseRateOutput
    inside_view: InsideViewOutput
    synthesis: SynthesisOutput
    raw_probability: float = Field(ge=0.0, le=1.0, description="Denormalized from synthesis.final_probability")


class SupervisorOutput(BaseModel):
    agent_probabilities: list[float] = Field(description="Raw probabilities from each agent")
    disagreements: list[str] = Field(description="Key disagreements identified")
    targeted_searches: list[SearchResult] = Field(default_factory=list)
    reconciliation_reasoning: str = Field(description="How disagreements were resolved")
    reconciled_probability: float = Field(ge=0.0, le=1.0)


class ForecastResult(BaseModel):
    question: Question
    agent_traces: list[AgentTrace]
    supervisor: SupervisorOutput
    raw_probability: float = Field(ge=0.0, le=1.0, description="Supervisor's reconciled probability")
    calibrated_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    brier_raw: Optional[float] = None
    brier_calibrated: Optional[float] = None


class PlattParams(BaseModel):
    a: float
    b: float
    n_samples: int
    brier_before: Optional[float] = None
    brier_after: Optional[float] = None


class ProcessClassification(str, Enum):
    GOOD_PROCESS_GOOD_OUTCOME = "good_process_good_outcome"
    GOOD_PROCESS_BAD_OUTCOME = "good_process_bad_outcome"
    BAD_PROCESS_GOOD_OUTCOME = "bad_process_good_outcome"
    BAD_PROCESS_BAD_OUTCOME = "bad_process_bad_outcome"


class PostmortemOutput(BaseModel):
    question_id: int
    question_title: str
    outcome: int
    forecast_probability: float
    brier_score: float
    process_classification: ProcessClassification
    process_reasoning: str = Field(description="Why this process classification")
    lessons: list[str] = Field(description="Actionable lessons for future forecasts")
    domain: Domain


class BatchResult(BaseModel):
    batch_id: int
    n_questions: int
    brier_pipeline: float
    brier_community: float
    brier_naive: float
    brier_by_domain: dict[str, float] = Field(default_factory=dict)
    brier_by_agent: dict[int, float] = Field(default_factory=dict)
    contamination_flags: list[int] = Field(default_factory=list, description="Question IDs flagged for contamination")
    contamination_rate: float = Field(default=0.0, description="Fraction of search rounds where outcome leakage was detected")
    platt_params: Optional[PlattParams] = None


class ChangelogEntry(BaseModel):
    batch_id: int
    target_file: str
    change_description: str
    error_pattern: str
    brier_old: float
    brier_new: float
    accepted: bool
    diff_summary: str


class RunSummary(BaseModel):
    start_time: str
    end_time: str
    duration_seconds: float
    batches_completed: list[int]
    final_brier_pipeline: Optional[float] = None
    final_brier_community: Optional[float] = None
    final_brier_naive: Optional[float] = None
    plot_path: Optional[str] = None


# --- Data utilities ---

def load_questions(path: Optional[Path] = None) -> list[Question]:
    """Load all questions from questions.jsonl."""
    path = path or PROJECT_ROOT / "data" / "questions.jsonl"
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(Question.model_validate_json(line))
    return questions


def sample_batch(questions: list[Question], batch_id: int, batch_size: int = 12) -> list[Question]:
    """Get a deterministic batch of questions by batch_id."""
    # Sort by question_id for deterministic ordering
    sorted_qs = sorted(questions, key=lambda q: q.question_id)
    start = batch_id * batch_size
    end = start + batch_size
    if start >= len(sorted_qs):
        raise ValueError(f"Batch {batch_id} exceeds dataset size ({len(sorted_qs)} questions)")
    return sorted_qs[start:end]
