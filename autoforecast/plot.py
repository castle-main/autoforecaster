"""Brier score plotting for AutoForecast."""

from __future__ import annotations

import json
from pathlib import Path

from .types import BatchResult, PROJECT_ROOT


def _load_batch_results() -> list[BatchResult]:
    """Load all batch score files, sorted by batch_id."""
    scores_dir = PROJECT_ROOT / "logs" / "scores"
    if not scores_dir.exists():
        raise FileNotFoundError(f"No scores directory at {scores_dir}")

    results = []
    for score_file in sorted(scores_dir.glob("batch_*.json")):
        with open(score_file) as f:
            data = json.load(f)
        results.append(BatchResult.model_validate(data))

    if not results:
        raise FileNotFoundError(f"No batch score files in {scores_dir}")

    results.sort(key=lambda r: r.batch_id)
    return results


def plot_brier_scores(output_path: Path | None = None) -> Path:
    """Plot pipeline vs community vs naive Brier scores over batches."""
    import matplotlib.pyplot as plt

    results = _load_batch_results()

    batch_ids = [r.batch_id for r in results]
    pipeline = [r.brier_pipeline for r in results]
    community = [r.brier_community for r in results]
    naive = [r.brier_naive for r in results]

    # Cumulative average for pipeline
    cumulative_avg = []
    running_sum = 0.0
    for i, val in enumerate(pipeline):
        running_sum += val
        cumulative_avg.append(running_sum / (i + 1))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(batch_ids, pipeline, "b-o", label="Pipeline", linewidth=2, markersize=5)
    ax.plot(batch_ids, community, "r--s", label="Community", linewidth=2, markersize=5)
    ax.plot(batch_ids, naive, "gray", linestyle=":", marker="^", label="Naive LLM", linewidth=2, markersize=5)
    ax.plot(batch_ids, cumulative_avg, "lightblue", linestyle="-", label="Pipeline (cumulative avg)", linewidth=1.5)

    ax.set_xlabel("Batch ID")
    ax.set_ylabel("Brier Score (lower is better)")
    ax.set_title("AutoForecast: Brier Scores Over Batches")
    ax.legend()
    ax.grid(True, alpha=0.3)

    output_path = output_path or PROJECT_ROOT / "logs" / "brier_scores.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path


def plot_domain_breakdown(output_path: Path | None = None) -> Path:
    """Plot Brier scores by domain across batches."""
    import matplotlib.pyplot as plt

    results = _load_batch_results()

    # Collect all domains
    all_domains: set[str] = set()
    for r in results:
        all_domains.update(r.brier_by_domain.keys())

    fig, ax = plt.subplots(figsize=(12, 6))

    for domain in sorted(all_domains):
        batch_ids = []
        scores = []
        for r in results:
            if domain in r.brier_by_domain:
                batch_ids.append(r.batch_id)
                scores.append(r.brier_by_domain[domain])
        if batch_ids:
            ax.plot(batch_ids, scores, "-o", label=domain.replace("_", " ").title(), markersize=4)

    ax.set_xlabel("Batch ID")
    ax.set_ylabel("Brier Score (lower is better)")
    ax.set_title("AutoForecast: Brier Scores by Domain")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    output_path = output_path or PROJECT_ROOT / "logs" / "brier_by_domain.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_path
