#!/usr/bin/env python3
"""Run batched postmortem analysis on test traces."""

from __future__ import annotations

import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from autoforecast.types import ForecastResult, PROJECT_ROOT
from autoforecast.postmortem import (
    run_batched_postmortems,
    update_memory,
    consolidate_memory,
)


def load_test_traces(trace_dir: Path, exclude_ids: set[int] | None = None) -> list[ForecastResult]:
    """Load individual (non-cluster) test traces, optionally excluding already-analyzed IDs."""
    results = []
    for path in sorted(trace_dir.glob("*.json")):
        if path.name.startswith("cluster_"):
            continue
        with open(path) as f:
            data = json.load(f)
        qid = data.get("question", {}).get("question_id")
        if exclude_ids and qid in exclude_ids:
            continue
        try:
            result = ForecastResult.model_validate(data)
            results.append(result)
        except Exception as e:
            print(f"  Warning: skipping {path.name}: {e}")
    return results


def load_analyzed_ids(pm_path: Path) -> set[int]:
    """Load question IDs already analyzed from previous postmortem results."""
    if not pm_path.exists():
        return set()
    with open(pm_path) as f:
        existing = json.load(f)
    return {pm["question_id"] for pm in existing}


def print_summary(postmortems, batch_findings):
    """Print a summary of postmortem results."""
    print("\n" + "=" * 70)
    print("POSTMORTEM SUMMARY")
    print("=" * 70)

    # Overall stats
    print(f"\nTotal questions analyzed: {len(postmortems)}")

    # Classification breakdown
    from autoforecast.types import ProcessClassification
    class_counts = defaultdict(int)
    for pm in postmortems:
        class_counts[pm.process_classification.value] += 1
    print("\nProcess classifications:")
    for cls, count in sorted(class_counts.items()):
        print(f"  {cls}: {count}")

    # Domain breakdown
    domain_stats = defaultdict(lambda: {"count": 0, "total_div": 0.0})
    for pm in postmortems:
        d = pm.domain.value
        domain_stats[d]["count"] += 1
        if pm.divergence is not None:
            domain_stats[d]["total_div"] += pm.divergence

    print("\nDomain breakdown (avg divergence from community):")
    for domain, stats in sorted(domain_stats.items()):
        avg_div = stats["total_div"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {domain}: {stats['count']} questions, avg divergence: {avg_div:+.4f}")

    # Top divergence questions
    sorted_by_div = sorted(postmortems, key=lambda pm: abs(pm.divergence or 0), reverse=True)
    print("\nTop 10 largest divergences from community:")
    for pm in sorted_by_div[:10]:
        print(f"  {pm.divergence:+.4f} | {pm.forecast_probability:.2%} vs {pm.community_prediction:.2%} | {pm.question_title[:60]}")

    # Batch-level findings
    print("\n" + "-" * 70)
    print("BATCH-LEVEL FINDINGS")
    print("-" * 70)
    for i, findings in enumerate(batch_findings, 1):
        print(f"\nBatch {i}:")
        if findings.get("patterns"):
            print("  Patterns:")
            for p in findings["patterns"]:
                print(f"    - {p}")
        if findings.get("lessons"):
            print("  Lessons:")
            for l in findings["lessons"]:
                print(f"    - {l}")


async def main():
    trace_dir = PROJECT_ROOT / "logs" / "test_traces"
    output_path = PROJECT_ROOT / "logs" / "postmortem_results.json"
    findings_path = PROJECT_ROOT / "logs" / "batch_findings.json"

    # Check for --new-only flag
    new_only = "--new-only" in sys.argv

    if new_only:
        analyzed_ids = load_analyzed_ids(output_path)
        print(f"Skipping {len(analyzed_ids)} already-analyzed questions...")
        results = load_test_traces(trace_dir, exclude_ids=analyzed_ids)
    else:
        results = load_test_traces(trace_dir)

    print(f"Loaded {len(results)} traces to analyze.")

    if not results:
        print("No new traces found. Exiting.")
        return

    # Run batched postmortems
    n_batches = max(1, len(results) // 20)
    postmortems, batch_findings = await run_batched_postmortems(results, n_batches=n_batches)

    # Update memory
    print("\nUpdating memory.md...")
    update_memory(postmortems)

    print("Consolidating memory.md...")
    await consolidate_memory()

    # Print summary
    print_summary(postmortems, batch_findings)

    # Save postmortem results (append to existing if --new-only)
    if new_only and output_path.exists():
        with open(output_path) as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    new_data = [pm.model_dump() for pm in postmortems]
    for d in new_data:
        d["domain"] = d["domain"] if isinstance(d["domain"], str) else str(d["domain"])
        d["process_classification"] = (
            d["process_classification"] if isinstance(d["process_classification"], str)
            else str(d["process_classification"])
        )
    all_data = existing_data + new_data
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2, default=str)
    print(f"\nPostmortem results saved to {output_path} ({len(all_data)} total)")

    # Save batch findings
    if new_only and findings_path.exists():
        with open(findings_path) as f:
            existing_findings = json.load(f)
    else:
        existing_findings = []
    all_findings = existing_findings + batch_findings
    with open(findings_path, "w") as f:
        json.dump(all_findings, f, indent=2)
    print(f"Batch findings saved to {findings_path}")


if __name__ == "__main__":
    asyncio.run(main())
