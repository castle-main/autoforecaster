"""Autoresearcher: self-improvement via prompt editing with A/B testing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

import anthropic
import numpy as np

from .orchestrator import _forecast_one
from .calibrate import load_params
from .types import (
    ChangelogEntry,
    Domain,
    ForecastResult,
    PostmortemOutput,
    PROJECT_ROOT,
)
from .utils import extract_json, load_jsonl, load_memory, load_program

MODEL = "claude-opus-4-6"


def _save_changelog_entry(entry: ChangelogEntry) -> None:
    path = PROJECT_ROOT / "logs" / "changelog.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(entry.model_dump_json() + "\n")


async def _analyze_error_patterns(
    postmortems: list[PostmortemOutput],
    changelog: list[dict],
    program: str,
) -> Optional[dict]:
    """Analyze postmortems for recurring error patterns and propose a prompt change.

    Returns dict with: target_file, new_content, change_description, error_pattern, bias_domain
    Or None if no change proposed.
    """
    client = anthropic.AsyncAnthropic(timeout=120.0)

    # Format postmortems
    pm_text = "\n\n".join([
        f"**{pm.question_title}** (domain: {pm.domain.value})\n"
        f"Classification: {pm.process_classification.value}\n"
        f"Brier: {pm.brier_score:.4f}\n"
        f"Lessons: {', '.join(pm.lessons)}"
        for pm in postmortems
    ])

    # Format recent changelog
    changelog_text = "\n".join([
        f"- Batch {e.get('batch_id')}: {e.get('target_file')} — {e.get('change_description')} ({'accepted' if e.get('accepted') else 'rejected'})"
        for e in changelog[-10:]
    ]) or "No previous changes."

    # Load current prompts
    prompts_dir = PROJECT_ROOT / "prompts"
    prompt_files = {}
    for pf in sorted(prompts_dir.glob("*.md")):
        prompt_files[pf.name] = pf.read_text()

    prompts_text = "\n\n---\n\n".join([
        f"### {name}\n```\n{content}\n```"
        for name, content in prompt_files.items()
    ])

    system = f"""You are the autoresearcher for a forecasting system. Your job is to improve the system's prompts based on error patterns in postmortem analyses.

## Constitution (program.md)
{program}

## Rules
- Only modify files in prompts/
- One file per change
- Must identify a pattern with at least 3 occurrences
- Check changelog for previously rejected changes — don't repeat them
- Prefer modifying earlier pipeline stages (decompose, research) as they have more leverage
- Changes must be specific and testable

## Current Prompts
{prompts_text}

## Recent Changelog
{changelog_text}"""

    response = await client.messages.create(
        model=MODEL,
        max_tokens=4096,
        temperature=0.0,
        system=system,
        messages=[{"role": "user", "content": (
            f"Here are the postmortems from the latest batch:\n\n{pm_text}\n\n"
            f"Analyze the error patterns. If you find a recurring pattern (3+ occurrences) "
            f"that could be addressed by modifying a prompt, propose the change.\n\n"
            f"Respond with a JSON object:\n"
            f'{{"propose_change": true, "target_file": "prompts/X.md", "new_content": "...", '
            f'"change_description": "...", "error_pattern": "...", "bias_domain": "domain_name"}}\n\n'
            f"Or if no change is warranted:\n"
            f'{{"propose_change": false, "reasoning": "..."}}'
        )}],
    )

    text = response.content[0].text
    try:
        proposal = extract_json(text)
    except json.JSONDecodeError:
        return None

    if not proposal.get("propose_change", False):
        return None

    return proposal


async def _ab_test(
    target_file: str,
    new_content: str,
    existing_results: list[ForecastResult],
) -> tuple[float, float]:
    """A/B test a prompt change using saved results for the old prompt.

    existing_results are from the batch that just ran (old prompt).
    Only the new prompt is re-run on the same questions.
    Returns (brier_old, brier_new).
    """
    # Compute brier_old directly from saved results — no re-run needed
    brier_old = float(np.mean([
        (r.raw_probability - r.question.outcome) ** 2 for r in existing_results
    ]))

    test_questions = [r.question for r in existing_results]

    memory = load_memory()

    platt_params = load_params()
    target_path = PROJECT_ROOT / target_file

    # Save original content
    original_content = target_path.read_text()

    # Swap in new prompt
    target_path.write_text(new_content)

    # Run with NEW prompt (parallel)
    try:
        new_results = list(await asyncio.gather(*[
            _forecast_one(q, memory, platt_params) for q in test_questions
        ]))

        brier_new = float(np.mean([
            (r.raw_probability - r.question.outcome) ** 2 for r in new_results
        ]))
    finally:
        # Restore original prompt (will be overwritten if accepted)
        target_path.write_text(original_content)

    return brier_old, brier_new


async def run_autoresearcher(
    postmortems: list[PostmortemOutput],
    batch_id: int,
    results: list[ForecastResult],
    ab_size: int | None = None,
) -> Optional[ChangelogEntry]:
    """Main autoresearcher entry point: analyze errors, propose change, A/B test, accept/reject."""
    program = load_program()
    changelog = load_jsonl(PROJECT_ROOT / "logs" / "changelog.jsonl")

    # Step 1: Analyze error patterns and propose change
    proposal = await _analyze_error_patterns(postmortems, changelog, program)
    if not proposal:
        return None

    target_file = proposal["target_file"]
    new_content = proposal["new_content"]
    change_description = proposal["change_description"]
    error_pattern = proposal["error_pattern"]

    print(f"  Proposed change to {target_file}: {change_description}")

    # Skip A/B testing when ab_size is 0 — apply change directly
    if ab_size == 0:
        target_path = PROJECT_ROOT / target_file
        target_path.write_text(new_content)
        print(f"  APPLIED (A/B testing disabled)")
        entry = ChangelogEntry(
            batch_id=batch_id,
            target_file=target_file,
            change_description=change_description,
            error_pattern=error_pattern,
            accepted=True,
            diff_summary=f"Applied change to {target_file} (no A/B test)",
        )
        _save_changelog_entry(entry)
        return entry

    # Subset results for A/B test if ab_size is set
    ab_results = results
    if ab_size is not None and ab_size < len(results):
        # Bias sampling toward the domain where the error pattern was found
        bias_domain = proposal.get("bias_domain")
        weights = np.array([
            2.0 if (bias_domain and r.question.domain and r.question.domain.value == bias_domain) else 1.0
            for r in results
        ])
        weights /= weights.sum()
        indices = np.random.choice(len(results), size=ab_size, replace=False, p=weights)
        ab_results = [results[i] for i in indices]

    # Step 2: A/B test using saved results for old prompt
    print(f"  Running A/B test ({len(ab_results)} questions, using saved results for old prompt)...")
    brier_old, brier_new = await _ab_test(target_file, new_content, ab_results)
    print(f"  A/B result: old={brier_old:.4f}, new={brier_new:.4f}")

    # Step 3: Accept or reject
    accepted = brier_new < brier_old
    if accepted:
        target_path = PROJECT_ROOT / target_file
        target_path.write_text(new_content)
        print(f"  ACCEPTED — Brier improved by {brier_old - brier_new:.4f}")
    else:
        print(f"  REJECTED — Brier worsened by {brier_new - brier_old:.4f}")

    # Step 4: Log decision
    entry = ChangelogEntry(
        batch_id=batch_id,
        target_file=target_file,
        change_description=change_description,
        error_pattern=error_pattern,
        brier_old=brier_old,
        brier_new=brier_new,
        accepted=accepted,
        diff_summary=f"{'Applied' if accepted else 'Reverted'} change to {target_file}",
    )
    _save_changelog_entry(entry)

    return entry
