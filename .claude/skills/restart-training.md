---
name: restart-training
description: Clear all logs, memory, and results from the autoforecaster to enable a clean-slate run.
user_invocable: true
---

# Restart Training

Clear all autoforecaster output so the pipeline can run again from scratch.

## Steps

1. Delete all trace files: `logs/traces/*.json`
2. Delete all score files: `logs/scores/*.json`
3. Delete the changelog: `logs/changelog.jsonl`
4. Delete the memory file: `autoforecast/memory.md`
5. Delete fitted calibration params: `autoforecast/data/platt_params.json`
6. Confirm what was deleted and what was already clean.

## Rules

- Only delete generated output files listed above. Never delete source code, prompts, questions.jsonl, or program.md.
- Preserve the directory structure (logs/traces/, logs/scores/, autoforecast/data/) — only remove file contents.
- Do NOT ask for confirmation — the user invoked this skill explicitly.
