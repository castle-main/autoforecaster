---
name: restart-training
description: Reset the autoforecaster to a clean state for fresh continuous training
user-invocable: true
allowed-tools: Bash, Write, Read
---

# Restart Training

Reset all training artifacts so the pipeline can run from scratch. Execute these steps:

1. **Empty trace logs**: `rm -f logs/traces/*.json`
2. **Empty score logs**: `rm -f logs/scores/*.json`
3. **Truncate changelog**: `truncate -s 0 logs/changelog.jsonl` (if it exists)
4. **Reset memory.md**: Write an empty file to `memory.md` in the project root
5. **Reset run_summary.json**: Write the following to `logs/run_summary.json`:
```json
{
  "start_time": null,
  "end_time": null,
  "duration_seconds": null,
  "batches_completed": [],
  "final_brier_pipeline": null,
  "final_brier_community": null,
  "final_brier_naive": null,
  "plot_path": null
}
```
6. **Reset Platt params**: Write the following to `data/platt_params.json`:
```json
{
  "a": 1.7320508075688772,
  "b": 0,
  "n_samples": 0,
  "brier_before": null,
  "brier_after": null
}
```

After all steps, print a confirmation summary listing each item that was reset.
