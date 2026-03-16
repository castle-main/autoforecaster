---
name: restart-testing
description: Clear all test traces and results so testing can run from scratch
user-invocable: true
allowed-tools: Bash, Read
---

# Restart Testing

Reset all testing artifacts so tests can run from scratch. Execute these steps:

1. **Empty test traces**: `rm -f logs/test_traces/*.json`
2. **Remove test results**: `rm -f logs/test_results.json`
3. **Remove test calibration report**: `rm -f logs/test_calibration.html`

After all steps, print a confirmation summary listing each item that was reset.
