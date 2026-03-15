# Build Log

## Phase 1: Types + Data Layer
**Files:** `autoforecast/__init__.py`, `autoforecast/types.py`
**Status:** PASS

- Built all pydantic models: Question, SearchResult, SearchRound, SearchTrace, DecomposeOutput, ResearchOutput, BaseRateOutput, InsideViewOutput, SynthesisOutput, AgentTrace, SupervisorOutput, ForecastResult, PlattParams, ProcessClassification, PostmortemOutput, BatchResult, ChangelogEntry, Domain enum
- Domain enum corrected to match actual data: conflict_security, climate_energy, diplomacy, elections, other_policy, regulation_policy, technology_policy, trade_economics
- Data utilities: load_questions(), sample_batch() (deterministic by batch_id), sample_ab_questions() (domain-biased with exclusion)
- Tests: loaded all 586 questions, verified domain/outcome distributions, batch sampling non-overlap, round-trip serialization

## Phase 2: Search Wrapper
**Files:** `autoforecast/search.py`
**Status:** PASS

- Perplexity `sonar-reasoning-pro` via httpx async client
- Date gating: prepends "Before {date}" to query, post-hoc filters citations with URL date patterns
- Temperature 0.0 for deterministic search results
- Test: real API call returned 1221 chars, 5 citations (1 filtered for post-close date)

## Phase 3: Single Agent Pipeline
**Files:** `autoforecast/agent.py`, `prompts/{decompose,research,base_rate,inside_view,synthesize}.md`
**Status:** PASS

- 5-stage pipeline using Anthropic tool_use for structured output
- Agentic search: initial queries from decompose, then LLM-driven follow-up queries up to 5 rounds
- Temperature 1.0 for agent diversity
- Memory injected into every stage prompt
- Test: 6 sub-questions, 4 search rounds, final probability 99% (question was about Trump saying "AI" in SOTU)

## Phase 4: Supervisor
**Files:** `autoforecast/supervisor.py`, `prompts/supervisor.md`
**Status:** PASS

- Two-step process: identify disagreements + decide on searches, then reconcile
- Temperature 0.0 for deterministic judgment
- Max 3 targeted search rounds
- Test: 3 agents (99%, 96%, 97%), supervisor identified 4 disagreements, ran 3 targeted searches, reconciled at 95%

## Phase 5: Calibration
**Files:** `autoforecast/calibrate.py`
**Status:** PASS

- LogisticRegression(C=1e10) on logit-transformed probabilities
- Clamping at [0.001, 0.999] before logit
- Minimum 10 samples for fit
- Test: synthetic biased data, Brier improved 0.2481 → 0.2352, edge cases (0.0, 0.5, 1.0) all valid

## Phase 6: Orchestrator (Backtest)
**Files:** `autoforecast/orchestrator.py`, `autoforecast/__main__.py`
**Status:** PASS

- Sequential per question, 3 agents parallel per question
- Platt scaling applied if params exist, passthrough if not
- Traces saved as JSON to logs/traces/{batch_id}_{question_id}.json
- CLI: `python -m autoforecast <start_batch> <num_batches>` or `--eval <batch_id>`
- Test: 2 questions, raw=97%/50%, calibrated=92%/39%, both traces saved

## Phase 7: Eval + Postmortem
**Files:** `autoforecast/eval.py`, `autoforecast/postmortem.py`, `prompts/postmortem.md`
**Status:** PASS

- Eval: Brier for pipeline (0.1911), community (0.2522), naive LLM (0.4004)
- Contamination detection: flags when naive beats community by >0.15
- Postmortems: independent process/outcome classification per Tetlock
- Memory updated with tagged lessons (<!-- source: {question_id} --> for A/B filtering)
- Test: correct classifications (bad_process_bad_outcome, good_process_good_outcome), memory.md populated

## Phase 8: Autoresearcher
**Files:** `autoforecast/autoresearcher.py`, `program.md`
**Status:** PASS

- Error pattern analysis: requires 3+ occurrences
- A/B test: 10 questions (60% from target domain), filtered memory
- Accept only if Brier improves
- Changelog logging for all decisions
- Test: given 5 fake postmortems with clear pattern, proposed change to decompose.md (earlier stage, higher leverage)

## Phase 9: Continuous Runner + Plotting
**Files:** `autoforecast/orchestrator.py` (modified), `autoforecast/types.py` (modified), `autoforecast/plot.py` (new)
**Status:** COMPLETE

- Added `RunSummary` pydantic model to types.py for persisting run metadata
- Added `deadline` parameter to `backtest_batch()` for time-bounded partial batches — eval/postmortem work on any-length result list
- Added `results` parameter to `run_eval()` to skip redundant disk I/O when called from continuous loop; returns `BatchResult`
- Added `run_continuous()`: loops backtest → eval → postmortem → autoresearcher for configurable duration (default 2h)
  - Graceful Ctrl+C: catches KeyboardInterrupt, saves progress, generates plot
  - Reloads memory after each batch (postmortem updates it)
  - Saves `RunSummary` to `logs/run_summary.json`
- Created `plot.py` with two functions:
  - `plot_brier_scores()`: pipeline (blue), community (red dashed), naive (gray dotted), plus cumulative average
  - `plot_domain_breakdown()`: one line per domain
  - Matplotlib imported lazily inside functions
- CLI additions:
  - `python -m autoforecast --run [hours]` — continuous run (default 2h)
  - `python -m autoforecast --run [hours] --start N` — start from batch N
  - `python -m autoforecast --plot` — generate plots from existing score data
