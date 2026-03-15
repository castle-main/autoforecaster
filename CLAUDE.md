# AutoForecast

Autonomous forecasting system that produces calibrated probability estimates for geopolitical and economic events. Encodes Tetlock's superforecasting methodology as a multi-agent pipeline with agentic web search and self-improving prompts.

---

## Architecture

### Actors

| Actor | Role | Reads | Writes |
|---|---|---|---|
| Forecasting Agent (×3, parallel) | Execute 5-stage reasoning chain | prompts/*.md, memory.md | returns trace to orchestrator |
| Supervisor | Resolve agent disagreements via targeted search | 3 agent traces | reconciled probability |
| Calibrator | Platt scaling (sklearn, no LLM) | raw probability, platt_params.json | calibrated probability |
| Postmortem Agent | Classify process quality, extract lessons | traces, outcomes, scores | memory.md, scores |
| Autoresearcher | Edit prompts based on error patterns | postmortems, changelog, prompts | prompt diffs, changelog |

### Forecasting Pipeline (per question)

```
Question
 ├─→ Agent 1 ─┐
 ├─→ Agent 2 ──┤─→ [decompose → research → base_rate → inside_view → synthesize]
 └─→ Agent 3 ─┘
        │
        ▼
   Supervisor (targeted search on disagreements)
        │
        ▼
   Platt Scaling (statistical correction)
        │
        ▼
   Calibrated Probability + Trace
```

Each stage produces a typed pydantic output feeding the next. Agents diverge naturally through nondeterministic search.

### Improvement Loop (batch-driven, from the start)

There is no separate baseline phase. The first batch is both the baseline and the first input to the improvement loop. Every batch simultaneously produces forecasts and feeds self-improvement.

```
Batch N (20 questions from questions.jsonl)
        │
        ▼
   Pipeline runs with current prompts → traces + raw probabilities
        │
        ▼
   Eval → Brier scores (overall, per domain, per agent, vs community, vs naive LLM)
        │
        ▼
   Platt calibrator refits on all available (raw, outcome) pairs
        │
        ▼
   Postmortem Agent runs on all batch questions → classifications + lessons → memory.md
        │
        ▼
   Autoresearcher reads postmortems + changelog
        → proposes prompt diff
        → A/B tests: runs pipeline on 10 questions (randomly sampled from questions.jsonl,
          biased toward the domain where the error pattern was found) with old vs new
          prompt. During A/B runs, agents see a filtered copy of memory.md that excludes
          any lessons extracted from the 10 test questions. This preserves accumulated
          knowledge from other runs while preventing hindsight contamination. Both
          old and new prompt runs see the same filtered memory.
        → accepts if Brier improves on those 10 questions, reverts otherwise
        → logs decision to changelog.jsonl
        │
        ▼
   Batch N+1 (next 20 questions, with possibly updated prompts)
```

### Operating Modes (build these two only)

- **Backtest** — run pipeline on a batch of resolved questions from questions.jsonl. Produces traces and raw probabilities.
- **Eval** — score a batch, refit Platt, run postmortems, trigger autoresearcher. Runs automatically after each backtest batch.

### File Ownership

```
autoforecast/
├── program.md              # HUMAN ONLY — autoresearcher constitution
├── prompts/                # AUTORESEARCHER (diffs, Brier-gated)
│   ├── decompose.md
│   ├── research.md
│   ├── base_rate.md
│   ├── inside_view.md
│   ├── synthesize.md
│   ├── supervisor.md
│   └── postmortem.md
├── memory.md               # POSTMORTEM AGENT (append + consolidate)
├── orchestrator.py         # entry point: backtest and eval modes
├── agent.py                # one agent's 5-stage pipeline
├── supervisor.py           # reconcile 3 traces via targeted search
├── search.py               # agentic search wrapper (Perplexity API)
├── calibrate.py            # Platt scaling (sklearn logistic regression)
├── eval.py                 # Brier scores, baselines, domain breakdowns
├── postmortem.py           # process classification + lesson extraction
├── autoresearcher.py       # self-improvement: read postmortems, edit prompts
├── types.py                # pydantic models for all stage outputs
├── data/
│   ├── questions.jsonl     # all 586 resolved questions (you provide)
│   └── platt_params.json   # fitted calibration parameters
└── logs/
    ├── changelog.jsonl     # all prompt/memory changes
    ├── traces/             # structured traces per forecast
    └── scores/             # Brier scores per batch
```

---

## Dataset

Input format (each line of questions.jsonl):
```json
{
  "post_id": 42221,
  "question_id": 42005,
  "id": "metaculus_42221",
  "source": "metaculus",
  "url": "https://www.metaculus.com/questions/42221/",
  "title": "Will Donald Trump say \"AI\" in his State of the Union address?",
  "description": "",
  "resolution_criteria": "",
  "created_date": "2026-02-18",
  "close_date": "2026-02-25",
  "resolved_date": "2026-02-25",
  "outcome": 1,
  "community_prediction_final": 0.62,
  "num_forecasters": 65,
  "tags": ["Artificial Intelligence"],
  "domain": "technology_policy"
}
```

**Dataset:**
- **questions.jsonl** (all 586 questions) — backtesting, Platt fitting, prompt iteration
- No fixed validation set. The autoresearcher samples 10 questions (biased toward the relevant domain) for A/B testing each prompt diff.

**community_prediction_final** is a baseline to beat, never a model input. Used in eval alongside naive LLM baseline.

---

## Key Design Decisions

- **Search is the biggest lever.** Agentic adaptive search via Perplexity API. Agents decide what to search iteratively based on what they've found. Non-agentic search barely beats no search (per Bridgewater AIA Forecaster paper).
- **Best-effort date gating.** When backtesting, search should make its best effort to only use data published before the question's close_date. Include date constraints in search queries (e.g., "before February 2026"). Perplexity does not support hard date cutoffs, so this is best-effort — but it matters for backtest integrity. If a source's publication date is after close_date, discard it.
- **Supervisor resolves via research, not aggregation.** LLMs overweight outliers when averaging. Supervisor identifies disagreements, does targeted searches to fact-check, then reconciles.
- **Platt scaling, not prompt-based calibration.** LLMs hedge toward 50% due to RLHF. Prompt-based correction is unreliable. Platt scaling (`sigmoid(a * logit(p) + b)`) fitted via logistic regression corrects this statistically.
- **No market prices as input.** Forecasts must be independent for honest evaluation and eventual trading use.
- **Postmortem on every question, not just misses.** Process quality and outcome quality are classified independently. A 30% prediction that resolves Yes isn't necessarily a bad forecast.
- **Memory is flat.** Single `memory.md` file, always in agent context. No index/fetch pattern until scale demands it.

---

## Evaluation

Three baselines per question:
1. **Pipeline** — calibrated probability (the system's output)
2. **Community** — `community_prediction_final` from dataset
3. **Naive LLM** — ask the model the question directly, no pipeline, no search

Brier score = (forecast - outcome)². Lower is better. Compute overall, per domain, per agent.

Contamination detection: if naive LLM beats community by a suspicious margin, flag the question.

---

## Build Plan

Build in phases. Each phase has a test gate. Proceed to next phase only if tests pass. Log everything to BUILD_LOG.md.

### Phase 1: Types + Data Layer
Build `types.py` (pydantic models for all stage outputs) and dataset loading/splitting utilities. Test: models serialize/deserialize cleanly, dataset splits correctly.

### Phase 2: Search Wrapper
Build `search.py` (Perplexity API wrapper with multi-round agentic search). Test: executes real searches, returns structured results, raises on errors.

### Phase 3: Single Agent Pipeline
Build `agent.py` (5-stage chain) and draft all `prompts/*.md`. Test: run one agent on one real question, inspect trace quality.

### Phase 4: Supervisor
Build `supervisor.py`. Test: run 3 agents on one question, feed to supervisor, verify it references specific disagreements.

### Phase 5: Calibration
Build `calibrate.py`. Test: fit on synthetic data with known bias, verify correction works, verify Brier comparison logic.

### Phase 6: Orchestrator (Backtest)
Build `orchestrator.py` with backtest mode. Test: run on 5 questions end-to-end, all traces logged, all outputs valid.

### Phase 7: Eval + Postmortem
Build `eval.py` and `postmortem.py`. Test: score the 5-question batch, compute baselines, postmortem classifies correctly, lessons append to memory.md.

### Phase 8: Autoresearcher
Build `autoresearcher.py` and draft `program.md`. Test: seed postmortems with a clear error pattern, verify it proposes a relevant diff, validates, and logs decision.

---

## Coding Standards

- **No hardcoded fallbacks.** If an API call fails, raise the error. Do not silently return empty results or default values.
- **Type everything.** All function signatures typed. All stage outputs are pydantic models.
- **Comment the why, not the what.** Every non-obvious decision gets a comment explaining reasoning.
- **Simple solutions.** No premature abstractions. No class hierarchies where a function will do.
- **Raise early.** Validate inputs at function boundaries. Fail fast with clear error messages.
- **Environment variables for config.** API keys, model names, agent count — all from .env, never hardcoded.
- **Log the build.** After each phase, append to BUILD_LOG.md: what was built, design decisions made, tests that pass.