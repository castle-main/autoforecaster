# AutoForecast

Autonomous forecasting system that produces calibrated probability estimates for geopolitical and economic events. Encodes Tetlock's superforecasting methodology as a multi-agent pipeline with agentic web search and self-improving prompts.

## How It Works

A question enters the pipeline and is independently analyzed by three agents (Claude, GPT, Gemini). Each agent runs a five-stage reasoning chain grounded in real-time web search. A supervisor identifies where the agents disagree, runs targeted searches to fact-check, and reconciles them into a single probability. Platt scaling then corrects the systematic biases that LLMs introduce (hedging toward 50%). The result is a calibrated probability with a full reasoning trace.

After each batch, the system evaluates its own performance, runs postmortems, and uses an autoresearcher to propose and A/B test prompt improvements — creating a self-improving loop.

## Architecture

### Forecasting Pipeline

```
Question
 ├──→ Agent 0 (Claude Opus)  ─┐
 ├──→ Agent 1 (GPT 5.4)      ─┤──→ [Decompose → Research → Base Rate → Inside View → Synthesize]
 └──→ Agent 2 (Gemini 3.1)   ─┘
               │
               ▼
     Supervisor (targeted search on disagreements)
               │
               ▼
     Platt Scaling (statistical calibration)
               │
               ▼
     Calibrated Probability + Trace
```

### Agent Stages

Each agent runs the same five-stage chain with its own LLM provider, producing natural diversity:

1. **Decompose** — Break the question into sub-questions and generate initial search queries.
2. **Research** — Agentic multi-round web search via Perplexity. A fast model (Haiku) decides whether to search more based on information gaps. Findings are synthesized into structured evidence.
3. **Base Rate** — Identify reference classes and estimate a prior probability from historical frequencies.
4. **Inside View** — Weigh case-specific factors (evidence for/against) to produce an inside-view estimate.
5. **Synthesize** — Combine base rate and inside view into a final probability with confidence intervals.

### Agentic Search

Search is powered by the Perplexity API with iterative follow-up:

- Initial queries come from the decompose stage
- After each round, a fast model evaluates whether information gaps remain and generates follow-up queries
- Up to 3 rounds per agent (reduced to 1 when shared cluster research is available)
- **Date gating**: Search queries include date constraints for backtesting integrity. Citations published after the question's close date are filtered out.
- **Contamination scanning**: Responses are checked for outcome-leaking language and redacted if detected.

### Supervisor

The supervisor does not average agent probabilities — LLMs overweight outliers when aggregating. Instead it:

1. Identifies specific disagreements between the three agents
2. Runs targeted searches to fact-check the disputed claims
3. Reconciles into a single probability with reasoning

For **mutually exclusive question clusters** (e.g., "Who wins the election?"), a cluster supervisor allocates probabilities across all options with a sum ≤ 1.0 constraint, ensuring coherent forecasts.

### Platt Calibration

LLMs systematically hedge toward 50% due to RLHF training. Platt scaling corrects this via logistic regression: `sigmoid(a * logit(p) + b)`, fitted on all historical (raw probability, outcome) pairs. The calibrator refits after every eval batch.

## Training & Self-Improvement

### The Improvement Loop

```
Batch N (questions from questions.jsonl)
    │
    ▼
Pipeline → traces + raw probabilities
    │
    ▼
Eval → Brier scores (overall, per domain, per agent, vs community, vs naive LLM)
    │
    ▼
Platt calibrator refits on all available (raw, outcome) pairs
    │
    ▼
Postmortem → process classifications + lessons → memory.md
    │
    ▼
Autoresearcher → proposes prompt diff → A/B test → accept/reject → changelog
    │
    ▼
Batch N+1 (with possibly updated prompts)
```

There is no separate baseline phase. The first batch is both the baseline and the first input to the improvement loop.

### Evaluation

Three baselines are compared per batch:

| Baseline | Description |
|----------|-------------|
| **Pipeline** | The system's calibrated output |
| **Community** | `community_prediction_final` from the dataset (never used as model input) |
| **Naive LLM** | Ask the model directly — no pipeline, no search |

**Brier score** = (forecast - outcome)². Lower is better. Computed overall, per domain, and per agent. If the naive LLM beats community by a suspicious margin, the question is flagged for contamination.

### Postmortem

Runs on every question — not just misses. Process quality and outcome quality are classified independently into a 2x2 matrix:

|  | Good Outcome | Bad Outcome |
|--|-------------|------------|
| **Good Process** | Ideal | Unlucky — process was sound |
| **Bad Process** | Lucky — don't repeat | Fix this |

Lessons are extracted and appended to `memory.md`.

### Autoresearcher

The autoresearcher reads postmortems from the batch, identifies recurring error patterns (minimum 3 occurrences), and proposes targeted prompt edits. Changes are validated through A/B testing:

1. Sample N questions (biased toward the error domain)
2. Run pipeline with old prompt vs. new prompt
3. Accept only if Brier score improves
4. Log decision to `changelog.jsonl`

During A/B runs, memory is filtered to exclude lessons from the test questions, preventing hindsight contamination.

### Memory

A single flat `memory.md` file injected into every agent stage. The postmortem agent appends lessons after each batch, and periodic consolidation removes duplicates. Memory grows organically as the system learns from its mistakes.

## Running the System

### Setup

Required environment variables:

```bash
ANTHROPIC_API_KEY=...    # Claude (Agent 0 + supervisor + search orchestration)
OPENAI_API_KEY=...       # GPT (Agent 1)
GEMINI_API_KEY=...       # Gemini (Agent 2)
PERPLEXITY_API_KEY=...   # Agentic web search
FORECAST_CONCURRENCY=5   # Optional: max parallel questions (default 5)
```

### CLI Modes

```bash
# Continuous training loop — backtest → eval → improve, repeat for N hours
python -m autoforecast --run 4

# Backtest specific batches
python -m autoforecast 0 3          # batches 0, 1, 2

# Evaluate a completed batch (score + postmortem + autoresearcher)
python -m autoforecast --eval 0

# Forecast unresolved testing questions (live mode)
python -m autoforecast --test

# Single interactive forecast
python -m autoforecast --ask "Will the Fed cut rates in June 2026?"

# Generate performance plots
python -m autoforecast --plot
```

### Flags

| Flag | Description | Default |
|------|-------------|---------|
| `--batch-size N` | Questions per batch | 12 |
| `--concurrency N` | Max parallel questions | 5 |
| `--ab-size N` | Questions sampled for A/B testing | 10 |
| `--max N` | Max questions for `--test` mode | all |
| `--start N` | Starting batch for `--run` mode | 0 |

## Codebase Structure

```
autoforecast/
├── orchestrator.py         # CLI entry point, backtest/eval/run/test/ask modes
├── agent.py                # Single agent's 5-stage pipeline, multi-provider dispatch
├── supervisor.py           # Reconcile 3 agent traces via targeted search
├── search.py               # Perplexity API wrapper with date gating + contamination scan
├── calibrate.py            # Platt scaling (sklearn logistic regression)
├── eval.py                 # Brier scores, baselines, domain breakdowns
├── postmortem.py           # Process classification + lesson extraction
├── autoresearcher.py       # Self-improvement: read postmortems, propose + A/B test prompt diffs
├── cluster.py              # Group mutually exclusive questions into clusters
├── cluster_research.py     # Shared research phase for clustered questions
├── types.py                # Pydantic models for all stage outputs
├── events.py               # Pipeline event system for UI updates + cost tracking
├── ui.py                   # Rich terminal UI with live progress
├── plot.py                 # Brier score charts (backtest)
├── test_plot.py            # Interactive HTML plot (test mode)
├── utils.py                # Shared helpers (load prompts, memory, JSONL)
│
prompts/                    # LLM prompts (edited by autoresearcher, A/B gated)
├── decompose.md
├── research.md
├── base_rate.md
├── inside_view.md
├── synthesize.md
├── supervisor.md
├── cluster_supervisor.md
├── cluster_research.md
├── postmortem.md
│
data/
├── questions.jsonl         # 586 resolved questions for backtesting
├── testing_questions.jsonl # Unresolved questions for live forecasting
├── platt_params.json       # Fitted calibration parameters
│
logs/
├── traces/                 # Structured traces per forecast
├── test_traces/            # Traces from --test mode (with resume support)
├── scores/                 # Brier scores per batch
├── changelog.jsonl         # All prompt/memory changes
├── run_summary.json        # Latest continuous run summary
│
memory.md                   # Accumulated forecasting lessons (flat file)
program.md                  # Autoresearcher constitution (human-only)
```

## Dataset

Input format (`data/questions.jsonl`):

```json
{
  "post_id": 42221,
  "question_id": 42005,
  "id": "metaculus_42221",
  "source": "metaculus",
  "url": "https://www.metaculus.com/questions/42221/",
  "title": "Will Donald Trump say \"AI\" in his State of the Union address?",
  "close_date": "2026-02-25",
  "outcome": 1,
  "community_prediction_final": 0.62,
  "num_forecasters": 65,
  "tags": ["Artificial Intelligence"],
  "domain": "technology_policy"
}
```

586 resolved questions across 8 domains: conflict & security, climate & energy, diplomacy, elections, other policy, regulation & policy, technology policy, trade & economics.
