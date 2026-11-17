# Holdout — strategy research that cannot see the answer

An LLM agent does quantitative strategy research under the constraints a real quant desk
would impose on a junior researcher — and under a few more, because it is a language model
with three years of crypto history memorised.

## Why "blind"?

An LLM asked to "find a BTC strategy" will leak what it remembers about BTC's chart.
Holdout removes that channel:

- the asset is an **alias** ("Asset A"), with no dates, no prices — only distributional
  facts about the training segment (vol, autocorrelation, share of bars above the 200-bar
  average);
- strategies are **data, not code**: a closed JSON DSL (8 indicators, ≤3 entry / ≤2 exit
  conditions, ATR stop, max hold), validated and clamped by the harness. The model cannot
  execute anything;
- the last 20% of bars is **sealed**. The agent may promote a candidate to the holdout at
  most 3 times per run and is told only PASS or FAIL — never the numbers.

## The loop

`Recall → Hypothesise → Backtest → Critique → Reflect → Promote` (per generation).

The backtester executes signals on the **next bar's open** with fee+slippage costs. The
critic applies seven gates: trade count, validation Sharpe, train→validation decay,
parameter stability under ±15% jitter (8 variants), cross-asset breadth, drawdown, and a
**deflated Sharpe ratio** (Bailey & López de Prado) whose bar rises with every strategy
tried — so the agent pays for its own multiple testing.

## Self-improvement

After each generation the agent writes ≤2 falsifiable lessons to SQLite-backed memory.
Retrieval ranks lessons by lexical relevance × a Laplace-smoothed utility prior: the share
of past recalls that preceded an improvement in the run's frontier. Lessons that keep
helping rise; lessons that mislead sink. The same memory persists across runs and assets.

## Evaluation

Every configuration has a control: **random search over the same DSL, same gates, same
trial budget, 8 seeds** (median / IQR / best-seed frontier drawn behind the agent's).
In the first live run the agent's frontier reached 0.18 by generation 4 while the random
median stayed at 0.00 (best seed 0.03) — and the gates promoted nothing to the sealed
holdout, which is the honest outcome for 15 trials on one asset.

## Setup

`agentkit` lives in its own private repository and is installed as a dependency.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # needs git access to cacheycat/agentkit
python app.py                            # http://localhost:8811
```

Working on both repos at once? Skip the pinned install and link a sibling checkout:

```bash
pip install fastapi uvicorn numpy certifi pydantic
ln -s ../agentkit/agentkit agentkit      # ignored by git
```

The LLM layer picks a provider automatically: `ANTHROPIC_API_KEY`, else a local `claude`
CLI (headless, tools disabled), else the deterministic baseline.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q && ruff check .
```

The suite includes a property check that every indicator value at bar `i` is unchanged when
later bars are removed, and that fills happen strictly after a signal bar.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/meta` | Assets, split dates, provider mode, memory stats |
| POST | `/api/research` | Start a research run (`asset`, `generations`, `per_gen`, `baseline`) |
| GET | `/api/equity` | Equity curve for a strategy on a segment |
| GET | `/api/memory` | Stored lessons with their learned utility |
| POST / GET | `/api/baseline` | Run / read the 8-seed random-search control |
| GET | `/api/runs`, `/api/runs/{id}`, `/api/runs/{id}/events` | Run list, detail, replayable SSE trace |

Data: Binance public 4h klines (BTC/ETH/SOL), cached on disk under `data/` (git-ignored) so
every backtest in a session sees identical bars. ~7,000 bars each; split 60/20/20.
