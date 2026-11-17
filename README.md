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

## Run it

```bash
python holdout/app.py   # http://localhost:8811
```

Data: Binance public 4h klines (BTC/ETH/SOL), cached on disk so every backtest in a
session sees identical bars. ~7,000 bars each; split 60/20/20.
