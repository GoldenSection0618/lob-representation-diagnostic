# LOB Representation Diagnostic

This is an independent PoW repository. It is not a fork of LOBench, and it is not a full reproduction of LOBench or SimLOB.

The question I want to answer is narrower and more useful: if a representation reconstructs the LOB with lower error, does that actually improve downstream mid-price trend prediction? If the answer is inconsistent, I want to know where the gap comes from: specific book levels, specific market regimes, or a mismatch between the reconstruction objective and the prediction task.

This project is not chasing SOTA and it is not a trading PnL study. The goal is to run controlled experiments and make the data split, labels, metrics, and failure cases explicit.

## What This Repo Does

- Builds a small chronological subset from external LOBench-style A-share data.
- Locks a 10-level, 100-step, 40-feature input contract.
- Trains simple and controllable reconstruction representation baselines.
- Attaches downstream mid-price trend prediction heads to the same representations.
- Compares reconstruction metrics against prediction metrics.
- Breaks reconstruction error down by order book level.
- Checks failure cases across market regimes such as spread, volatility, and trend strength.
- Adds latency, compression, and efficiency profiling only when it helps explain the trade-off.

## What This Repo Does Not Claim

- It does not claim SOTA performance.
- It does not fully reproduce LOBench or SimLOB.
- It does not evaluate real trading profitability.
- It does not commit or redistribute external, proprietary, or private LOB data.
- It does not generalize a small-subset result to every stock, market, or period.

## Current State

Step 3 is complete. I built a minimal chronological subset from the external processed CSV for `sz000001`.

Key facts:

- Input file: `~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv`
- Output directory: `data/processed/minimal_subset/`
- Window length: `100`
- Feature width: `40`
- Label: `trend5`
- Threshold: `0.0001`
- Split: `70/15/15`
- Final samples: `7802`
- Shape: `X=(7802, 100, 40)`, `y=(7802,)`
- Checks passed: feature contract, label contract, window alignment, chronological split, and output safety.

Next step: Step 4, implement reconstruction baselines without changing the Step 3 data contract.

## Repository Layout

- `configs/`: experiment configs for reconstruction baselines and prediction heads.
- `src/data/`: external data loading, field mapping, label generation, and subset construction.
- `src/models/`: representation models and downstream prediction heads.
- `src/losses/`: reconstruction objectives and weighted variants.
- `src/analysis/`: reconstruction-prediction alignment, level-wise error, and regime failure analysis.
- `src/utils/`: metrics, seed control, profiling, and shared utilities.
- `scripts/`: runnable stage-by-stage entry points.
- `results/`: experiment outputs.
- `figures/`: generated plots and diagnostics.
- Root docs: environment notes, data contract, execution log, and technical memo.

## External References

LOBench and SimLOB are used as external references for data and methodology. I do not copy upstream code into this repository as the main project body. This repo stores only original scripts, configs, analysis code, notes, and memos.
