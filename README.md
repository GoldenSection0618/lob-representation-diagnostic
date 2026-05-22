# LOB Representation Diagnostic Study

This is an independent PoW repository.

This is not a fork of LOBench.

This is not a full reproduction of LOBench or SimLOB.

This project studies whether lower LOB reconstruction error reliably transfers to better downstream mid-price trend prediction.

The goal is diagnostic analysis, not SOTA performance.

## Research Question

Does better LOB reconstruction imply better downstream prediction?

## Project Scope

This repository is an independent Proof-of-Work (PoW) repo for building a small-scale diagnostic benchmark.

- LOBench / SimLOB are upstream references.
- Upstream code is not copied as the main project.
- The project uses LOBench-style data and task definitions as external references.
- The repository focuses on controlled experiments, analysis scripts, diagnostic metrics, and technical memo.

## What This Repo Does

- Builds a small chronological subset from LOBench-style LOB data.
- Trains simple controllable representation baselines.
- Compares reconstruction metrics with downstream mid-price trend prediction.
- Analyzes level-wise reconstruction error.
- Analyzes regime-specific failure cases.
- Optionally profiles latency, compression ratio, and efficiency trade-offs.

## What This Repo Does Not Claim

- It does not claim SOTA.
- It does not fully reproduce LOBench or SimLOB.
- It does not evaluate trading profitability.
- It does not redistribute proprietary or private LOB data.
- It does not generalize small-subset results to all market settings.

## Planned Steps

- Step 1: Repository initialization and scope locking.
- Step 2: Inspect LOBench / SimLOB data pipeline.
- Step 3: Build a small chronological subset.
- Step 4: Implement reconstruction baselines.
- Step 5: Train downstream prediction heads.
- Step 6: Analyze reconstruction-prediction alignment.
- Step 7: Analyze level-wise and regime-specific failure cases.
- Step 8: Profile efficiency trade-offs.
- Step 9: Write final technical memo.

## Repository Structure

- `configs/`: configuration files for reconstruction baselines and prediction head.
- `src/data/`: data loading, subset construction, and label-related utilities.
- `src/models/`: representation models and downstream prediction head modules.
- `src/losses/`: reconstruction objectives and weighted variants.
- `src/analysis/`: diagnostic analysis modules.
- `src/utils/`: shared utilities for metrics, seed control, and profiling.
- `scripts/`: stage-by-stage runnable scripts.
- `results/`: experiment outputs and result artifacts.
- `figures/`: generated plots and visual diagnostics.
- Root docs: environment/data notes, execution log, and technical memo.

## Upstream References

- LOBench is used as an external reference.
- SimLOB is used as a methodological reference.
- This repo only stores original scripts, configs, notes, analysis code, and technical memo.

## Current Status

Step 3 chronological subset construction completed. Step 4 reconstruction baselines pending.
