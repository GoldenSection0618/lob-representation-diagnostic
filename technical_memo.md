# Technical Memo

This memo will hold the final experiment story. For now, it records the research framing I want to keep fixed before training results start influencing the narrative.

## Problem

I am not asking only whether a model can reconstruct the LOB more accurately. I am asking whether that better reconstruction makes the downstream mid-price trend predictor better.

Those two objectives can diverge. A reconstruction loss averages across the whole book, while trend prediction often depends more on top-of-book behavior, short-term order imbalance, spread changes, and local liquidity shocks. If a model spends capacity reconstructing distant levels or low-value noise, the reconstruction metric can improve while the prediction task gets little benefit.

That is why this PoW is a diagnostic study, not a leaderboard exercise.

## Experimental Setup

The first stage uses one controlled subset:

- Symbol: `sz000001`
- Data source: `~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv`
- Input: `(N, 100, 40)`
- Label: `trend5`
- Split: chronological `70/15/15`
- Samples: `7802`

I do not use random split in the main experiment. This makes the metrics more conservative, but it better matches the time-series setting that the model would face in practice.

## What I Plan to Compare

The baselines should answer a few concrete questions:

- When reconstruction error decreases, do prediction accuracy and macro-F1 improve as well?
- Which levels drive the error? Top-of-book mistakes usually matter more than distant-level mistakes for trend prediction.
- Which regimes break the representation? Spread widening, higher volatility, and weak trend periods are the first places to inspect.
- Is the representation worth its cost? If it only improves reconstruction while making prediction slower or less stable, it is not useful in practice.

## Metrics

Reconstruction side:

- Overall MSE / MAE
- Price-side error and volume-side error
- Level-wise error
- Top-of-book error

Prediction side:

- Accuracy
- Macro-F1
- Per-class precision / recall
- Confusion matrix

Diagnostic side:

- Correlation between reconstruction metrics and prediction metrics.
- Relationship between level-wise error and prediction failure.
- Performance sliced by spread, volatility, and mid-price movement strength.

## Current Limits

Only the Step 3 data path is complete right now. There are no training results yet, so I should not draw model-quality conclusions.

Known limits:

- The current subset covers only one segment of `sz000001`.
- Data stays outside the repo, so reproduction requires the external dataset path locally.
- The first label is fixed to `trend5`; other horizons are not yet part of the main experiment.
- There is no multi-symbol, multi-date, or cross-regime robustness result yet.

Next step: build Step 4 reconstruction baselines. After those runs, this memo should add result tables, failure cases, and conclusions.
