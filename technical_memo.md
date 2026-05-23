# Technical Memo

This memo will hold the final experiment story. For now, it records the research framing I want to keep fixed before training results start influencing the narrative.

## Problem

I am not asking only whether a model can reconstruct the LOB more accurately. I am asking whether that better reconstruction transfers to downstream mid-price trend prediction under leakage-aware chronological evaluation.

Those two objectives can diverge. A reconstruction loss averages across the whole book, while trend prediction often depends more on top-of-book behavior, short-term order imbalance, spread changes, and local liquidity shocks. If a model spends capacity reconstructing distant levels or low-value noise, the reconstruction metric can improve while the prediction task gets little benefit.

That is why this PoW is a diagnostic study, not a leaderboard exercise.

## Experimental Setup

The first stage uses one controlled subset:

- Symbol: `sz000001`
- Data source: `~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv`
- Input: `(N, 100, 40)`
- Label: `trend5`
- Split: boundary-purged chronological `70/15/15`
- Samples: `7802`

The main protocol is stricter than a plain chronological split because overlapping boundary windows are removed between train/validation and validation/test.

Random split is not part of the main experiment.

No-purge chronological split is postponed and treated as future ablation work, not current main delivery.

## What I Plan to Compare

Step 5 has completed prediction-only baseline evaluation under the locked boundary-purged chronological protocol. The next experimental step is to implement reconstruction baselines and measure whether reconstruction quality aligns with downstream prediction performance.

The next comparison is between reconstruction baselines with different reconstruction quality, followed by prediction heads trained on their representations.

The comparisons should answer a few concrete questions:

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

Step 5 prediction-only baselines are now available, but reconstruction-alignment conclusions are still out of scope.

Known limits:

- The current subset covers only one segment of `sz000001`.
- Data stays outside the repo, so reproduction requires the external dataset path locally.
- The first label is fixed to `trend5`; other horizons are not yet part of the main experiment.
- There is no multi-symbol, multi-date, or cross-regime robustness result yet.
- Reconstruction baselines have not been run yet.
- Current model-quality conclusions are limited to prediction-only baselines under the locked protocol.
- Reconstruction-prediction alignment is not claimed yet.

## Step 5 Snapshot (Prediction-Only)

Test-split summary on the locked Step 3 subset:

| Model | Accuracy | Balanced Accuracy | Macro-F1 | MCC | Log Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority | 0.4501 | 0.3333 | 0.2069 | 0.0000 | 1.2228 |
| logistic_regression | 0.4122 | 0.3504 | 0.3338 | 0.0250 | 9.2487 |
| mlp | 0.4531 | 0.3535 | 0.2760 | 0.0589 | 1.7594 |

Interpretation bounds:

- These are prediction-only baseline results.
- They do not validate reconstruction quality or reconstruction-to-prediction transfer.
- Any reconstruction alignment claim must wait for later steps.

Next step: implement reconstruction baselines and then test reconstruction-prediction alignment under the same locked protocol.
