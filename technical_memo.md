# Technical Memo

This memo is the working narrative for the experiment. I keep the framing explicit so later model results do not rewrite the question after the fact.

## Core Question

The project is not asking whether a model can make the LOB look better under reconstruction loss. It asks whether that better reconstruction transfers into mid-price trend prediction under a leakage-aware chronological evaluation.

The distinction matters. Reconstruction loss averages across the book. Trend prediction often lives closer to top-of-book behavior, short-term imbalance, spread changes, and local liquidity shocks. A model can spend capacity on distant levels or stable noise and still look good on reconstruction while doing little for the decision target.

That is the diagnostic angle of this PoW.

## Current Setup

The first controlled subset is:

- Symbol: `sz000001`
- Source: `~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv`
- Input: `(N, 100, 40)`
- Label: `trend5`
- Split: boundary-purged chronological `70/15/15`
- Samples: `7802`

The split is stricter than plain chronological evaluation because overlapping historical windows are removed at train/validation and validation/test boundaries. Random split is not part of the main experiment. No-purge chronological split is left for later ablation work.

## Baseline Snapshot

Step 5 has completed prediction-only baselines on the locked subset. These results set a floor; they do not say anything yet about reconstruction quality or representation transfer.

| Model | Accuracy | Balanced Accuracy | Macro-F1 | MCC | Log Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority | 0.4501 | 0.3333 | 0.2069 | 0.0000 | 1.2228 |
| logistic_regression | 0.4122 | 0.3504 | 0.3338 | 0.0250 | 9.2487 |
| mlp | 0.4531 | 0.3535 | 0.2760 | 0.0589 | 1.7594 |

The best macro-F1 comes from logistic regression, but the log loss is weak. That is a practical signal: the model separates classes slightly better than the majority baseline, but its probabilities are not reliable.

## Next Comparison

The next step is to train reconstruction baselines, extract representations, and attach prediction heads under the same split. I want the comparison to answer:

- Does lower reconstruction error improve accuracy or macro-F1?
- Are top-of-book errors more predictive of downstream failure than deeper-level errors?
- Do spread widening, higher volatility, or weak-trend periods break the representation first?
- Does any representation justify its latency or compression cost?

## Metrics

Reconstruction metrics:

- Overall MSE / MAE
- Price-side and volume-side error
- Level-wise error
- Top-of-book error

Prediction metrics:

- Accuracy
- Macro-F1
- Per-class precision / recall
- Confusion matrix

Diagnostic cuts:

- Correlation between reconstruction metrics and prediction metrics
- Level-wise error versus prediction failure
- Performance by spread, volatility, and mid-price movement strength

## Limits

The current evidence is intentionally narrow:

- Only one segment of `sz000001` is covered.
- The external dataset is required locally and is not committed to the repo.
- The active label is `trend5`; other horizons remain future sensitivity checks.
- There is no multi-symbol, multi-date, or cross-regime robustness result yet.
- Reconstruction baselines have not been run.
- Reconstruction-prediction alignment is not claimed.

The next meaningful milestone is Step 6-level analysis after reconstruction baselines exist: compare reconstruction quality, prediction quality, and failure modes under the same locked protocol.
