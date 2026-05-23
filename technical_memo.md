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

## Step 6 Reconstruction Snapshot

Step 6 completed reconstruction-only baselines on the same locked split.

| Model | latent_dim | Test Normalized MSE | Test Normalized MAE | Test Original MAE | Relative MSE vs Last Snapshot |
| --- | ---: | ---: | ---: | ---: | ---: |
| pca | 128 | 0.1912 | 0.2396 | 0.1360 | 0.2885 |
| pca | 32 | 0.2924 | 0.3148 | 0.1713 | 0.4413 |
| mlp_ae | 32 | 0.4718 | 0.4478 | 0.2295 | 0.7120 |
| last_snapshot_repeat | 40 | 0.6626 | 0.3563 | 0.1701 | 1.0000 |
| train_mean_window | - | 1.0437 | 0.6121 | 0.3535 | 1.5750 |

Observed Step 6 pattern:

- PCA dominates reconstruction quality across tested latent dimensions.
- Both best PCA and best MLP-AE improve over `last_snapshot_repeat` on normalized MSE.
- For the best model, volume-side reconstruction error is materially larger than price-side reconstruction error.

## Next Comparison

The next step is Step 7 alignment analysis under the same split. I want the comparison to answer:

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
- Balanced accuracy
- Macro-F1
- MCC
- Log loss
- Per-class precision / recall / F1
- Raw and row-normalized confusion matrix

Directional diagnostics:

- Non-neutral recall
- Non-neutral precision
- Directional accuracy on true non-neutral samples
- Up/down macro-F1
- Opposite-direction rate

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
- Step 6 runs reconstruction-only baselines; it does not include prediction-head training.
- Reconstruction-prediction alignment is still not claimed.

The next meaningful milestone is Step 7: align per-sample reconstruction errors with downstream prediction outcomes under the same locked protocol.
