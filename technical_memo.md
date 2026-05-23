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
- Sample stride: `4`
- Split: boundary-purged chronological `70/15/15`
- Samples: `7952`

The split is stricter than plain chronological evaluation because overlapping historical windows are removed at train/validation and validation/test boundaries. Random split is not part of the main experiment. No-purge chronological split is left for later ablation work.

## Baseline Snapshot

Step 5 completed prediction-only baselines on the stride-4 locked subset. These results set a floor; they do not say anything yet about reconstruction quality or representation transfer.

| Model | Accuracy | Balanced Accuracy | Macro-F1 | MCC | Log Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority | 0.6441 | 0.3333 | 0.2612 | 0.0000 | 0.8980 |
| logistic_regression | 0.4826 | 0.4098 | 0.3972 | 0.1007 | 4.1624 |
| mlp | 0.4036 | 0.4513 | 0.3816 | 0.1624 | 1.2767 |

The best macro-F1 comes from logistic regression, but the best log loss comes from the majority baseline. That is a practical signal: directional class separation and probability quality are not the same thing.
For selection policy, macro-F1 is the primary metric for this imbalanced directional diagnostic, while log loss is retained as the LOBench-compatible cross-entropy-style probability-quality metric.

## Step 6 Reconstruction Snapshot

Step 6 completed reconstruction-only baselines on the same stride-4 locked split.

| Model | latent_dim | Test Normalized MSE | Test Normalized MAE | Test Original MAE | Relative MSE vs Last Snapshot |
| --- | ---: | ---: | ---: | ---: | ---: |
| pca | 128 | 0.1838 | 0.1871 | 0.1267 | 0.1003 |
| pca | 32 | 0.4245 | 0.3093 | 0.2133 | 0.2317 |
| mlp_ae | 64 | 0.4251 | 0.3736 | 0.2424 | 0.2320 |
| last_snapshot_repeat | 40 | 1.8321 | 0.4410 | 0.2872 | 1.0000 |
| train_mean_window | - | 2.2013 | 0.9195 | 0.5141 | 1.2015 |

Observed Step 6 pattern:

- PCA dominates reconstruction quality across tested latent dimensions.
- Both best PCA and best MLP-AE improve over `last_snapshot_repeat` on normalized MSE.
- Step 6 artifacts use `model_variant` as the canonical variant key, while keeping `model` (family) and `latent_dim` (numeric compression dimension) as separate fields.
- Step 6 exports `lobench_compatible_reconstruction_metrics.csv`; on test, `pca@128` is also best by LOBench-compatible weighted MSE (`0.2917`).
- Imbalance MAE is validity-gated; top1/top5 imbalance validity remains below threshold for the best model, so Step 7 should prefer volume-sum and volume-difference diagnostics.
- `original_mae` / `original_rmse` are measured in Step 3 input feature space after inverse-transforming the Step 6 scaler, not in raw exchange order-flow scale.

## Next Comparison

The next step is Step 7 alignment analysis under the same split. I want the comparison to answer:

- Does lower reconstruction error improve accuracy or macro-F1?
- Are top-of-book errors more predictive of downstream failure than deeper-level errors?
- Do spread widening, higher volatility, or weak-trend periods break the representation first?
- Does any representation justify its latency or compression cost?
- Step 7 now has both interfaces needed for sample-level alignment: `per_sample_reconstruction_errors.csv` (Step 6) and `per_sample_predictions.csv` (Step 5).
- Step 7 should rename Step 5 `model` to `prediction_model` and Step 6 `model_variant` to `reconstruction_variant`; the sample-level join key is `sample_id + split`.

## Metrics

Reconstruction metrics:

- Overall MSE / MAE
- Price-side and volume-side error
- LOBench-compatible weighted MSE / all-loss
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
- `direction_correct_non_neutral` is encoded as 1.0/0.0 for true non-neutral samples and null for neutral samples.
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

The next meaningful milestone is Step 7: align per-sample reconstruction errors with downstream prediction outcomes under the same locked protocol. Step 7 should treat `direction_correct_non_neutral` as numeric 1.0/0.0 for true non-neutral samples and null for neutral samples.
