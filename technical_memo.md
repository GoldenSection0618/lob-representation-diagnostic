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

## Step 7 Alignment Results

Step 7 completed reconstruction-prediction alignment under the same stride-4 boundary-purged chronological protocol. It keeps two evidence chains separate: sample-level difficulty alignment and frozen-latent transfer.

Join contract:

- Step 5 prediction rows: `7056`
- Step 6 reconstruction rows: `79520`
- Expected joined rows: `70560`
- Actual joined rows: `70560`
- Duplicate `sample_id + split + prediction_model + reconstruction_variant` rows: `0`
- Status: `passed`

Frozen-latent transfer leaderboard on the test split:

| Source | Variant | Test Macro-F1 | Balanced Accuracy | MCC | Log Loss |
| --- | --- | ---: | ---: | ---: | ---: |
| frozen latent head | last_snapshot_repeat@40 | 0.4355 | 0.5509 | 0.2579 | 1.0743 |
| raw window baseline | logistic_regression | 0.3972 | 0.4098 | 0.1007 | 4.1624 |
| matched raw-window head | raw_window_logistic_tuned | 0.3904 | 0.4079 | 0.0978 | 2.0500 |
| raw window baseline | mlp | 0.3816 | 0.4513 | 0.1624 | 1.2767 |
| frozen latent head | pca@128 | 0.3624 | 0.4143 | 0.1281 | 1.2174 |
| frozen latent head | pca@32 | 0.3619 | 0.4085 | 0.1200 | 1.1262 |
| raw window baseline | majority | 0.2612 | 0.3333 | 0.0000 | 0.8980 |

The best frozen latent head beats the fixed Step 5 raw-window logistic baseline and the matched raw-window logistic head by test macro-F1 in this controlled run. The matched raw-window head uses the same Step 7 head policy as the latent heads: train-only `StandardScaler`, `class_weight="balanced"`, C grid `{0.01, 0.1, 1.0, 10.0}`, and validation macro-F1 selection with MCC/log-loss tie-breaks. It selected `C=0.1` and reached test macro-F1 `0.3904`. This does not imply a general SOTA claim; the result is limited to `sz000001`, `trend5`, and this stride-4 subset.

Rank alignment:

| Variant | Test Recon MSE | Test Macro-F1 | Recon Rank | Prediction Rank | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| pca@128 | 0.1838 | 0.3624 | 1 | 2 | aligned |
| pca@64 | 0.2803 | 0.3505 | 2 | 6 | better reconstruction than prediction |
| pca@32 | 0.4245 | 0.3619 | 3 | 3 | aligned |
| mlp_ae@64 | 0.4251 | 0.3417 | 4 | 7 | better reconstruction than prediction |
| last_snapshot_repeat@40 | 1.8321 | 0.4355 | 9 | 1 | better prediction than reconstruction |

The reconstruction-best variant (`pca@128`) and frozen-head prediction-best variant (`last_snapshot_repeat@40`) do not match. Across nine frozen-latent variants, Spearman(`test_recon_normalized_mse`, `test_pred_macro_f1`) is `-0.2000`; Spearman(`test_recon_last_step_mse`, `test_pred_macro_f1`) is stronger at `-0.7333`, but the variant count is too small for significance claims. In this run, overall reconstruction MSE is not a reliable standalone proxy for downstream trend prediction.

Sample-level failure diagnostics for the Step 5 `logistic_regression` predictor are weak but informative. For incorrect prediction on the test split, mean AUROC by reconstruction diagnostic is highest for `spread_mae` (`0.5204`), then `top_of_book_mse` (`0.5035`), while `normalized_mse` is below random-direction discrimination (`0.4744`). This points away from using aggregate reconstruction error alone and toward local book-state diagnostics in Step 7 interpretation.

## Step 8 Fairness and Robustness

Step 8 adds controls around the Step 7 transfer claim. It does not change the split, data construction, latent artifacts, or reconstruction encoders.

Fair transfer comparison on the test split:

| Variant | Source | Selection Basis | Test Macro-F1 | MCC | Delta vs Tuned Raw |
| --- | --- | --- | ---: | ---: | ---: |
| raw_window_logistic_untuned | raw window baseline | fixed Step 5 logistic | 0.3972 | 0.1007 | 0.0068 |
| raw_window_logistic_tuned | raw window tuned control | val macro-F1, MCC/log-loss tie-break | 0.3904 | 0.0978 | 0.0000 |
| best_frozen_latent_head | frozen latent head | post hoc best Step 7 test macro-F1 | 0.4355 | 0.2579 | 0.0452 |
| pca@128_frozen_latent_head | frozen latent head | reconstruction-best test normalized MSE | 0.3624 | 0.1281 | -0.0280 |

The tuned raw-window logistic control selected `C=0.1`. The post hoc best frozen-latent head (`last_snapshot_repeat@40`) remains ahead of both raw-window logistic controls on test macro-F1. Paired bootstrap on the same test samples gives delta `0.0452` for best frozen latent versus tuned raw logistic, with 95% CI `[0.0082, 0.0823]` and `fraction_delta_gt_0=0.9930`.

This is still descriptive: `best_frozen_latent_head` is selected post hoc from Step 7 test macro-F1, not pre-registered before looking at test performance.

Rank sensitivity:

| Variant Set | N | Recon MSE vs Macro-F1 Spearman | Best Reconstruction | Best Prediction | Interpretation |
| --- | ---: | ---: | --- | --- | --- |
| all_latent_variants | 9 | -0.2000 | pca@128 | last_snapshot_repeat@40 | rank_mismatch_persists |
| exclude_last_snapshot_repeat | 8 | -0.7143 | pca@128 | pca@128 | rank_mismatch_weakens |
| pca_only | 5 | -0.9000 | pca@128 | pca@128 | rank_mismatch_weakens |
| mlp_ae_only | 3 | 1.0000 | mlp_ae@64 | mlp_ae@16 | rank_mismatch_persists |

`last_snapshot_repeat@40` has zero last-step reconstruction error by construction. Removing it changes the best-prediction variant from `last_snapshot_repeat@40` to `pca@128`, so the strongest rank-mismatch claim does not survive that sensitivity check. The safer conclusion is: the transfer advantage over tuned raw-window logistic is supported in this subset, while the reconstruction-prediction rank mismatch is partially supported and structurally influenced by the last-snapshot baseline.

Overall reconstruction MSE is not a reliable standalone downstream proxy across all tested variants in this one-symbol, one-horizon, stride-4 subset, but the claim is weaker after excluding `last_snapshot_repeat@40`.

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
- Step 7 trains only simple logistic heads on frozen saved latents; it does not retrain reconstruction encoders.
- Step 7 does not establish cross-symbol, cross-horizon, or trading-PnL generality.

The next meaningful milestone is to decide whether Step 7's rank mismatch should motivate a targeted representation ablation, a multi-symbol sensitivity check, or a narrower top-of-book prediction diagnostic.
