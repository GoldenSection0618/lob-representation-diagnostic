# Technical Memo

## 1. Problem Definition

This proof of work tests a narrow diagnostic question:

> Does better limit order book reconstruction imply better downstream mid-price trend prediction under a leakage-aware chronological split?

The project separates reconstruction quality from predictive usefulness. A reconstruction model can reduce average book-level error by modeling stable or distant levels, while the prediction target may depend more on top-of-book state, local liquidity, spread behavior, or short-horizon directional changes. The central test is therefore not whether reconstruction loss can be made small, but whether lower reconstruction error aligns with better `trend5` prediction under a controlled protocol.

This memo is not a claim of full LOBench reproduction. It does not claim state-of-the-art prediction or reconstruction performance. It does not evaluate trading profitability, execution quality, or portfolio performance. It also does not claim general market predictability across symbols, horizons, dates, or regimes.

The current evidence is limited to one symbol, one horizon, one stride-4 subset, and one boundary-purged chronological split.

## 2. Experimental Setup

The fixed protocol is:

| Field | Value |
| --- | --- |
| Dataset source | LOBench A-share processed data, kept outside the repository |
| Symbol | `sz000001` |
| Label | `trend5` |
| Window length | `100` |
| Feature dimension | `40` |
| Input tensor | `(N, 100, 40)` |
| Sample stride | `4` |
| Split protocol | Boundary-purged chronological `70/15/15` |
| Total samples | `7952` |
| Train samples | `5600` |
| Validation samples | `1200` |
| Test samples | `1152` |

The split is chronological and boundary-purged. Historical rows overlapping train/validation and validation/test boundaries are removed, so adjacent windows do not leak across split boundaries. Random split and no-purge chronological split are excluded from the main evidence chain.

External data remains local and is not committed. Generated data arrays and latent arrays are also not part of the repository evidence. The committed artifacts are lightweight metrics, summaries, run configs, and figures.

## 3. Metrics and Evidence

The evidence chain uses different metrics for different purposes.

Prediction metrics:

- `macro_f1`, the primary metric for imbalanced directional diagnosis.
- Balanced accuracy and MCC, to reduce dependence on the majority neutral class.
- Log loss, retained as a cross-entropy-style probability-quality metric.
- Directional diagnostics, including non-neutral precision/recall, directional accuracy on true non-neutral samples, up/down macro-F1, and opposite-direction rate.

Reconstruction metrics:

- Normalized MSE and normalized MAE in the Step 6 standardized reconstruction space.
- Original-space MAE/RMSE after inverse-transforming the Step 6 scaler back to the Step 3 input feature space, not raw exchange order-flow scale.
- Price, volume, level-wise, temporal, top-of-book, midprice, spread, and volume-sum/difference diagnostics.
- LOBench-compatible reconstruction metrics, including weighted MSE and all-loss style summaries.

Alignment diagnostics:

- Sample-level joins between Step 5 predictions and Step 6 reconstruction diagnostics.
- Spearman and point-biserial associations between reconstruction diagnostics and prediction outcomes.
- Error-quantile response curves.
- Model-level rank alignment between reconstruction quality and frozen-latent prediction quality.

Robustness controls:

- A tuned raw-window logistic control using the same C grid and validation-selection policy as the frozen latent heads.
- A raw logistic test-oracle reference, clearly marked as post hoc and not a valid selection baseline.
- Paired bootstrap on the same test samples.
- Rank sensitivity after excluding `last_snapshot_repeat@40`.
- Last-snapshot sensitivity, noting that `last_snapshot_repeat@40` has zero last-step reconstruction error by construction.

## 4. Results

### Prediction Baselines

Step 5 trains prediction-only baselines on the locked stride-4 subset. These baselines set the raw-window prediction reference and do not use reconstruction features.

Test split:

| Model | Accuracy | Balanced Accuracy | Macro-F1 | MCC | Log Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| majority | 0.6441 | 0.3333 | 0.2612 | 0.0000 | 0.8980 |
| logistic_regression | 0.4826 | 0.4098 | 0.3972 | 0.1007 | 4.1624 |
| mlp | 0.4036 | 0.4513 | 0.3816 | 0.1624 | 1.2767 |

The fixed raw-window logistic regression baseline has the best Step 5 test macro-F1 (`0.3972`). The majority baseline has the best log loss (`0.8980`), which shows that directional separation and probability calibration are different properties in this subset. Macro-F1 remains the primary diagnostic metric for directional classification, while log loss is reported for probability-quality context.

### Reconstruction Baselines

Step 6 trains reconstruction-only baselines on the same split. It does not train prediction heads.

Test split:

| Model | Latent Dim | Test Normalized MSE | Test Normalized MAE | Test Original MAE | Relative MSE vs Last Snapshot |
| --- | ---: | ---: | ---: | ---: | ---: |
| pca | 128 | 0.1838 | 0.1871 | 0.1267 | 0.1003 |
| pca | 32 | 0.4245 | 0.3093 | 0.2133 | 0.2317 |
| mlp_ae | 64 | 0.4251 | 0.3736 | 0.2424 | 0.2320 |
| last_snapshot_repeat | 40 | 1.8321 | 0.4410 | 0.2872 | 1.0000 |
| train_mean_window | - | 2.2013 | 0.9195 | 0.5141 | 1.2015 |

`pca@128` is the best reconstruction variant among the tested baselines by test normalized MSE (`0.1838`). It is also best by the LOBench-compatible weighted MSE on test (`0.2917`). The reconstruction results support the expected control behavior: PCA with higher latent dimension reconstructs the standardized windows much better than the last-snapshot and train-mean baselines.

The imbalance diagnostics are validity-gated. Because top1/top5 imbalance validity remains below threshold for the best model, downstream alignment emphasizes volume-sum and volume-difference diagnostics rather than imbalance MAE as a primary field.

### Sample-Level Alignment Contract

Step 7 joins Step 5 per-sample predictions with Step 6 per-sample reconstruction diagnostics on `sample_id`, `split`, and `y_true`, after renaming model columns to avoid semantic collision.

Join contract:

| Field | Value |
| --- | ---: |
| Step 5 prediction rows | 7056 |
| Step 6 reconstruction rows | 79520 |
| Expected joined rows | 70560 |
| Actual joined rows | 70560 |
| Duplicate joined keys | 0 |
| Status | passed |

This supports the validity of the sample-level alignment panel for validation/test diagnostics.

### Frozen-Latent Transfer

Step 7 trains logistic heads on frozen reconstruction latents. The reconstruction encoders are not retrained. The head uses a train-only `StandardScaler`, `class_weight="balanced"`, C grid `{0.01, 0.1, 1.0, 10.0}`, and validation macro-F1 selection with MCC and log-loss tie-breaks.

Compact transfer leaderboard on the test split:

| Source | Variant | Selection Basis | Test Macro-F1 | Balanced Accuracy | MCC | Log Loss |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| frozen latent head | last_snapshot_repeat@40 | post hoc best Step 7 test macro-F1 | 0.4355 | 0.5509 | 0.2579 | 1.0743 |
| raw window baseline | logistic_regression | fixed Step 5 logistic | 0.3972 | 0.4098 | 0.1007 | 4.1624 |
| matched raw-window head | raw_window_logistic_tuned | validation macro-F1, MCC/log-loss tie-break | 0.3904 | 0.4079 | 0.0978 | 2.0500 |
| raw window oracle reference | raw_window_logistic_test_oracle | post hoc best raw-grid test macro-F1 | 0.4101 | 0.4532 | 0.1473 | 1.3221 |
| raw window baseline | mlp | fixed Step 5 MLP | 0.3816 | 0.4513 | 0.1624 | 1.2767 |
| frozen latent head | pca@128 | reconstruction-best test normalized MSE | 0.3624 | 0.4143 | 0.1281 | 1.2174 |
| raw window baseline | majority | fixed majority baseline | 0.2612 | 0.3333 | 0.0000 | 0.8980 |

The best frozen latent head, `last_snapshot_repeat@40`, has higher test macro-F1 (`0.4355`) than the fixed raw-window logistic baseline (`0.3972`) and the validation-selected tuned raw-window logistic control (`0.3904`). The raw logistic C grid contains a test-oracle point at `C=0.01` with test macro-F1 `0.4101`; this is reported for transparency but is not a valid model-selection baseline because it is selected after observing test performance.

This transfer result is supported for the current subset, but it is descriptive rather than fully confirmatory because `best_frozen_latent_head` is selected post hoc from Step 7 test macro-F1.

### Rank Alignment

Model-level rank alignment compares reconstruction quality against frozen-latent prediction quality on the test split.

| Variant | Test Recon MSE | Test Macro-F1 | Recon Rank | Prediction Rank | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| pca@128 | 0.1838 | 0.3624 | 1 | 2 | aligned |
| pca@64 | 0.2803 | 0.3505 | 2 | 6 | better reconstruction than prediction |
| pca@32 | 0.4245 | 0.3619 | 3 | 3 | aligned |
| mlp_ae@64 | 0.4251 | 0.3417 | 4 | 7 | better reconstruction than prediction |
| last_snapshot_repeat@40 | 1.8321 | 0.4355 | 9 | 1 | better prediction than reconstruction |

Across the nine frozen-latent variants, the best reconstruction variant is `pca@128`, while the best frozen-head prediction variant is `last_snapshot_repeat@40`. Spearman(`test_recon_normalized_mse`, `test_pred_macro_f1`) is `-0.2000`, while Spearman(`test_recon_last_step_mse`, `test_pred_macro_f1`) is `-0.7333`. These are descriptive correlations only, because the variant count is small.

### Fairness and Robustness

Step 8 adds controls around the Step 7 transfer and rank conclusions without changing data construction, split logic, reconstruction encoders, or latent artifacts.

Fair transfer comparison on the test split:

| Variant | Source | Selection Basis | Test Macro-F1 | MCC | Delta vs Tuned Raw |
| --- | --- | --- | ---: | ---: | ---: |
| raw_window_logistic_untuned | raw window baseline | fixed Step 5 logistic | 0.3972 | 0.1007 | 0.0068 |
| raw_window_logistic_tuned | raw window tuned control | validation macro-F1, MCC/log-loss tie-break | 0.3904 | 0.0978 | 0.0000 |
| raw_window_logistic_test_oracle | raw window oracle reference | post hoc best raw-grid test macro-F1 | 0.4101 | 0.1473 | 0.0198 |
| best_frozen_latent_head | frozen latent head | post hoc best Step 7 test macro-F1 | 0.4355 | 0.2579 | 0.0452 |
| pca@128_frozen_latent_head | frozen latent head | reconstruction-best test normalized MSE | 0.3624 | 0.1281 | -0.0280 |

The validation-selected tuned raw-window logistic control selects `C=0.1` and reaches test macro-F1 `0.3904`. Paired bootstrap on the same test samples gives a macro-F1 delta of `0.0452` for best frozen latent versus tuned raw logistic, with 95% CI `[0.0082, 0.0823]` and `fraction_delta_gt_0=0.9930`.

This supports the narrow claim that the post hoc best frozen latent head is stronger than the tuned raw-window logistic control on test macro-F1 in this subset. It does not make the comparison fully pre-registered or general.

Rank sensitivity:

| Variant Set | N | Recon MSE vs Macro-F1 Spearman | Best Reconstruction | Best Prediction | Interpretation |
| --- | ---: | ---: | --- | --- | --- |
| all_latent_variants | 9 | -0.2000 | pca@128 | last_snapshot_repeat@40 | rank_mismatch_persists |
| exclude_last_snapshot_repeat | 8 | -0.7143 | pca@128 | pca@128 | rank_mismatch_weakens |
| pca_only | 5 | -0.9000 | pca@128 | pca@128 | rank_mismatch_weakens |
| mlp_ae_only | 3 | 1.0000 | mlp_ae@64 | mlp_ae@16 | rank_mismatch_persists |

The all-variant rank mismatch is influenced by `last_snapshot_repeat@40`. After excluding that special baseline, `pca@128` becomes both reconstruction-best and prediction-best. The rank-mismatch claim is therefore only partially supported.

## 5. Failure and Mismatch Analysis

The final interpretation is conservative.

First, aggregate reconstruction quality and downstream prediction quality are not interchangeable in the full variant set. `pca@128` is best by test reconstruction MSE, while `last_snapshot_repeat@40` is best by frozen-head test macro-F1. This is the clearest mismatch observed in Step 7.

Second, that mismatch has an important structural caveat. `last_snapshot_repeat@40` has zero last-step reconstruction error by construction, because it repeats the final observed snapshot across the window. It is a weak full-window reconstructor but preserves a specific local state that appears useful for the `trend5` logistic head in this subset.

Third, after excluding `last_snapshot_repeat@40`, the strongest mismatch weakens. In the `exclude_last_snapshot_repeat` subset, `pca@128` becomes both reconstruction-best and prediction-best. The same is true for the PCA-only subset. This prevents a strong general claim that reconstruction rank and prediction rank diverge among ordinary compressed reconstruction variants.

Fourth, sample-level diagnostics do not support aggregate normalized MSE as a strong standalone failure signal. For the Step 5 `logistic_regression` predictor on the test split, the mean AUROC for incorrect prediction is highest for `spread_mae` (`0.5204`), followed by `top_of_book_mse` (`0.5035`), while `normalized_mse` is below random-direction discrimination (`0.4744`). These values are weak, but they point toward local book-state diagnostics rather than aggregate reconstruction error alone.

The resulting claim is:

- Supported: the Step 7 sample-level join is valid.
- Supported within this subset: the post hoc best frozen latent head beats the fixed and tuned raw-window logistic controls on test macro-F1.
- Partially supported: reconstruction-best and prediction-best variants differ across all latent variants.
- Partially supported: overall reconstruction MSE is not a reliable standalone downstream proxy in this controlled run.
- Scope-limited: all conclusions are restricted to `sz000001`, `trend5`, the stride-4 subset, and the boundary-purged chronological split.

## 6. Limitations

The main limitations are:

- One symbol: only `sz000001` is evaluated.
- One horizon: only `trend5` is evaluated.
- One subset: the current evidence uses one stride-4 sample construction.
- No multi-symbol robustness.
- No multi-horizon robustness.
- No trading PnL, execution, slippage, cost, or portfolio evaluation.
- No random-split or no-purge ablation in the main evidence chain.
- No cross-regime or multi-date stress test.
- Reconstruction encoders are not retrained in Step 7 or Step 8.
- The best frozen latent head is selected post hoc from Step 7 test macro-F1.
- The paired bootstrap comparison is descriptive, not fully pre-registered confirmatory evidence.
- The raw logistic test-oracle point is a transparency reference, not a selection-valid baseline.

These limits are intentional for the current PoW scope. The goal is a controlled diagnostic, not a broad market-modeling benchmark.

## 7. Future Work

Future extensions should tighten the claim before broadening it:

- Pre-register the latent-head selection rule before evaluating test performance.
- Repeat the protocol across additional A-share symbols.
- Repeat the protocol across additional prediction horizons.
- Add sensitivity checks across time segments or market regimes.
- Test top-of-book-focused reconstruction objectives.
- Compare aggregate reconstruction loss against local book-state diagnostics such as spread, midprice, top-of-book error, and volume-sum/difference error.
- Evaluate whether last-snapshot-like representations remain predictive after stricter controls or alternative labels.
- Add trading-oriented evaluation only after the diagnostic relationship between reconstruction and prediction is better established.
