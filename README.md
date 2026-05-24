# LOB Representation Diagnostic

This is an independent PoW repo, not a LOBench fork and not a SimLOB/LOBench reproduction.

The question is simple: when a model reconstructs the limit order book better, does that improvement survive into downstream mid-price trend prediction once the evaluation split is leakage-aware? If it does not, I want to see where the mismatch comes from: top-of-book versus deeper levels, quiet versus stressed regimes, or the reconstruction objective itself.

I am not optimizing for a leaderboard score or a trading PnL claim. The value of this repo is a clean diagnostic trail: data contract, split policy, baseline results, and failure analysis that can be audited later.

## Current Position

The main evaluation path is locked to LOBench-style `sample_stride=4` window sampling plus a boundary-purged chronological split. Train comes first, validation follows, test comes last, and overlapping sliding-window history is purged at train/validation and validation/test boundaries. In code, that policy is enforced through `build_sliding_windows()`, `chronological_split()`, and `_enforce_non_overlap_boundary()`.

Random split is not part of the main experiment. A no-purge chronological split is also not part of Step 4/5/6/7/8; if I add either later, it will be labeled as an auxiliary diagnostic rather than the primary result.

Step 3, Step 5, Step 6, Step 7, and Step 8 have been run on the stride-4 main protocol. Step 8 adds fairness and robustness controls around the Step 7 transfer and rank-alignment conclusions for this one-symbol, one-horizon subset.

Current data run:

- Source: `~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv`
- Output: `data/processed/minimal_subset/`
- Window: `100`
- Feature width: `40`
- Label: `trend5`
- Sample stride: `4`
- Threshold: `0.0001`
- Split: `70/15/15`, boundary-purged chronological
- Samples: `7952`
- Shapes: `X=(7952, 100, 40)`, `y=(7952,)`
- Split sizes: `train=5600`, `val=1200`, `test=1152`
- Boundary samples dropped: `48`
- Data checks passed: feature contract, label contract, window alignment, chronological split, output safety

Step 5 prediction-only test results:

- `majority`: `macro_f1=0.2612`, `balanced_accuracy=0.3333`, `mcc=0.0000`, `log_loss=0.8980`
- `logistic_regression`: `macro_f1=0.3972`, `balanced_accuracy=0.4098`, `mcc=0.1007`, `log_loss=4.1624`
- `mlp`: `macro_f1=0.3816`, `balanced_accuracy=0.4513`, `mcc=0.1624`, `log_loss=1.2767`

`logistic_regression` is the best test model by macro-F1, while `majority` is best by log loss. I treat that as a useful warning: directional class separation and probability quality are not the same thing.

Artifacts from Step 5 live under:

- `results/step5_prediction_baselines/`
- `figures/step5_prediction_baselines/`

Step 6 reconstruction-only test snapshot:

- Best test normalized-MSE model: `pca@128` (`normalized_mse=0.1838`, `normalized_mae=0.1871`, `original_mae=0.1267`)
- Strong compression-constrained point (`latent_dim<=40`): `pca@32` (`normalized_mse=0.4245`)
- Both best `pca` and best `mlp_ae` variants beat `last_snapshot_repeat` on test normalized-MSE
- LOBench-compatible reconstruction metrics are exported in `lobench_compatible_reconstruction_metrics.csv`; on test, `pca@128` is also best by weighted MSE.
- Step 6 saved local latent arrays for Step 7 under `artifacts/step6_reconstruction_baselines/latents/` (ignored, not committed).

Artifacts from Step 6 live under:

- `results/step6_reconstruction_baselines/`
- `figures/step6_reconstruction_baselines/`
- local latent arrays: `artifacts/step6_reconstruction_baselines/latents/` (ignored, not committed)

Step 7 reconstruction-prediction alignment snapshot:

- Join contract: `passed`; expected and actual joined rows are both `70560`.
- Best Step 5 raw-window baseline by test macro-F1: `logistic_regression` (`0.3972`).
- Matched raw-window logistic head with the same Step 7 C-grid policy: `raw_window_logistic_tuned` (`macro_f1=0.3904`, selected `C=0.1`).
- Best frozen-latent logistic head by test macro-F1: `last_snapshot_repeat@40` (`0.4355`, `balanced_accuracy=0.5509`, `mcc=0.2579`).
- Best reconstruction variant by test normalized MSE: `pca@128` (`0.1838`).
- The reconstruction-best and frozen-head prediction-best variants are not the same.
- The best frozen-latent head beats both the fixed Step 5 logistic baseline and the matched tuned raw-window logistic head in this run.
- Across the nine frozen-latent variants, Spearman(`test_recon_normalized_mse`, `test_pred_macro_f1`) is `-0.2000`; this is descriptive only and does not support treating overall reconstruction MSE as a reliable downstream proxy.
- For the Step 5 `logistic_regression` sample-level failure view, `spread_mae` has the highest mean AUROC for incorrect prediction (`0.5204`), followed by `top_of_book_mse` (`0.5035`); these associations are weak and diagnostic, not causal claims.

Artifacts from Step 7 live under:

- `results/step7_alignment/`
- `figures/step7_alignment/`

Step 8 fairness and robustness snapshot:

- Tuned raw-window logistic control selected `C=0.1` and reached test macro-F1 `0.3904`, below the untuned Step 5 logistic baseline (`0.3972`).
- The raw-window logistic grid has a post hoc test-oracle point at `C=0.01` with test macro-F1 `0.4101`; this is not selection-valid, and it remains below the best frozen-latent head.
- The post hoc best frozen-latent head remains `last_snapshot_repeat@40` with test macro-F1 `0.4355`.
- Paired bootstrap for best frozen-latent head vs tuned raw-window logistic gives macro-F1 delta `0.0452`, 95% CI `[0.0082, 0.0823]`, and `fraction_delta_gt_0=0.9930`.
- The best frozen-latent head is selected post hoc from Step 7 test macro-F1, so the bootstrap comparison is descriptive rather than a fully pre-registered confirmatory test.
- Rank mismatch persists across all latent variants, but weakens after excluding `last_snapshot_repeat@40`; without it, `pca@128` is both reconstruction-best and prediction-best.
- `last_snapshot_repeat@40` has zero last-step reconstruction error by construction.

Artifacts from Step 8 live under:

- `results/step8_fairness_robustness/`
- `figures/step8_fairness_robustness/`

For a file-by-file map of committed evidence, see `docs/artifact_index.md`. For the current 01-05 reproduction commands, see `docs/reproduction_guide.md`.

## Reproduction

The current main-protocol commands are collected in `docs/reproduction_guide.md`. The pipeline requires the external processed A-share dataset locally and does not commit raw data or generated tensors.

## Scope

This repo does:

- Build a small chronological subset from external LOBench-style A-share data.
- Keep a fixed 10-level, 100-step, 40-feature input contract.
- Establish prediction-only floors before representation experiments.
- Compare reconstruction quality against downstream prediction once alignment analysis is available.
- Break reconstruction error down by book level and regime.
- Track practical costs such as latency and compression only when they clarify the trade-off.

This repo does not:

- Claim SOTA.
- Fully reproduce LOBench or SimLOB.
- Evaluate real trading profitability.
- Commit or redistribute external, private, or proprietary LOB data.
- Generalize one small subset to every symbol, venue, and market period.

## Layout

- `src/data/`: external data loading, field mapping, labels, subset construction.
- `src/models/`: prediction and reconstruction baseline models.
- `src/analysis/`: prediction/reconstruction metrics and diagnostic utilities.
- `scripts/`: runnable stage entry points.
- `docs/`: protocol, artifact index, and reproduction notes.
- `results/`: experiment outputs.
- `figures/`: plots and visual diagnostics.

LOBench and SimLOB remain external references. This repo stores my own scripts, configs, notes, analysis code, and memos.
