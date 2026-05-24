# PoW Execution Log

This log keeps the decisions and run facts that matter for reproduction. It avoids narrative filler.

Note: this file is a chronological archive. Early sections include superseded dense-window pilot facts from before the protocol was revised to LOBench-style `sample_stride=4`. Current main evidence is documented in `README.md`, `docs/data_note.md`, `technical_memo.md`, `docs/evaluation_protocol.md`, and `docs/artifact_index.md`.

## Step 1: Repository Boundary

I initialized `~/lob-representation-diagnostic` as an independent diagnostic repo. It is not a LOBench fork and not a full reproduction.

The question I locked at the start: does better LOB reconstruction reliably transfer into better downstream mid-price trend prediction?

Initial project skeleton at the time. Some support documents were later moved under `docs/` during repository cleanup:

- `README.md`
- `environment.md` (now `docs/environment.md`)
- `data_note.md` (now `docs/data_note.md`)
- `technical_memo.md`
- `configs/` (not used in the final layout)
- `src/`
- `scripts/`
- `results/`
- `figures/`

The first practical task was to inspect the LOBench / SimLOB-style data pipeline and pin down format, loader entry points, labels, split behavior, and minimal subset requirements.

## Step 2: Upstream Inspection

I inspected `~/LOBench` at commit `c8fe9e7`.

Files reviewed:

- `~/LOBench/data/data_ashare.py`
- `~/LOBench/data/data_processing.py`
- `~/LOBench/data/data_prepare.py`
- `~/LOBench/data/data_sampling.py`
- `~/LOBench/config_template.json`
- `~/LOBench/README.md`

What mattered:

- `data_ashare.py` is the main A-share path.
- `data_processing.py` carries the processed/simulation-style feature and label path.
- `data_prepare.py` works on saved tensors and random split, so it is not the main PoW contract.
- The inspected upstream code uses `CSV`, `NPZ`, and `PT`.
- The 40-feature layout expands 10 levels of bid/ask prices and volumes.
- `midPrice = (bestBidPrice1 + bestAskPrice1)/2`.
- `spread = bestAskPrice1 - bestBidPrice1`.
- `data_processing.py` builds three-class trend labels from a thresholded future rolling-mean gap over horizons `{1,3,5,7,10}`.
- `data_ashare.py` uses a different `{-1,0,1}` label convention with a relative threshold; I do not mix it into the main contract.
- Upstream defaults lean random-split. The PoW main protocol is chronological and boundary-purged.

I also verified external processed A-share files under `~/datasets/LOBench-A-share-processed`. They remain outside git.

The inspection left three implementation choices: source field names, whether to regenerate labels, and sampling stride. Step 3 turned those into code and metadata.

## Step 3: Minimal Chronological Subset

I built one clean data path: read one external processed CSV, map it into the canonical 40-feature layout, generate `trend5`, create `window=100` samples, and split them chronologically with boundary purge.

Files touched:

- `src/data/load_lobench.py`
- `src/data/labeling.py`
- `src/data/make_subset.py`
- `src/data/checks.py`
- `scripts/01_prepare_data.py`
- `docs/data_note.md`
- `README.md`
- `docs/archive/execution_log.md`

Dry run:

```bash
mamba run -n lob python scripts/01_prepare_data.py --input-csv ~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv --symbol sz000001 --output-dir data/processed/minimal_subset --window-len 100 --label-horizon 5 --threshold 0.0001 --split-ratio 70/15/15 --row-limit 50000 --max-samples 8000 --dry-run
```

Output run:

```bash
mamba run -n lob python scripts/01_prepare_data.py --input-csv ~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv --symbol sz000001 --output-dir data/processed/minimal_subset --window-len 100 --label-horizon 5 --threshold 0.0001 --split-ratio 70/15/15 --row-limit 50000 --max-samples 8000
```

Run facts:

- Input: `~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv`
- Output: `data/processed/minimal_subset/`
- Metadata: `data/processed/minimal_subset/metadata.json`
- Raw rows used: `50000`
- Usable rows after label trimming: `49990`
- Final samples: `7802`
- `X=(7802, 100, 40)`
- `y=(7802,)`
- Split sizes: `train=5600`, `val=1200`, `test=1002`
- Train max label row: `5698`
- Validation min label row: `5798`
- Validation max label row: `6997`
- Test min label row: `7097`

Checks passed:

- `feature_contract_check`
- `label_contract_check`
- `window_alignment_check`
- `chronological_split_check`
- `output_safety_check`

## Step 4: Evaluation Protocol Lock

Step 4 locked the leakage-aware protocol before baseline modeling.

No data pipeline changes were made. No random split was added. No no-purge chronological split was added. The existing `chronological_split()` path, with mandatory boundary purge, remains the main protocol.

The invariant is:

- Train label rows come before validation label rows, and validation comes before test.
- Train/validation boundary windows do not share historical rows.
- Validation/test boundary windows do not share historical rows.

Step 5 could then evaluate prediction-only baselines without reopening the split policy.

## Step 5: Prediction-Only Baselines

I built prediction floors directly on the locked Step 3 subset. This step measures how far simple supervised models get before any reconstruction or representation learning is introduced.

Files added or modified:

- `src/data/prediction_dataset.py`
- `src/analysis/prediction_metrics.py`
- `src/models/prediction_baselines.py`
- `scripts/02_prediction_baselines.py`
- `results/step5_prediction_baselines/*`
- `figures/step5_prediction_baselines/*`
- `README.md`
- `technical_memo.md`
- `docs/archive/execution_log.md`

Command:

```bash
mamba run -n lob python scripts/02_prediction_baselines.py \
  --subset-dir data/processed/minimal_subset \
  --output-dir results/step5_prediction_baselines \
  --figures-dir figures/step5_prediction_baselines \
  --seed 42 \
  --models majority,logistic_regression,mlp \
  --max-epochs 100 \
  --batch-size 256 \
  --device auto
```

The script used the existing `samples.csv` split and did not re-split data.

Split sizes:

- train: `5600`
- val: `1200`
- test: `1002`

Test metrics:

- `majority`: `accuracy=0.4501`, `balanced_accuracy=0.3333`, `macro_f1=0.2069`, `mcc=0.0000`, `log_loss=1.2228`
- `logistic_regression`: `accuracy=0.4122`, `balanced_accuracy=0.3504`, `macro_f1=0.3338`, `mcc=0.0250`, `log_loss=9.2487`
- `mlp`: `accuracy=0.4531`, `balanced_accuracy=0.3535`, `macro_f1=0.2760`, `mcc=0.0589`, `log_loss=1.7594`

Generated result files:

- `results/step5_prediction_baselines/metrics.csv`
- `results/step5_prediction_baselines/classification_report.json`
- `results/step5_prediction_baselines/directional_metrics.json`
- `results/step5_prediction_baselines/confusion_matrices.json`
- `results/step5_prediction_baselines/prediction_distributions.json`
- `results/step5_prediction_baselines/run_config.json`
- `results/step5_prediction_baselines/summary.md`

Generated figures:

- `figures/step5_prediction_baselines/primary_metrics_by_model.png`
- `figures/step5_prediction_baselines/class_distribution_true_vs_pred.png`
- `figures/step5_prediction_baselines/confusion_matrix_best_model_normalized.png`
- `figures/step5_prediction_baselines/directional_error_summary.png`
- `figures/step5_prediction_baselines/log_loss_by_model.png`

Step 5 kept the Step 3 data contract and Step 4 protocol unchanged. Reconstruction baselines are still the next step.

## Step 6: Reconstruction Baselines

I built reconstruction-only baselines on the locked Step 3 subset without touching subset generation, split construction, or label generation.

Files added or modified:

- `src/models/reconstruction_baselines.py`
- `src/analysis/reconstruction_metrics.py`
- `scripts/03_reconstruction_baselines.py`
- `results/step6_reconstruction_baselines/*`
- `figures/step6_reconstruction_baselines/*`
- `README.md`
- `technical_memo.md`
- `docs/archive/execution_log.md`

Command:

```bash
mamba run -n lob python scripts/03_reconstruction_baselines.py \
  --subset-dir data/processed/minimal_subset \
  --output-dir results/step6_reconstruction_baselines \
  --figures-dir figures/step6_reconstruction_baselines \
  --artifact-dir artifacts/step6_reconstruction_baselines \
  --seed 42 \
  --models train_mean_window,last_snapshot_repeat,pca,mlp_ae \
  --pca-latent-dims 8,16,32,64,128 \
  --mlp-latent-dims 16,32,64 \
  --max-epochs 100 \
  --batch-size 256 \
  --device auto
```

Split sizes (unchanged from Step 3):

- train: `5600`
- val: `1200`
- test: `1002`

Models and latent dimensions run:

- `train_mean_window`
- `last_snapshot_repeat` (`latent_dim=40`)
- `pca` with `latent_dim in {8,16,32,64,128}`
- `mlp_ae` with `latent_dim in {16,32,64}`

Best test reconstruction by normalized MSE:

- `pca@128`
- `normalized_mse=0.191170`
- `normalized_mae=0.239629`
- `original_mae=0.135957`
- `relative_mse_vs_last_snapshot=0.288494`

Generated result files:

- `results/step6_reconstruction_baselines/metrics.csv`
- `results/step6_reconstruction_baselines/rate_distortion.csv`
- `results/step6_reconstruction_baselines/feature_group_errors.csv`
- `results/step6_reconstruction_baselines/level_wise_errors.csv`
- `results/step6_reconstruction_baselines/temporal_errors.csv`
- `results/step6_reconstruction_baselines/derived_lob_errors.csv`
- `results/step6_reconstruction_baselines/per_sample_reconstruction_errors.csv`
- `results/step6_reconstruction_baselines/model_manifest.json`
- `results/step6_reconstruction_baselines/latent_manifest.json`
- `results/step6_reconstruction_baselines/run_config.json`
- `results/step6_reconstruction_baselines/summary.md`

Generated figures:

- `figures/step6_reconstruction_baselines/rate_distortion_curve.png`
- `figures/step6_reconstruction_baselines/reconstruction_scorecard_by_model.png`
- `figures/step6_reconstruction_baselines/feature_group_error_by_model.png`
- `figures/step6_reconstruction_baselines/level_wise_error_heatmap_best_model.png`
- `figures/step6_reconstruction_baselines/temporal_error_profile.png`

Protocol invariants preserved:

- Step 6 did not modify Step 3 subset generation or data contract.
- Step 6 did not modify Step 4 boundary-purged chronological split policy.
- Step 6 did not add random split or no-purge split.
- Step 6 did not train prediction heads.
- Step 6 did not run reconstruction-prediction alignment.

Next step is Step 7 alignment analysis using `per_sample_reconstruction_errors.csv` together with prediction outputs under the same locked protocol.

## Step 6 Artifact Contract Fix: model_variant Key

I patched Step 6 output artifacts to add `model_variant` as the unique model-configuration key, while keeping `model` as the base family key.

Applied to:

- `results/step6_reconstruction_baselines/metrics.csv`
- `results/step6_reconstruction_baselines/rate_distortion.csv`
- `results/step6_reconstruction_baselines/feature_group_errors.csv`
- `results/step6_reconstruction_baselines/level_wise_errors.csv`
- `results/step6_reconstruction_baselines/temporal_errors.csv`
- `results/step6_reconstruction_baselines/derived_lob_errors.csv`
- `results/step6_reconstruction_baselines/per_sample_reconstruction_errors.csv`

Variant mapping rule now used:

- `train_mean_window` -> `train_mean_window`
- `last_snapshot_repeat` + `latent_dim=40` -> `last_snapshot_repeat@40`
- `pca` + `latent_dim=d` -> `pca@d`
- `mlp_ae` + `latent_dim=d` -> `mlp_ae@d`

This change does not alter Step 3 data, Step 4 protocol, Step 6 split behavior, or any model-training logic.

## Step 6 Artifact Schema Sync: Canonical Variant Key

I re-ran Step 6 generation so `model_variant` is produced natively by `scripts/03_reconstruction_baselines.py` across all Step 6 long tables and `model_manifest.json`.

Canonical key policy remains:

- `model`: model family
- `latent_dim`: numeric compression dimension
- `model_variant`: unique variant identifier for Step 7 joins and grouping

No change was made to Step 3 data, Step 4 split protocol, or Step 6 model set.

## Step 6 Metric Safety Patch: Imbalance Validity Gate

I patched reconstruction-derived volume diagnostics to avoid unstable imbalance MAE values on processed features.

Changes:

- Added validity checks for imbalance diagnostics:
  - `volume_nonnegative_ratio`
  - `imbalance_denominator_small_ratio`
  - `imbalance_valid_ratio`
- `top1_imbalance_mae` and `top5_imbalance_mae` are now computed only on valid points.
- If valid ratio is below threshold (`0.95`), imbalance MAE is set to null and validity flags are false.
- Added fallback diagnostics independent of non-negative-ratio assumptions:
  - `top1_volume_sum_mae`, `top5_volume_sum_mae`
  - `top1_volume_diff_mae`, `top5_volume_diff_mae`

Outputs updated by re-running Step 6:

- `results/step6_reconstruction_baselines/derived_lob_errors.csv`
- `results/step6_reconstruction_baselines/per_sample_reconstruction_errors.csv`
- `results/step6_reconstruction_baselines/summary.md`

Protocol scope unchanged:

- No Step 3 data change.
- No Step 4 split/protocol change.
- No random split or no-purge split added.
- No prediction-head training added.

## Step 5 Interface Patch: Per-Sample Prediction Outputs

I added per-sample prediction exports required for Step 7 alignment.

Updated script:

- `scripts/02_prediction_baselines.py`

New output:

- `results/step5_prediction_baselines/per_sample_predictions.csv`

Schema includes:

- `sample_id`, `original_sample_id`, `split`, `label_row`, `y_true`
- `model`, `y_pred`, `correct`, `confidence`
- `proba_0`, `proba_1`, `proba_2`
- `is_non_neutral_true`, `is_non_neutral_pred`
- `direction_correct_non_neutral`, `opposite_direction_error`

This patch does not change prediction training logic. It only exposes sample-level outputs for Step 7 joins.

## Step 6 Reproducibility and Figure Sync Patch

I synchronized Step 6 artifacts with the imbalance-gating and fallback-diagnostic contract.

Updated script:

- `scripts/03_reconstruction_baselines.py`

Changes:

- `run_config.json` now records imbalance-gate parameters (`eps_threshold`, `valid_ratio_threshold`, valid condition, invalid policy).
- Added figure: `figures/step6_reconstruction_baselines/derived_lob_error_by_model.png`.
- `summary.md` now includes the gate thresholds and keeps fallback volume diagnostics explicit.

Protocol scope unchanged:

- No Step 3 data changes.
- No Step 4 split/protocol changes.
- No random split / no-purge split.
- No prediction-head training.

## Step 5 Semantics Patch: Non-Neutral Direction Field

I fixed per-sample directional semantics in `results/step5_prediction_baselines/per_sample_predictions.csv`.

- `direction_correct_non_neutral` is encoded as 1.0/0.0 when `y_true in {0,2}`.
- Neutral samples store this field as null instead of false.
- `opposite_direction_error` remains boolean, with neutral rows as false.

This avoids accidental denominator leakage if Step 7 aggregates directional correctness directly from the per-sample table.

I also made the Step 5 per-sample prediction interface self-describing in `run_config.json`:

- `per_sample_predictions.csv` is generated for val/test only.
- Step 7 joins predictions to reconstruction diagnostics on `sample_id` and `split`.
- Step 5 `model` means prediction model and should be renamed to `prediction_model` before Step 7 alignment.
- `direction_correct_non_neutral` uses numeric 1.0/0.0 for true non-neutral samples and null for neutral samples.

## Step 6 Metric-Space Clarification

I added explicit metric-space notes to avoid misreading `original_mae` / `original_rmse` as raw exchange-scale errors.

- `run_config.json` now includes `metric_space_note`:
  - normalized space: after Step 6 train-only scaler
  - original space: Step 3 input feature space before Step 6 scaler (not raw order-flow scale)

I also synchronized this note in Step 6 summary and technical memo text.

## Step 5/6 LOBench-Compatible Metric Bridge

I added a small metric-bridge layer without changing the Step 5 or Step 6 training objectives.

Step 5 now records the model-selection policy explicitly:

- primary diagnostic metric: macro-F1, tie-broken by MCC then log loss
- LOBench reference metric: `cross_entropy_loss` / `log_loss`
- best by macro-F1: `logistic_regression`
- best by log loss / cross entropy: `majority`

Step 6 now exports `results/step6_reconstruction_baselines/lobench_compatible_reconstruction_metrics.csv`.

The new Step 6 file includes LOBench-style MSE, MAE, price loss, volume loss, weighted price loss, weighted volume loss, weighted MSE, regularization loss, and all loss. These are evaluation-only metrics in normalized Step 6 space with `factor=1.5`, `alpha=0.8`, and `reg_factor=10.0`; reconstruction training objectives are unchanged.

I reran Step 6 with `--save-latents` after the code change. The Step 6 figures were regenerated by the script but produced no git diff.

## Step 6 Plot Readability Patch: Derived LOB Error Scale Separation

I fixed `derived_lob_error_by_model.png` to avoid mixing price-scale and volume-scale metrics on one y-axis.

Updated plotting logic:

- Upper subplot: `midprice_mae`, `spread_mae`
- Lower subplot: `top1_volume_sum_mae`, `top5_volume_sum_mae`, `top1_volume_diff_mae`, `top5_volume_diff_mae`

This is a visualization-only patch; it does not change Step 6 reconstruction metrics.

## Protocol Revision: Switch Main Sampling from stride=1 to stride=4

I revised the main sampling protocol from dense stride-1 pilot windows to LOBench-style `sample_stride=4`.

- The earlier stride-1 run was a dense-window pilot, not the final main protocol.
- The main protocol is now `sample_stride=4` plus the existing boundary-purged chronological split.
- Reason: reduce near-duplicate overlapping windows and align with upstream LOBench-style A-share sampling convention.
- Boundary-purged chronological splitting remains unchanged.
- Random split and no-purge chronological split remain excluded from the main protocol.
- Old Step 3/5/6 generated artifacts were removed before the stride-4 rerun.

Removed generated directories:

```bash
rm -rf data/processed/minimal_subset
rm -rf results/step5_prediction_baselines
rm -rf figures/step5_prediction_baselines
rm -rf results/step6_reconstruction_baselines
rm -rf figures/step6_reconstruction_baselines
rm -rf artifacts/step6_reconstruction_baselines
```

## Step 3 Rerun: sample_stride=4 Main Subset

I first ran the stride-4 Step 3 command with `--dry-run` as a preflight, then ran the same command without `--dry-run` because downstream Step 5/6 require `X.npy`, `y.npy`, and `samples.csv`.

Dry-run command:

```bash
mamba run -n lob python scripts/01_prepare_data.py \
  --input-csv ~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv \
  --symbol sz000001 \
  --output-dir data/processed/minimal_subset \
  --window-len 100 \
  --label-horizon 5 \
  --threshold 0.0001 \
  --split-ratio 70/15/15 \
  --row-limit 50000 \
  --max-samples 8000 \
  --sample-stride 4 \
  --dry-run
```

Generation command:

```bash
mamba run -n lob python scripts/01_prepare_data.py \
  --input-csv ~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv \
  --symbol sz000001 \
  --output-dir data/processed/minimal_subset \
  --window-len 100 \
  --label-horizon 5 \
  --threshold 0.0001 \
  --split-ratio 70/15/15 \
  --row-limit 50000 \
  --max-samples 8000 \
  --sample-stride 4
```

Actual stride-4 subset facts:

- total samples after boundary purge: `7952`
- split sizes: train=`5600`, val=`1200`, test=`1152`
- `X` shape: `(7952, 100, 40)`
- `y` shape: `(7952,)`
- label row ranges: train `99..22495`, val `22595..27391`, test `27491..32095`
- dropped boundary samples: `48`
- boundary purge applied: `true`
- train/val and val/test historical rows do not overlap
- sample IDs are contiguous after final keep

## Step 5 Rerun: Prediction Baselines on sample_stride=4

Command:

```bash
mamba run -n lob python scripts/02_prediction_baselines.py \
  --subset-dir data/processed/minimal_subset \
  --output-dir results/step5_prediction_baselines \
  --figures-dir figures/step5_prediction_baselines \
  --seed 42 \
  --models majority,logistic_regression,mlp \
  --max-epochs 100 \
  --batch-size 256 \
  --device auto
```

Split sizes:

- train: `5600`
- val: `1200`
- test: `1152`

Test metrics:

- `majority`: `accuracy=0.6441`, `balanced_accuracy=0.3333`, `macro_f1=0.2612`, `mcc=0.0000`, `log_loss=0.8980`
- `logistic_regression`: `accuracy=0.4826`, `balanced_accuracy=0.4098`, `macro_f1=0.3972`, `mcc=0.1007`, `log_loss=4.1624`
- `mlp`: `accuracy=0.4036`, `balanced_accuracy=0.4513`, `macro_f1=0.3816`, `mcc=0.1624`, `log_loss=1.2767`

Best test model by macro-F1 tie-broken by MCC then log loss: `logistic_regression`.

Validation checks:

- `per_sample_predictions.csv` contains val/test rows for all three prediction models.
- `direction_correct_non_neutral` is null for true neutral rows and non-null for true down/up rows.
- Step 5 remains prediction-only; no reconstruction models are used.

## Step 6 Rerun: Reconstruction Baselines on sample_stride=4

Command:

```bash
mamba run -n lob python scripts/03_reconstruction_baselines.py \
  --subset-dir data/processed/minimal_subset \
  --output-dir results/step6_reconstruction_baselines \
  --figures-dir figures/step6_reconstruction_baselines \
  --artifact-dir artifacts/step6_reconstruction_baselines \
  --seed 42 \
  --models train_mean_window,last_snapshot_repeat,pca,mlp_ae \
  --pca-latent-dims 8,16,32,64,128 \
  --mlp-latent-dims 16,32,64 \
  --max-epochs 100 \
  --batch-size 256 \
  --device auto
```

Split sizes:

- train: `5600`
- val: `1200`
- test: `1152`

Best test reconstruction:

- best by normalized MSE: `pca@128`
- `normalized_mse=0.183824`
- `normalized_mae=0.187050`
- `original_mae=0.126703`
- `relative_mse_vs_last_snapshot=0.100333`
- best LOBench-compatible weighted MSE: `pca@128` (`weighted_mse_loss=0.291670`, `all_loss=1.206065`)

Validation checks:

- `metrics.csv`, `rate_distortion.csv`, `derived_lob_errors.csv`, and `per_sample_reconstruction_errors.csv` were regenerated.
- `per_sample_reconstruction_errors.csv` has `79520` rows, equal to `7952` samples x `10` reconstruction variants.
- `model_variant` remains the canonical reconstruction variant key.
- `run_config.json` includes `imbalance_gate`, `metric_space_note`, Step 4 protocol note, and Step 3 metadata summary.
- Step 6 remains reconstruction-only and does not train prediction heads.
- No `artifacts/` files were committed in the Step 6 results commit; latent artifacts were refreshed later under the same stride-4 protocol.

## sample_stride=4 Protocol Revision Commits

Commits made for the main protocol revision:

- `8bb48b7` - `step3: add stride-aware window sampling`
- `c50b4a4` - `chore: remove stride1 generated artifacts`
- `a6da617` - `step3: rerun subset with sample_stride4`
- `58c5c1e` - `step3: expose stride in metadata summary`
- `67064d5` - `step5: rerun prediction baselines with sample_stride4`
- `9e25234` - `step6: rerun reconstruction baselines with sample_stride4`
- `3b1fb9c` - `step6: refresh stride4 latent artifacts`

## Step 6 Latent Artifact Refresh

I refreshed Step 6 latent artifacts for the stride-4 main protocol.

Command:

```bash
mamba run -n lob python scripts/03_reconstruction_baselines.py \
  --subset-dir data/processed/minimal_subset \
  --output-dir results/step6_reconstruction_baselines \
  --figures-dir figures/step6_reconstruction_baselines \
  --artifact-dir artifacts/step6_reconstruction_baselines \
  --seed 42 \
  --models train_mean_window,last_snapshot_repeat,pca,mlp_ae \
  --pca-latent-dims 8,16,32,64,128 \
  --mlp-latent-dims 16,32,64 \
  --max-epochs 100 \
  --batch-size 256 \
  --device auto \
  --save-latents
```

Validation:

- `latent_manifest.json` now records `latents_saved=true` and `save_latents_flag=true`.
- 27 local `.npy` latent files were generated under `artifacts/step6_reconstruction_baselines/latents/`.
- Manifest shapes match the saved local latent arrays.
- The train-mean baseline has no latent representation, so its three manifest entries remain `latents_saved=false`.
- No `artifacts/` files were committed.
- Step 6 best reconstruction remains `pca@128` with unchanged reconstruction metric values; rerun differences in `metrics.csv` and `model_manifest.json` are timing fields.

## Step 7: Reconstruction-Prediction Alignment

Command:

```bash
mamba run -n lob python scripts/04_alignment_analysis.py \
  --step5-dir results/step5_prediction_baselines \
  --step6-dir results/step6_reconstruction_baselines \
  --latent-artifact-dir artifacts/step6_reconstruction_baselines/latents \
  --output-dir results/step7_alignment \
  --figures-dir figures/step7_alignment \
  --seed 42 \
  --head-c-grid 0.01,0.1,1.0,10.0 \
  --primary-prediction-model-for-sample-analysis logistic_regression \
  --selection-metric macro_f1
```

Inputs consumed:

- `results/step5_prediction_baselines/metrics.csv`
- `results/step5_prediction_baselines/per_sample_predictions.csv`
- `results/step5_prediction_baselines/run_config.json`
- `results/step6_reconstruction_baselines/metrics.csv`
- `results/step6_reconstruction_baselines/lobench_compatible_reconstruction_metrics.csv`
- `results/step6_reconstruction_baselines/per_sample_reconstruction_errors.csv`
- `results/step6_reconstruction_baselines/latent_manifest.json`
- local latent arrays under `artifacts/step6_reconstruction_baselines/latents/`

Join contract:

- status: `passed`
- expected joined rows: `70560`
- actual joined rows: `70560`
- duplicate key count: `0`
- Step 5 predictions covered val/test only.
- Step 6 reconstruction diagnostics covered train/val/test; Step 7 sample-level alignment used val/test only.

Latent transfer setup:

- latent variants used: `last_snapshot_repeat@40`, `pca@8`, `pca@16`, `pca@32`, `pca@64`, `pca@128`, `mlp_ae@16`, `mlp_ae@32`, `mlp_ae@64`
- head model: logistic regression with train-only latent `StandardScaler`
- class weighting: `balanced`
- C grid: `0.01, 0.1, 1.0, 10.0`
- selection: validation macro-F1, tie-broken by validation MCC then validation log loss
- reconstruction encoders were loaded only through saved latent arrays and were not retrained

Best latent-head test result:

- best variant: `last_snapshot_repeat@40`
- selected C: `10.0`
- test macro-F1: `0.435540`
- test balanced accuracy: `0.550928`
- test MCC: `0.257922`
- test log loss: `1.074270`
- best Step 5 raw-window baseline by test macro-F1: `logistic_regression` (`0.397216`)
- matched raw-window logistic head: `raw_window_logistic_tuned`, selected C `0.1`, test macro-F1 `0.390383`, test balanced accuracy `0.407893`, test MCC `0.097794`, test log loss `2.049989`
- frozen latent head beat the fixed Step 5 raw-window logistic baseline by test macro-F1: `true`
- frozen latent head beat the matched tuned raw-window logistic head by test macro-F1: `true`

Reconstruction-prediction rank result:

- best test reconstruction normalized-MSE variant: `pca@128` (`0.183824`)
- best test frozen-head macro-F1 variant: `last_snapshot_repeat@40` (`0.435540`)
- same variant: `false`
- Spearman(`test_recon_normalized_mse`, `test_pred_macro_f1`): `-0.200000`
- Spearman(`test_recon_last_step_mse`, `test_pred_macro_f1`): `-0.733333`
- These correlations are descriptive only because there are only nine frozen-latent variants.

Sample-level failure diagnostic snapshot:

- primary sample-analysis prediction model: `logistic_regression`
- highest mean AUROC for incorrect prediction on test: `spread_mae` (`0.520404`)
- `top_of_book_mse` mean AUROC for incorrect prediction on test: `0.503453`
- `normalized_mse` mean AUROC for incorrect prediction on test: `0.474387`
- This does not support using overall reconstruction MSE alone as the downstream prediction proxy.

Generated result files:

- `results/step7_alignment/join_contract.json`
- `results/step7_alignment/sample_alignment_panel.csv`
- `results/step7_alignment/sample_diagnostic_association.csv`
- `results/step7_alignment/error_quantile_response.csv`
- `results/step7_alignment/failure_mode_error_delta.csv`
- `results/step7_alignment/latent_head_metrics.csv`
- `results/step7_alignment/latent_head_predictions.csv`
- `results/step7_alignment/transfer_baseline_comparison.csv`
- `results/step7_alignment/model_level_rank_alignment.csv`
- `results/step7_alignment/model_level_correlations.csv`
- `results/step7_alignment/run_config.json`
- `results/step7_alignment/summary.md`

Generated figures:

- `figures/step7_alignment/transfer_vs_raw_baselines.png`
- `figures/step7_alignment/latent_head_primary_metrics.png`
- `figures/step7_alignment/reconstruction_prediction_rank_alignment.png`
- `figures/step7_alignment/compression_prediction_tradeoff.png`
- `figures/step7_alignment/diagnostic_outcome_association_heatmap.png`
- `figures/step7_alignment/error_quantile_failure_curve.png`

Scope guard:

- Step 7 uses the locked stride-4 boundary-purged chronological protocol.
- No random split or no-purge split was added.
- No Step 3/4/5/6 outputs were regenerated for Step 7.
- No reconstruction encoders were retrained.
- Evidence remains limited to `sz000001`, `trend5`, and this stride-4 subset.

## Step 7 Matched Raw-Window Head Addendum

I added a matched raw-window logistic head to avoid comparing tuned frozen-latent heads against only a fixed-C Step 5 logistic baseline.

Implementation change:

- `scripts/04_alignment_analysis.py` now loads `data/processed/minimal_subset/X.npy`, `y.npy`, and `samples.csv`.
- It flattens the raw Step 3 windows and trains `raw_window_logistic_tuned` with the same head policy used for frozen latents.
- The policy is train-only `StandardScaler`, `class_weight="balanced"`, C grid `0.01, 0.1, 1.0, 10.0`, validation macro-F1 selection, MCC tie-break, then log-loss tie-break.
- Test remains final evaluation only.

Rerun command:

```bash
mamba run -n lob python scripts/04_alignment_analysis.py \
  --step5-dir results/step5_prediction_baselines \
  --step6-dir results/step6_reconstruction_baselines \
  --latent-artifact-dir artifacts/step6_reconstruction_baselines/latents \
  --output-dir results/step7_alignment \
  --figures-dir figures/step7_alignment \
  --seed 42 \
  --head-c-grid 0.01,0.1,1.0,10.0 \
  --primary-prediction-model-for-sample-analysis logistic_regression \
  --selection-metric macro_f1
```

Matched comparison result on test:

- `last_snapshot_repeat@40`: macro-F1 `0.435540`
- fixed Step 5 `logistic_regression`: macro-F1 `0.397216`
- `raw_window_logistic_tuned`: selected C `0.1`, macro-F1 `0.390383`

Interpretation after this addendum:

- The best frozen latent head beats both the fixed Step 5 raw-window logistic baseline and the matched tuned raw-window logistic head in this run.
- This strengthens the transfer comparison but does not change the scope guard: the evidence remains one symbol, one horizon, and one stride-4 subset.

## Step 8: Fairness and Robustness Checks

Command:

```bash
mamba run -n lob python scripts/05_fairness_robustness.py \
  --subset-dir data/processed/minimal_subset \
  --step5-dir results/step5_prediction_baselines \
  --step6-dir results/step6_reconstruction_baselines \
  --step7-dir results/step7_alignment \
  --output-dir results/step8_fairness_robustness \
  --figures-dir figures/step8_fairness_robustness \
  --seed 42 \
  --c-grid 0.01,0.1,1.0,10.0 \
  --bootstrap-iterations 1000 \
  --selection-metric macro_f1 \
  --primary-metric macro_f1
```

Tuned raw-window logistic control:

- input: flattened raw Step 3 windows, input_dim `4000`
- scaler: train-only `StandardScaler`
- classifier: `LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs")`
- C grid: `0.01, 0.1, 1.0, 10.0`
- selected C: `0.1`
- selected by: validation macro-F1, tie-broken by validation MCC then validation log loss
- validation macro-F1: `0.489837`
- test macro-F1: `0.390383`
- tuned raw logistic beat untuned Step 5 logistic: `false`
- raw logistic test-oracle best C: `0.01`
- raw logistic test-oracle test macro-F1: `0.410148`
- oracle status: post hoc reference only, not a valid model-selection baseline

Fair comparison result:

- untuned Step 5 logistic test macro-F1: `0.397216`
- tuned raw-window logistic test macro-F1: `0.390383`
- raw-window logistic test-oracle macro-F1: `0.410148`
- best frozen latent head: `last_snapshot_repeat@40`
- best frozen latent head selection basis: `posthoc_best_test_macro_f1_from_step7`
- best frozen latent head test macro-F1: `0.435540`
- pca@128 frozen latent head test macro-F1: `0.362361`
- best frozen latent head delta vs tuned raw logistic: `0.045157`
- best frozen latent head delta vs raw logistic test-oracle: `0.025392`

Paired bootstrap result:

- comparison: `best_frozen_latent_head` vs `raw_window_logistic_tuned`
- metric: macro-F1
- n bootstrap: `1000`
- delta observed: `0.045157`
- 95% CI: `[0.008235, 0.082278]`
- fraction_delta_gt_0: `0.993000`
- interpretation: descriptive paired test-sample robustness check; best latent variant was selected post hoc from Step 7 test macro-F1.

Rank sensitivity result:

- all latent variants: `rank_mismatch_persists`
- exclude `last_snapshot_repeat@40`: `rank_mismatch_weakens`
- pca only: `rank_mismatch_weakens`
- mlp_ae only: `rank_mismatch_persists`
- After excluding `last_snapshot_repeat@40`, `pca@128` is both reconstruction-best and prediction-best.

Last-snapshot sensitivity:

- `last_snapshot_repeat@40` has `last_step_mse=0` by construction.
- With last snapshot included, best prediction variant is `last_snapshot_repeat@40`.
- Without last snapshot, best prediction variant is `pca@128`.
- last-step-MSE Spearman vs macro-F1 changes from `-0.733333` to `-0.619048`.

Generated result files:

- `results/step8_fairness_robustness/raw_logistic_tuning_grid.csv`
- `results/step8_fairness_robustness/raw_logistic_tuned_metrics.csv`
- `results/step8_fairness_robustness/raw_logistic_tuned_predictions.csv`
- `results/step8_fairness_robustness/fair_transfer_comparison.csv`
- `results/step8_fairness_robustness/paired_bootstrap_delta.csv`
- `results/step8_fairness_robustness/rank_sensitivity.csv`
- `results/step8_fairness_robustness/last_snapshot_sensitivity.csv`
- `results/step8_fairness_robustness/final_claim_table.csv`
- `results/step8_fairness_robustness/run_config.json`
- `results/step8_fairness_robustness/summary.md`

Generated figures:

- `figures/step8_fairness_robustness/fair_transfer_macro_f1_with_ci.png`
- `figures/step8_fairness_robustness/tuning_grid_val_test_curve.png`
- `figures/step8_fairness_robustness/rank_sensitivity_by_variant_set.png`
- `figures/step8_fairness_robustness/last_snapshot_sensitivity.png`

Scope guard:

- Step 8 does not modify Step 3 data construction or `chronological_split()`.
- No random split or no-purge split was added.
- No reconstruction encoders were retrained.
- No new reconstruction models were introduced.
- No multi-symbol or multi-horizon expansion was run.
- Latent path reproducibility had already been fixed before Step 8 and was not redone here.

## Step 8 Rank Sensitivity Rank-Field Fix

I corrected the rank fields in `rank_sensitivity.csv` so they are explicit about rank scope.

Issue:

- `best_reconstruction_variant`, `best_prediction_variant`, and `same_best_variant` were already computed within each variant set.
- The rank fields used alongside them were inherited from Step 7 all-variant global ranks.
- This made rows such as `exclude_last_snapshot_repeat` look inconsistent: `pca@128` was both reconstruction-best and prediction-best within the set, but its prediction rank was shown as global rank `2`.

Fix:

- `scripts/05_fairness_robustness.py` now recomputes subset-local ranks inside each `variant_set`.
- `rank_sensitivity.csv` now writes:
  - `reconstruction_best_rank_pred_macro_f1_within_set`
  - `prediction_best_rank_recon_mse_within_set`
  - `reconstruction_best_rank_pred_macro_f1_global`
  - `prediction_best_rank_recon_mse_global`

Validation:

- For `exclude_last_snapshot_repeat`, `pca_only`, and `high_capacity_latent_dim_gt_40`, the within-set ranks are now `1/1` when `pca@128` is both reconstruction-best and prediction-best.
- Global ranks are still retained as explicit reference fields.
- Step 8 conclusions are unchanged.
