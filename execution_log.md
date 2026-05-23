# PoW Execution Log

This log keeps the decisions and run facts that matter for reproduction. It avoids narrative filler.

## Step 1: Repository Boundary

I initialized `~/lob-representation-diagnostic` as an independent diagnostic repo. It is not a LOBench fork and not a full reproduction.

The question I locked at the start: does better LOB reconstruction reliably transfer into better downstream mid-price trend prediction?

Initial project skeleton:

- `README.md`
- `environment.md`
- `data_note.md`
- `technical_memo.md`
- `configs/`
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
- `data_note.md`
- `README.md`
- `execution_log.md`

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
- `execution_log.md`

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
- `execution_log.md`

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
