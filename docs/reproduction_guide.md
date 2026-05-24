# Reproduction Guide

This guide collects the current main-protocol and diagnostic commands. It assumes the external processed A-share dataset exists locally and is not committed to this repository.

External input:

- `~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv`

Runtime convention:

- Use `mamba run -n lob python ...` when the `lob` environment is available.
- Keep raw data, NumPy arrays, latent arrays, and checkpoints out of git.

## Step 3: Build Stride-4 Subset

```bash
DATA_CSV=~/datasets/LOBench-A-share-processed/sz000001-level10_processed.csv

mamba run -n lob python scripts/01_prepare_data.py \
  --input-csv "$DATA_CSV" \
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

Expected current subset facts:

- total samples: `7952`
- split sizes: train `5600`, val `1200`, test `1152`
- boundary drops: `48`

## Step 4: Protocol Lock

Step 4 is a protocol-lock/documentation step and has no separate runnable command.

## Step 5: Prediction Baselines

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

## Step 6: Reconstruction Baselines and Latents

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

Local latent arrays are required by Step 7 but remain ignored under `artifacts/`.

## Step 7: Reconstruction-Prediction Alignment

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

## Step 8: Fairness and Robustness

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

## Step 9: Validation-Selected Transfer Audit

```bash
mamba run -n lob python scripts/06_validation_selected_transfer.py \
  --step5-dir results/step5_prediction_baselines \
  --step7-dir results/step7_alignment \
  --step8-dir results/step8_fairness_robustness \
  --output-dir results/step9_validation_selection_audit \
  --bootstrap-iterations 1000 \
  --seed 42
```

## Step 10: Split Protocol Decomposition

```bash
mamba run -n lob python scripts/07_split_protocol_decomposition.py \
  --subset-dir data/processed/minimal_subset \
  --output-dir results/step10_split_protocol_decomposition \
  --figures-dir figures/step10_split_protocol_decomposition \
  --random-seeds 42,43,44,45,46 \
  --block-size 512 \
  --embargo-size 25
```

Step 10 uses the existing stride-4 sample universe and compares `chronological_purged`, `random_window_naive`, `random_block_purged`, and `chronological_no_purge`. It is a protocol-layer diagnostic, not a full Step 6 to Step 9 rerun. This command regenerates both Step 10 CSV artifacts and Step 10 figures.

## Validation

```bash
python -m compileall src scripts
```

Protocol guard checks:

```bash
grep -R "random_split" -n src scripts || true
grep -R "no-purge" -n src scripts || true
grep -R "fit.*PCA" -n scripts/04_alignment_analysis.py scripts/05_fairness_robustness.py src/analysis || true
grep -R "MLPAutoencoderReconstructor" -n scripts/04_alignment_analysis.py scripts/05_fairness_robustness.py src/analysis || true
```

Interpretation:

- `random_split` should not appear as an uncontrolled data module call.
- `random_window_naive` and `chronological_no_purge` are expected in Step 10 as explicitly labeled diagnostics.
- Step 7, Step 8, and Step 9 should not fit PCA or retrain MLP-AE reconstruction encoders.
