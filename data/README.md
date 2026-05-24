# Local Data Directory

`data/` is a local workspace for external inputs and generated intermediate data used by the diagnostic pipeline.

Only this README is committed from `data/`. Raw exchange data, processed CSV inputs, generated NumPy arrays, model checkpoints, and latent arrays must remain untracked.

When reproduced, the current main subset is generated locally under `data/processed/minimal_subset/` by `scripts/01_prepare_data.py` from the external A-share processed dataset.

Expected local input location:

- `~/datasets/LOBench-A-share-processed/`

Git policy is enforced by `.gitignore`: data files stay local, committed result and audit artifacts live under `results/`, and committed visual diagnostics live under `figures/`.
