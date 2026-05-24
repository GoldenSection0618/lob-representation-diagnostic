# Local Data Directory

`data/` is reserved for local, generated, or external data used to run the diagnostic pipeline.

The repository does not commit raw exchange data, processed CSV inputs, NumPy arrays, model checkpoints, or latent artifacts. The current main subset is generated locally under `data/processed/minimal_subset/` by `scripts/01_prepare_data.py` from the external A-share processed dataset.

Expected local input location:

- `~/datasets/LOBench-A-share-processed/`

Git policy is enforced by `.gitignore`: data files stay local, while lightweight result summaries and committed experiment artifacts live under `results/` and `figures/`.
