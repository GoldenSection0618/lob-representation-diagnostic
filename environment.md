# Environment

This repository runs locally at `~/lob-representation-diagnostic`. The upstream LOBench checkout lives at `~/LOBench`. They are sibling directories. I treat the upstream repo as read-only reference material for data contracts, fields, and label logic; this PoW repo contains only my own code and notes.

## Runtime

- Conda environment: `lob`
- Python scripts should run through `mamba run -n lob ...`.
- Dependency changes should go through `mamba`, not direct `pip install` or direct `conda install`.
- Step 2/3 covered data contract inspection and subset construction. No full model training has been run yet.

Core dependencies:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `pyyaml`
- `tqdm`

Training will later require:

- `torch`
- `lightning` or `pytorch-lightning`, mainly for compatibility with the upstream datamodule style. These are not hard requirements for the current data preparation stage.

## Upstream Reference

The inspected local LOBench commit is `c8fe9e7`.

Files inspected:

- `~/LOBench/data/data_ashare.py`
- `~/LOBench/data/data_processing.py`
- `~/LOBench/data/data_prepare.py`
- `~/LOBench/data/data_sampling.py`
- `~/LOBench/config_template.json`
- `~/LOBench/README.md`

The upstream repo has multiple data entry points. `data_ashare.py` is closer to the A-share main path. `data_processing.py` covers processed/simulation-style data. `data_prepare.py` mainly splits already materialized tensors. That difference matters for labels and split behavior, so I do not copy one file blindly as the project contract.

## Data Paths

These paths may exist locally, but must not be committed:

- `data/raw/`
- `data/processed/`
- `data/external/`
- `~/LOBench/dataset/real_data/`
- `~/LOBench/dataset/train_data/`
- `~/LOBench/dataset/simu_data/`

The external processed dataset comes from Hugging Face dataset `mythezone/LOBench-A-share-processed`. Local path:

- `~/datasets/LOBench-A-share-processed`

I verified that this local directory contains multiple `*-level10_processed.csv` files, including `sz000001`, `sz000002`, `sz002415`, `sz000858`, `sz300147`, and `sz300750`. These files remain external inputs. The repository does not commit CSV, NPZ, PT, or large tensor artifacts.

## Reproducibility Boundary

This PoW currently reproduces field mapping, label generation, window construction, chronological splitting, and metadata recording. It does not reproduce the external dataset itself, because the data stays outside git. Before running experiments, the external dataset path must exist locally.
