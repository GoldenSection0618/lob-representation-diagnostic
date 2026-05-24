# Environment

The PoW repo lives at `~/lob-representation-diagnostic`. The upstream LOBench checkout lives next to it at `~/LOBench`. I use upstream as read-only reference material and keep project code, notes, and outputs in this repo.

## Runtime

- Conda environment: `lob`
- Run project Python through `mamba run -n lob ...`.
- Change dependencies with `mamba`; do not use direct `pip install` or direct `conda install` for this project.

Core packages:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `pyyaml`
- `tqdm`

Training package used by the local MLP baselines:

- `torch==2.10.0`

## Upstream Checkout

Inspected LOBench commit: `c8fe9e7`.

Files inspected:

- `~/LOBench/data/data_ashare.py`
- `~/LOBench/data/data_processing.py`
- `~/LOBench/data/data_prepare.py`
- `~/LOBench/data/data_sampling.py`
- `~/LOBench/config_template.json`
- `~/LOBench/README.md`

I use these files to understand the data and label conventions, not as copied project code. The distinction matters because upstream has multiple split and label paths. The PoW uses one conservative baseline protocol for Steps 3-9 and later compares split protocols in Step 10.

## Data

Local paths that may exist but must stay out of git:

- `data/raw/`
- `data/processed/`
- `data/external/`
- `~/LOBench/dataset/real_data/`
- `~/LOBench/dataset/train_data/`
- `~/LOBench/dataset/simu_data/`

External processed data:

- Dataset: `mythezone/LOBench-A-share-processed`
- Local path: `~/datasets/LOBench-A-share-processed`

The local directory contains multiple `*-level10_processed.csv` files, including `sz000001`, `sz000002`, `sz002415`, `sz000858`, `sz300147`, and `sz300750`. They are inputs, not repository artifacts. This repo does not commit CSV, NPZ, PT, or large tensor files.

## Reproducibility Boundary

The repo reproduces field mapping, label generation, window construction, boundary-purged chronological splitting, baseline execution, and metadata recording. It does not reproduce the external dataset itself. A local copy of the external dataset is required before running the scripts.

Step 10 additionally reproduces split-protocol diagnostics on the same kept sample universe.
