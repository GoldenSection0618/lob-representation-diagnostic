# Environment

## System

- PoW repository path: `~/lob-representation-diagnostic`
- Upstream reference path: `~/LOBench`
- Relationship: sibling directories under `~/`; upstream is read-only reference for this PoW.

## Python

- Python version assumption: 3.10.x
- Virtual environment assumption: local `.venv` in PoW repo root (not committed)
- Step 2 scope: inspection + notes only; no full training execution.

## Core Dependencies

- numpy
- pandas
- scikit-learn
- matplotlib
- pyyaml
- tqdm

## Optional Dependencies

- torch (required later for Step 4/5 model training, not required for Step 2 note updates)
- lightning / pytorch-lightning (upstream datamodule usage; optional for current Step 2)

## External Repositories

- LOBench local path: `~/LOBench`
- LOBench commit hash (inspected): `c8fe9e7`
- Inspected files:
  - `data/data_ashare.py`
  - `data/data_processing.py`
  - `data/data_prepare.py`
  - `data/data_sampling.py`
  - `config_template.json`
  - `README.md`

## Data Paths

These paths are expected locally but must not be committed in this PoW repo:

- PoW local data (gitignored):
  - `data/raw/`
  - `data/processed/`
  - `data/external/`
- Upstream reference paths (read-only in Step 2):
  - `~/LOBench/dataset/real_data/`
  - `~/LOBench/dataset/train_data/`
  - `~/LOBench/dataset/simu_data/`

## Reproducibility Notes

- Step 2 outputs are documentation-only contract updates (`data_note.md`, `environment.md`, `execution_log.md`).
- No upstream source code is copied into this repository.
- No real data, large tensors, `.npz`, `.pt`, or `.csv` outputs are committed.
