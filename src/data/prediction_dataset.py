"""Load Step 3 subset arrays for prediction-only baselines without re-splitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


VALID_SPLITS = ("train", "val", "test")


def _validate_subset(X: np.ndarray, y: np.ndarray, samples: pd.DataFrame) -> None:
    if len(X) != len(y) or len(X) != len(samples):
        raise ValueError(
            f"Length mismatch: len(X)={len(X)}, len(y)={len(y)}, len(samples)={len(samples)}"
        )

    if "split" not in samples.columns:
        raise ValueError("samples.csv must contain a 'split' column.")

    split_values = set(samples["split"].astype(str).unique().tolist())
    if not split_values.issubset(set(VALID_SPLITS)):
        raise ValueError(
            f"Invalid split values in samples.csv: {sorted(split_values)}; expected subset of {VALID_SPLITS}."
        )

    for split in VALID_SPLITS:
        if (samples["split"] == split).sum() == 0:
            raise ValueError(f"Split '{split}' is empty in samples.csv.")

    if "sample_id" in samples.columns:
        sid = samples["sample_id"].to_numpy()
        if len(sid) > 1 and not np.all(np.diff(sid) > 0):
            raise ValueError("sample_id must be strictly increasing.")


def load_prediction_arrays(subset_dir: str | Path) -> Dict[str, object]:
    """Load Step 3 subset files and return fixed train/val/test arrays."""
    subset_path = Path(subset_dir).expanduser().resolve()
    x_path = subset_path / "X.npy"
    y_path = subset_path / "y.npy"
    samples_path = subset_path / "samples.csv"
    metadata_path = subset_path / "metadata.json"

    missing = [p for p in [x_path, y_path, samples_path, metadata_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing subset files: {[str(p) for p in missing]}")

    X = np.load(x_path)
    y = np.load(y_path)
    samples = pd.read_csv(samples_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    _validate_subset(X, y, samples)

    out: Dict[str, object] = {
        "metadata": metadata,
        "sample_tables": {},
    }

    for split in VALID_SPLITS:
        mask = samples["split"] == split
        split_samples = samples.loc[mask].copy().reset_index(drop=True)
        idx = np.where(mask.to_numpy())[0]
        out[split] = (X[idx], y[idx])
        out["sample_tables"][split] = split_samples

    return out
