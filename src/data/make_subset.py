"""Chronological subset construction from canonical features and labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SubsetData:
    """Container for subset arrays and sample index table."""

    X: np.ndarray
    y: np.ndarray
    sample_table: pd.DataFrame
    metadata: Dict[str, object]


def build_sliding_windows(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    window_len: int = 100,
    label_col: str = "trend5",
    sample_stride: int = 4,
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build chronological sliding windows without shuffle or balancing."""
    if label_col not in labels.columns:
        raise ValueError(f"label_col '{label_col}' not found in labels.")
    sample_stride = int(sample_stride)
    if sample_stride <= 0:
        raise ValueError("sample_stride must be a positive integer.")

    feature_values = features.to_numpy(dtype=np.float32)
    label_values = labels[label_col].to_numpy(dtype=np.int64)

    if len(feature_values) != len(label_values):
        raise ValueError("Feature and label row counts do not match.")
    if len(feature_values) < window_len:
        raise ValueError("Not enough rows to form one sample window.")

    samples = []
    targets = []
    rows = []

    sample_id = 0
    for label_row in range(window_len - 1, len(feature_values), sample_stride):
        row_start = label_row - window_len + 1
        row_end = label_row
        window = feature_values[row_start : row_end + 1]

        samples.append(window)
        targets.append(label_values[label_row])
        rows.append(
            {
                "sample_id": sample_id,
                "row_start": row_start,
                "row_end": row_end,
                "label_row": label_row,
            }
        )
        sample_id += 1

    X = np.stack(samples, axis=0)
    y = np.asarray(targets, dtype=np.int64)
    sample_table = pd.DataFrame(rows)

    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ValueError("max_samples must be positive.")
        X = X[:max_samples]
        y = y[:max_samples]
        sample_table = sample_table.iloc[:max_samples].copy()

    return X, y, sample_table


def _enforce_non_overlap_boundary(
    sample_table: pd.DataFrame,
    left_end_idx: int,
    right_start_idx: int,
) -> int:
    """Shift right start index until windows no longer overlap with left end."""
    n = len(sample_table)
    if right_start_idx >= n:
        return right_start_idx
    left_row_end = int(sample_table.iloc[left_end_idx]["row_end"])
    while right_start_idx < n and int(sample_table.iloc[right_start_idx]["row_start"]) <= left_row_end:
        right_start_idx += 1
    return right_start_idx


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    sample_table: pd.DataFrame,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    window_len: int | None = None,
    label_col: str | None = None,
    sample_stride: int | None = None,
) -> SubsetData:
    """Split samples chronologically by label order with boundary overlap purge."""
    if not np.isclose(sum(split_ratio), 1.0):
        raise ValueError(f"split_ratio must sum to 1.0, got {split_ratio}")

    n = len(sample_table)
    if n < 3:
        raise ValueError("At least 3 samples are required for train/val/test split.")

    n_train_target = int(n * split_ratio[0])
    n_val_target = int(n * split_ratio[1])
    n_train_target = max(n_train_target, 1)
    n_val_target = max(n_val_target, 1)

    train_end = n_train_target - 1
    val_start = train_end + 1
    val_start_purged = _enforce_non_overlap_boundary(sample_table, train_end, val_start)

    val_end = val_start_purged + n_val_target - 1
    if val_end >= n - 1:
        val_end = n - 2
    if val_end < val_start_purged:
        raise ValueError("Validation split became empty after boundary purge.")

    test_start = val_end + 1
    test_start_purged = _enforce_non_overlap_boundary(sample_table, val_end, test_start)

    if test_start_purged >= n:
        raise ValueError("Test split became empty after boundary purge.")

    split_col = []
    dropped_boundary_sample_ids = []

    for idx, row in sample_table.iterrows():
        sid = int(row["sample_id"])
        if idx <= train_end:
            split_col.append("train")
        elif val_start <= idx < val_start_purged:
            split_col.append("drop_boundary")
            dropped_boundary_sample_ids.append(sid)
        elif val_start_purged <= idx <= val_end:
            split_col.append("val")
        elif test_start <= idx < test_start_purged:
            split_col.append("drop_boundary")
            dropped_boundary_sample_ids.append(sid)
        else:
            split_col.append("test")

    with_split = sample_table.copy()
    with_split["split"] = split_col

    keep_mask = with_split["split"].isin(["train", "val", "test"])
    kept = with_split.loc[keep_mask].reset_index(drop=True)

    keep_indices = keep_mask.to_numpy().nonzero()[0]
    X_kept = X[keep_indices]
    y_kept = y[keep_indices]

    # Re-assign sample_id to contiguous ids for split integrity checks.
    kept["original_sample_id"] = kept["sample_id"]
    kept["sample_id"] = np.arange(len(kept), dtype=np.int64)

    metadata = {
        "split_ratio_requested": list(split_ratio),
        "sample_stride": None if sample_stride is None else int(sample_stride),
        "window_len": None if window_len is None else int(window_len),
        "label_col": label_col,
        "target_counts": {
            "train": n_train_target,
            "val": n_val_target,
            "test": n - n_train_target - n_val_target,
        },
        "actual_counts": {
            "train": int((kept["split"] == "train").sum()),
            "val": int((kept["split"] == "val").sum()),
            "test": int((kept["split"] == "test").sum()),
        },
        "dropped_boundary_sample_count": len(dropped_boundary_sample_ids),
        "dropped_boundary_sample_ids_head": dropped_boundary_sample_ids[:20],
        "boundary_purge_applied": len(dropped_boundary_sample_ids) > 0,
        "split_ranges": {
            split: {
                "sample_count": int((kept["split"] == split).sum()),
                "label_row_min": int(kept.loc[kept["split"] == split, "label_row"].min()),
                "label_row_max": int(kept.loc[kept["split"] == split, "label_row"].max()),
                "row_start_min": int(kept.loc[kept["split"] == split, "row_start"].min()),
                "row_end_max": int(kept.loc[kept["split"] == split, "row_end"].max()),
            }
            for split in ["train", "val", "test"]
        },
    }

    return SubsetData(X=X_kept, y=y_kept, sample_table=kept, metadata=metadata)
