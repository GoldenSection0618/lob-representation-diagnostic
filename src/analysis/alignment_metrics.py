"""Metric helpers for Step 7 reconstruction-prediction alignment."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

CLASS_ORDER = [0, 1, 2]


def spearmanr_safe(x: Sequence[float], y: Sequence[float]) -> float:
    """Compute Spearman correlation with pandas ranks and safe degenerate handling."""
    xs = pd.Series(x, dtype="float64")
    ys = pd.Series(y, dtype="float64")
    valid = xs.notna() & ys.notna()
    if int(valid.sum()) < 3:
        return float("nan")
    xr = xs[valid].rank(method="average")
    yr = ys[valid].rank(method="average")
    if xr.nunique() < 2 or yr.nunique() < 2:
        return float("nan")
    return float(xr.corr(yr))


def point_biserial_safe(values: Sequence[float], binary: Sequence[float]) -> float:
    xs = pd.Series(values, dtype="float64")
    ys = pd.Series(binary, dtype="float64")
    valid = xs.notna() & ys.notna()
    if int(valid.sum()) < 3:
        return float("nan")
    yv = ys[valid]
    if yv.nunique() != 2:
        return float("nan")
    xv = xs[valid]
    if xv.nunique() < 2:
        return float("nan")
    return float(xv.corr(yv))


def auroc_safe(score: Sequence[float], binary_target: Sequence[float]) -> float:
    xs = pd.Series(score, dtype="float64")
    ys = pd.Series(binary_target, dtype="float64")
    valid = xs.notna() & ys.notna()
    if int(valid.sum()) < 3:
        return float("nan")
    yv = ys[valid].astype(int)
    if yv.nunique() != 2:
        return float("nan")
    return float(roc_auc_score(yv.to_numpy(), xs[valid].to_numpy()))


def cliffs_delta(
    failure_values: Sequence[float],
    reference_values: Sequence[float],
    max_per_group: int = 5000,
    seed: int = 42,
) -> float:
    """Compute Cliff's delta, using deterministic capped samples for large groups."""
    a = pd.Series(failure_values, dtype="float64").dropna().to_numpy()
    b = pd.Series(reference_values, dtype="float64").dropna().to_numpy()
    if len(a) == 0 or len(b) == 0:
        return float("nan")

    rng = np.random.default_rng(seed)
    if len(a) > max_per_group:
        a = a[np.sort(rng.choice(len(a), size=max_per_group, replace=False))]
    if len(b) > max_per_group:
        b = b[np.sort(rng.choice(len(b), size=max_per_group, replace=False))]

    b_sorted = np.sort(b)
    greater = np.searchsorted(b_sorted, a, side="left").sum()
    less_or_equal = np.searchsorted(b_sorted, a, side="right")
    less = (len(b_sorted) - less_or_equal).sum()
    return float((greater - less) / (len(a) * len(b_sorted)))


def multiclass_bin_metrics(df: pd.DataFrame) -> Dict[str, float | int]:
    y_true = df["y_true"].astype(int).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()
    if len(df) == 0:
        return {
            "n_samples": 0,
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "balanced_accuracy": float("nan"),
            "mcc": float("nan"),
            "mean_confidence": float("nan"),
            "mean_proba_true": float("nan"),
            "opposite_direction_rate": float("nan"),
            "directional_accuracy_non_neutral": float("nan"),
            "true_down_count": 0,
            "true_neutral_count": 0,
            "true_up_count": 0,
        }

    non_neutral = np.isin(y_true, [0, 2])
    if non_neutral.any():
        directional_accuracy = float((y_true[non_neutral] == y_pred[non_neutral]).mean())
        opposite_rate = float(df.loc[non_neutral, "opposite_direction_error"].astype(bool).mean())
    else:
        directional_accuracy = float("nan")
        opposite_rate = float("nan")

    return {
        "n_samples": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "mean_confidence": float(df["confidence"].mean()),
        "mean_proba_true": float(df["proba_true"].mean()),
        "opposite_direction_rate": opposite_rate,
        "directional_accuracy_non_neutral": directional_accuracy,
        "true_down_count": int((y_true == 0).sum()),
        "true_neutral_count": int((y_true == 1).sum()),
        "true_up_count": int((y_true == 2).sum()),
    }


def assign_quantile_bins(values: pd.Series, n_bins: int = 4) -> pd.Series:
    """Assign Q1..Qn bins, falling back to ranked values when duplicate edges occur."""
    labels = [f"Q{i}" for i in range(1, n_bins + 1)]
    valid = values.notna()
    out = pd.Series(pd.NA, index=values.index, dtype="object")
    if int(valid.sum()) == 0:
        return out
    try:
        out.loc[valid] = pd.qcut(values.loc[valid], q=n_bins, labels=labels, duplicates="drop").astype(str)
    except ValueError:
        ranks = values.loc[valid].rank(method="first")
        out.loc[valid] = pd.qcut(ranks, q=n_bins, labels=labels, duplicates="drop").astype(str)
    return out


def class_aligned_proba(raw_proba: np.ndarray, classes: Iterable[int], class_order: Sequence[int] = CLASS_ORDER) -> np.ndarray:
    aligned = np.zeros((raw_proba.shape[0], len(class_order)), dtype=np.float64)
    class_to_dst = {int(c): i for i, c in enumerate(class_order)}
    for src_idx, cls in enumerate(classes):
        if int(cls) in class_to_dst:
            aligned[:, class_to_dst[int(cls)]] = raw_proba[:, src_idx]
    row_sum = aligned.sum(axis=1, keepdims=True)
    return np.divide(
        aligned,
        row_sum,
        out=np.full_like(aligned, 1.0 / len(class_order)),
        where=row_sum > 0,
    )
