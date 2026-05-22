"""Mid-price and trend label construction for LOBench-style contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_HORIZONS: Tuple[int, ...] = (1, 3, 5, 7, 10)


@dataclass
class LabeledData:
    """Container for aligned features/labels and label metadata."""

    features: pd.DataFrame
    labels: pd.DataFrame
    metadata: Dict[str, object]


def _trend_from_midprice(midprice: pd.Series, horizon: int, threshold: float) -> pd.Series:
    gap = midprice.rolling(window=horizon).mean().shift(-horizon) - midprice
    trend = np.where(gap > threshold, 2, np.where(gap < -threshold, 0, 1))
    trend_series = pd.Series(trend, index=midprice.index, name=f"trend{horizon}")
    trend_series[gap.isna()] = np.nan
    return trend_series


def generate_lobench_labels(
    features: pd.DataFrame,
    threshold: float = 0.0001,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    window_len: int = 100,
) -> LabeledData:
    """Generate LOBench-style labels and trim tail rows with future-label NaNs."""
    horizons = tuple(sorted(set(int(h) for h in horizons)))
    for required in DEFAULT_HORIZONS:
        if required not in horizons:
            horizons = tuple(sorted(set(horizons + (required,))))

    labels = pd.DataFrame(index=features.index)
    labels["midPrice"] = (features["bestBidPrice1"] + features["bestAskPrice1"]) / 2.0
    labels["spread"] = features["bestAskPrice1"] - features["bestBidPrice1"]

    class_distribution: Dict[str, Dict[str, int]] = {}
    for horizon in horizons:
        col = f"trend{horizon}"
        labels[col] = _trend_from_midprice(labels["midPrice"], horizon=horizon, threshold=threshold)

    required_label_cols = ["midPrice", "spread", "trend1", "trend3", "trend5", "trend7", "trend10"]
    labels = labels[required_label_cols]

    rows_before = int(len(labels))
    valid_mask = labels.notna().all(axis=1)
    trimmed_features = features.loc[valid_mask].reset_index(drop=True)
    trimmed_labels = labels.loc[valid_mask].reset_index(drop=True)
    rows_after = int(len(trimmed_labels))
    rows_trimmed = rows_before - rows_after

    spread_non_negative_ratio = float((trimmed_labels["spread"] >= 0).mean()) if rows_after else 0.0
    if spread_non_negative_ratio < 0.95:
        raise ValueError(
            f"Spread non-negative ratio too low: {spread_non_negative_ratio:.4f} < 0.95"
        )

    for col in ["trend1", "trend3", "trend5", "trend7", "trend10"]:
        unique_vals = set(trimmed_labels[col].astype(int).unique().tolist())
        if not unique_vals.issubset({0, 1, 2}):
            raise ValueError(f"Label column {col} has invalid class values: {sorted(unique_vals)}")
        counts = trimmed_labels[col].value_counts().sort_index().to_dict()
        class_distribution[col] = {str(int(k)): int(v) for k, v in counts.items()}

    if rows_after <= window_len:
        raise ValueError(
            f"Usable rows after label trimming ({rows_after}) must be > window_len ({window_len})."
        )

    metadata: Dict[str, object] = {
        "label_threshold": float(threshold),
        "label_horizons": list(DEFAULT_HORIZONS),
        "rows_before_label_trim": rows_before,
        "rows_after_label_trim": rows_after,
        "rows_trimmed_by_future_label": rows_trimmed,
        "spread_non_negative_ratio": spread_non_negative_ratio,
        "class_distribution": class_distribution,
    }

    return LabeledData(features=trimmed_features, labels=trimmed_labels, metadata=metadata)
