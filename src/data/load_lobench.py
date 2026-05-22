"""Utilities for loading external LOBench-style processed LOB CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


CANONICAL_FEATURE_COLUMNS: List[str] = [
    "bestBidPrice10", "bestBidPrice9", "bestBidPrice8", "bestBidPrice7", "bestBidPrice6",
    "bestBidPrice5", "bestBidPrice4", "bestBidPrice3", "bestBidPrice2", "bestBidPrice1",
    "bestAskPrice1", "bestAskPrice2", "bestAskPrice3", "bestAskPrice4", "bestAskPrice5",
    "bestAskPrice6", "bestAskPrice7", "bestAskPrice8", "bestAskPrice9", "bestAskPrice10",
    "bestBidVolume10", "bestBidVolume9", "bestBidVolume8", "bestBidVolume7", "bestBidVolume6",
    "bestBidVolume5", "bestBidVolume4", "bestBidVolume3", "bestBidVolume2", "bestBidVolume1",
    "bestAskVolume1", "bestAskVolume2", "bestAskVolume3", "bestAskVolume4", "bestAskVolume5",
    "bestAskVolume6", "bestAskVolume7", "bestAskVolume8", "bestAskVolume9", "bestAskVolume10",
]


def _alt_name_for_canonical(canonical_name: str) -> str:
    if canonical_name.startswith("best"):
        return canonical_name[4:]
    return canonical_name


def canonical_name_mapping(columns: List[str]) -> Tuple[Dict[str, str], str]:
    """Build a source->canonical rename map and return naming mode."""
    column_set = set(columns)
    rename_map: Dict[str, str] = {}
    used_sources = set()

    for canonical in CANONICAL_FEATURE_COLUMNS:
        if canonical in column_set:
            source = canonical
        else:
            alt = _alt_name_for_canonical(canonical)
            if alt in column_set:
                source = alt
            else:
                raise ValueError(
                    f"Missing required feature column '{canonical}' (or alternate '{alt}')."
                )

        if source in used_sources:
            raise ValueError(f"Duplicate source feature mapping detected for '{source}'.")
        rename_map[source] = canonical
        used_sources.add(source)

    if all(src == dst for src, dst in rename_map.items()):
        mode = "canonical_best"
    elif all(not src.startswith("best") for src in rename_map):
        mode = "ashare_no_best"
    else:
        mode = "mixed"
    return rename_map, mode


def _detect_time_column(df: pd.DataFrame) -> Tuple[Optional[str], Dict[str, Optional[str]]]:
    candidates = [
        "index", "timestamp", "datetime", "date", "time", "trading_time", "event_time",
    ]
    lower_to_original = {c.lower(): c for c in df.columns}

    time_col = None
    for key in candidates:
        if key in lower_to_original:
            time_col = lower_to_original[key]
            break

    if time_col is None:
        return None, {
            "chronological_proxy": "row_index",
            "time_col": None,
            "time_start": None,
            "time_end": None,
        }

    series = pd.to_datetime(df[time_col], errors="coerce")
    if series.notna().any():
        time_start = str(series.dropna().iloc[0])
        time_end = str(series.dropna().iloc[-1])
    else:
        time_start = None
        time_end = None

    return time_col, {
        "chronological_proxy": "time_column" if time_start and time_end else "row_index",
        "time_col": time_col,
        "time_start": time_start,
        "time_end": time_end,
    }


@dataclass
class LoadedLOBData:
    """Container for canonical feature frame and loading metadata."""

    features: pd.DataFrame
    metadata: Dict[str, object]


def load_external_processed_csv(
    input_csv: str,
    window_len: int = 100,
    row_limit: Optional[int] = None,
) -> LoadedLOBData:
    """Load processed LOB CSV and enforce canonical 40-feature contract."""
    csv_path = Path(input_csv).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, nrows=row_limit)
    if df.empty:
        raise ValueError("Loaded CSV is empty.")

    original_columns = list(df.columns)
    duplicate_columns = [c for c in original_columns if original_columns.count(c) > 1]
    if duplicate_columns:
        raise ValueError(f"Duplicate columns found in input CSV: {sorted(set(duplicate_columns))}")

    rename_map, naming_mode = canonical_name_mapping(original_columns)
    df = df.rename(columns=rename_map)

    feature_columns = CANONICAL_FEATURE_COLUMNS.copy()
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing canonical feature columns after mapping: {missing}")

    features = df[feature_columns].copy()
    if list(features.columns) != feature_columns:
        raise ValueError("Feature order does not match canonical order.")

    for col in feature_columns:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    nan_count = int(features.isna().sum().sum())
    inf_count = int(np.isinf(features.to_numpy()).sum())
    if nan_count > 0 or inf_count > 0:
        raise ValueError(
            f"Invalid feature values detected: nan_count={nan_count}, inf_count={inf_count}."
        )

    if len(features) < window_len:
        raise ValueError(
            f"Insufficient rows for windowing: rows={len(features)}, window_len={window_len}."
        )

    time_col, time_meta = _detect_time_column(df)

    metadata: Dict[str, object] = {
        "input_csv": str(csv_path),
        "raw_row_count": int(len(df)),
        "feature_row_count": int(len(features)),
        "feature_count": int(features.shape[1]),
        "feature_order": feature_columns,
        "feature_naming_mode": naming_mode,
        "feature_mapping": dict(rename_map),
        "time_column": time_col,
        "time_info": time_meta,
    }
    return LoadedLOBData(features=features, metadata=metadata)
