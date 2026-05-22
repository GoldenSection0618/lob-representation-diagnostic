"""Integrity checks for Step 3 chronological subset construction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _result(name: str, passed: bool, details: Dict[str, object]) -> Dict[str, object]:
    return {"name": name, "passed": bool(passed), "details": details}


def feature_contract_check(
    feature_columns: Sequence[str],
    canonical_order: Sequence[str],
) -> Dict[str, object]:
    feature_columns = list(feature_columns)
    canonical_order = list(canonical_order)

    missing = [c for c in canonical_order if c not in feature_columns]
    duplicates = sorted({c for c in feature_columns if feature_columns.count(c) > 1})
    passed = (
        len(feature_columns) == 40
        and feature_columns == canonical_order
        and not missing
        and not duplicates
    )
    details = {
        "feature_count": len(feature_columns),
        "expected_count": 40,
        "feature_order_matches": feature_columns == canonical_order,
        "missing_features": missing,
        "duplicate_features": duplicates,
    }
    return _result("feature_contract_check", passed, details)


def label_contract_check(
    labels: pd.DataFrame,
    label_col: str,
    window_len: int,
) -> Dict[str, object]:
    label_exists = label_col in labels.columns
    label_values = set(labels[label_col].astype(int).unique().tolist()) if label_exists else set()
    class_counts = (
        {str(int(k)): int(v) for k, v in labels[label_col].value_counts().sort_index().to_dict().items()}
        if label_exists else {}
    )
    usable_rows = len(labels)
    passed = label_exists and label_values.issubset({0, 1, 2}) and usable_rows > window_len

    details = {
        "label_col": label_col,
        "label_exists": label_exists,
        "label_values": sorted(label_values),
        "class_counts": class_counts,
        "usable_rows": usable_rows,
        "window_len": window_len,
    }
    return _result("label_contract_check", passed, details)


def window_alignment_check(
    X: np.ndarray,
    y: np.ndarray,
    sample_table: pd.DataFrame,
    window_len: int = 100,
    feature_dim: int = 40,
) -> Dict[str, object]:
    checks = {}
    checks["shape_matches"] = tuple(X.shape[1:]) == (window_len, feature_dim)
    checks["len_matches"] = len(X) == len(y) == len(sample_table)

    sample_ids = sample_table["sample_id"].to_numpy()
    checks["sample_id_strictly_increasing"] = bool(np.all(np.diff(sample_ids) > 0)) if len(sample_ids) > 1 else True

    row_start = sample_table["row_start"].to_numpy()
    row_end = sample_table["row_end"].to_numpy()
    label_row = sample_table["label_row"].to_numpy()

    checks["index_relation"] = bool(np.all((row_start < row_end) & (row_end <= label_row)))
    checks["label_not_earlier_than_window_end"] = bool(np.all(label_row >= row_end))

    checks["row_start_monotonic"] = bool(np.all(np.diff(row_start) > 0)) if len(row_start) > 1 else True
    checks["row_end_monotonic"] = bool(np.all(np.diff(row_end) > 0)) if len(row_end) > 1 else True
    checks["label_row_monotonic"] = bool(np.all(np.diff(label_row) > 0)) if len(label_row) > 1 else True

    passed = all(checks.values())
    details = {**checks, "X_shape": tuple(X.shape), "y_shape": tuple(y.shape), "num_samples": int(len(y))}
    return _result("window_alignment_check", passed, details)


def chronological_split_check(
    sample_table: pd.DataFrame,
    split_ratio: Tuple[float, float, float],
) -> Dict[str, object]:
    required_splits = ["train", "val", "test"]
    subsets = {k: sample_table[sample_table["split"] == k].copy() for k in required_splits}

    for name, subset in subsets.items():
        if subset.empty:
            return _result(
                "chronological_split_check",
                False,
                {"error": f"Split '{name}' is empty.", "split_sizes": {k: len(v) for k, v in subsets.items()}},
            )

    train, val, test = subsets["train"], subsets["val"], subsets["test"]

    checks = {
        "label_row_order_train_val": int(train["label_row"].max()) < int(val["label_row"].min()),
        "label_row_order_val_test": int(val["label_row"].max()) < int(test["label_row"].min()),
    }

    ranges = {}
    for name, subset in subsets.items():
        sid = subset["sample_id"].to_numpy()
        contiguous = bool(np.all(np.diff(sid) == 1)) if len(sid) > 1 else True
        checks[f"sample_id_contiguous_{name}"] = contiguous
        ranges[name] = {"sample_id_min": int(sid.min()), "sample_id_max": int(sid.max())}

    sid_sets = {k: set(v["sample_id"].tolist()) for k, v in subsets.items()}
    checks["no_sample_id_overlap"] = (
        sid_sets["train"].isdisjoint(sid_sets["val"])
        and sid_sets["train"].isdisjoint(sid_sets["test"])
        and sid_sets["val"].isdisjoint(sid_sets["test"])
    )

    lr_sets = {k: set(v["label_row"].tolist()) for k, v in subsets.items()}
    checks["no_label_row_overlap"] = (
        lr_sets["train"].isdisjoint(lr_sets["val"])
        and lr_sets["train"].isdisjoint(lr_sets["test"])
        and lr_sets["val"].isdisjoint(lr_sets["test"])
    )

    all_sorted = sample_table.sort_values("sample_id").reset_index(drop=True)
    checks["no_shuffle_signature"] = bool(np.all(np.diff(all_sorted["label_row"].to_numpy()) > 0))

    total = len(sample_table)
    expected = {
        "train": split_ratio[0] * total,
        "val": split_ratio[1] * total,
        "test": split_ratio[2] * total,
    }
    actual = {k: len(v) for k, v in subsets.items()}
    approx_ok = True
    tolerance = max(2, int(0.05 * total))
    for k in required_splits:
        if total > 0:
            if abs(actual[k] - expected[k]) > tolerance:
                approx_ok = False
    checks["split_size_approx_match"] = approx_ok

    overlap_train_val = int(train["row_end"].max()) >= int(val["row_start"].min())
    overlap_val_test = int(val["row_end"].max()) >= int(test["row_start"].min())
    checks["no_window_overlap_train_val"] = not overlap_train_val
    checks["no_window_overlap_val_test"] = not overlap_val_test

    details = {
        **checks,
        "split_sizes": actual,
        "expected_split_sizes_float": expected,
        "sample_id_ranges": ranges,
        "boundary_window_overlap_detected": {
            "train_val": overlap_train_val,
            "val_test": overlap_val_test,
        },
    }

    passed = all(checks.values())
    return _result("chronological_split_check", passed, details)


def output_safety_check(
    repo_root: str,
    output_dir: str,
    generated_files: Sequence[str],
) -> Dict[str, object]:
    repo = Path(repo_root).resolve()
    out = Path(output_dir).resolve()

    try:
        rel = out.relative_to(repo)
        in_repo = True
    except ValueError:
        rel = None
        in_repo = False

    expected_prefix = Path("data/processed/minimal_subset")
    in_expected_output = in_repo and str(rel).startswith(str(expected_prefix))

    gitignore = (repo / ".gitignore").read_text(encoding="utf-8")
    required_patterns = ["data/", "data/processed/", "*.csv", "*.npz", "*.npy", "*.pt", "*.parquet"]
    missing_patterns = [p for p in required_patterns if p not in gitignore]

    tracked_generated = []
    for file_path in generated_files:
        p = Path(file_path).resolve()
        if not p.exists():
            continue
        try:
            rel_file = p.relative_to(repo)
        except ValueError:
            continue
        # If file is tracked by git, this command returns 0.
        from subprocess import run

        res = run(
            ["git", "ls-files", "--error-unmatch", str(rel_file)],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            tracked_generated.append(str(rel_file))

    passed = in_expected_output and not missing_patterns and not tracked_generated
    details = {
        "output_dir": str(out),
        "output_dir_in_repo": in_repo,
        "output_dir_under_expected_prefix": in_expected_output,
        "missing_gitignore_patterns": missing_patterns,
        "tracked_generated_files": tracked_generated,
    }
    return _result("output_safety_check", passed, details)
