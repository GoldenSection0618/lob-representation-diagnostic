"""Step 3 entrypoint: build a minimal chronological LOB subset from external CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.checks import (
    chronological_split_check,
    feature_contract_check,
    label_contract_check,
    output_safety_check,
    window_alignment_check,
)
from src.data.labeling import DEFAULT_HORIZONS, generate_lobench_labels
from src.data.load_lobench import CANONICAL_FEATURE_COLUMNS, load_external_processed_csv
from src.data.make_subset import build_sliding_windows, chronological_split


def parse_split_ratio(text: str) -> Tuple[float, float, float]:
    normalized = text.replace("/", ",")
    parts = [p.strip() for p in normalized.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("split ratio must have 3 parts, e.g., '70/15/15' or '0.7,0.15,0.15'.")

    values = [float(p) for p in parts]
    total = sum(values)
    if total > 1.5:
        values = [v / 100.0 for v in values]
    s = sum(values)
    if abs(s - 1.0) > 1e-9:
        raise ValueError(f"split ratio must sum to 1.0, got {values}.")
    return float(values[0]), float(values[1]), float(values[2])


def summarize_split_ranges(sample_table) -> Dict[str, Dict[str, int]]:
    out = {}
    for split in ["train", "val", "test"]:
        part = sample_table[sample_table["split"] == split]
        out[split] = {
            "sample_count": int(len(part)),
            "sample_id_min": int(part["sample_id"].min()),
            "sample_id_max": int(part["sample_id"].max()),
            "label_row_min": int(part["label_row"].min()),
            "label_row_max": int(part["label_row"].max()),
            "row_start_min": int(part["row_start"].min()),
            "row_end_max": int(part["row_end"].max()),
        }
    return out


def write_outputs(
    output_dir: Path,
    subset,
    metadata: Dict[str, object],
    dry_run: bool,
) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_files = [str(output_dir / "metadata.json")]

    if not dry_run:
        np.save(output_dir / "X.npy", subset.X)
        np.save(output_dir / "y.npy", subset.y)
        subset.sample_table.to_csv(output_dir / "samples.csv", index=False)
        generated_files.extend(
            [
                str(output_dir / "X.npy"),
                str(output_dir / "y.npy"),
                str(output_dir / "samples.csv"),
            ]
        )

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=True)

    return generated_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Build chronological minimal subset from external LOB CSV.")
    parser.add_argument("--input-csv", required=True, help="Path to external processed CSV file.")
    parser.add_argument("--symbol", required=True, help="Instrument symbol label for metadata.")
    parser.add_argument(
        "--output-dir",
        default="data/processed/minimal_subset",
        help="Output directory for generated subset files.",
    )
    parser.add_argument("--window-len", type=int, default=100)
    parser.add_argument("--label-horizon", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.0001)
    parser.add_argument("--split-ratio", default="70/15/15")
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    split_ratio = parse_split_ratio(args.split_ratio)
    label_col = f"trend{args.label_horizon}"

    loaded = load_external_processed_csv(
        input_csv=args.input_csv,
        window_len=args.window_len,
        row_limit=args.row_limit,
    )

    feature_check = feature_contract_check(loaded.features.columns, CANONICAL_FEATURE_COLUMNS)
    if not feature_check["passed"]:
        raise RuntimeError(f"feature_contract_check failed: {feature_check['details']}")

    labeled = generate_lobench_labels(
        features=loaded.features,
        threshold=args.threshold,
        horizons=DEFAULT_HORIZONS,
        window_len=args.window_len,
    )

    label_check = label_contract_check(labeled.labels, label_col=label_col, window_len=args.window_len)
    if not label_check["passed"]:
        raise RuntimeError(f"label_contract_check failed: {label_check['details']}")

    X, y, sample_table = build_sliding_windows(
        features=labeled.features,
        labels=labeled.labels,
        window_len=args.window_len,
        label_col=label_col,
        max_samples=args.max_samples,
    )

    subset = chronological_split(X=X, y=y, sample_table=sample_table, split_ratio=split_ratio)

    align_check = window_alignment_check(
        subset.X,
        subset.y,
        subset.sample_table,
        window_len=args.window_len,
        feature_dim=40,
    )
    if not align_check["passed"]:
        raise RuntimeError(f"window_alignment_check failed: {align_check['details']}")

    chrono_check = chronological_split_check(subset.sample_table, split_ratio=split_ratio)
    if not chrono_check["passed"]:
        raise RuntimeError(f"chronological_split_check failed: {chrono_check['details']}")

    repo_root = Path(__file__).resolve().parents[1]
    output_dir = (repo_root / args.output_dir).resolve()

    metadata = {
        "step": "step3_minimal_subset",
        "symbol": args.symbol,
        "input_csv": str(Path(args.input_csv).expanduser().resolve()),
        "output_dir": str(output_dir),
        "dry_run": bool(args.dry_run),
        "window_len": int(args.window_len),
        "label_horizon": int(args.label_horizon),
        "label_col": label_col,
        "threshold": float(args.threshold),
        "split_ratio": list(split_ratio),
        "row_limit": args.row_limit,
        "max_samples": args.max_samples,
        "load_metadata": loaded.metadata,
        "label_metadata": labeled.metadata,
        "subset_metadata": subset.metadata,
        "checks": {
            "feature_contract_check": feature_check,
            "label_contract_check": label_check,
            "window_alignment_check": align_check,
            "chronological_split_check": chrono_check,
        },
        "summary": {
            "raw_rows": loaded.metadata["raw_row_count"],
            "usable_rows_after_label_trim": labeled.metadata["rows_after_label_trim"],
            "total_samples": int(len(subset.y)),
            "X_shape": list(subset.X.shape),
            "y_shape": list(subset.y.shape),
            "class_distribution_label_col": label_check["details"]["class_counts"],
            "split_ranges": summarize_split_ranges(subset.sample_table),
        },
    }

    generated_files = write_outputs(output_dir, subset=subset, metadata=metadata, dry_run=args.dry_run)

    safety_check = output_safety_check(
        repo_root=str(repo_root),
        output_dir=str(output_dir),
        generated_files=generated_files,
    )
    metadata["checks"]["output_safety_check"] = safety_check
    if not safety_check["passed"]:
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=True)
        raise RuntimeError(f"output_safety_check failed: {safety_check['details']}")

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=True)

    split_ranges = metadata["summary"]["split_ranges"]

    print("=== Step 3 Minimal Subset Summary ===")
    print(f"symbol: {args.symbol}")
    print(f"input path: {metadata['input_csv']}")
    print(f"total raw rows: {metadata['summary']['raw_rows']}")
    print(f"usable rows after label trimming: {metadata['summary']['usable_rows_after_label_trim']}")
    print(f"total samples: {metadata['summary']['total_samples']}")
    print(f"X shape: {tuple(metadata['summary']['X_shape'])}")
    print(f"y shape: {tuple(metadata['summary']['y_shape'])}")
    print(f"label horizon: {args.label_horizon}")
    print(f"class distribution ({label_col}): {metadata['summary']['class_distribution_label_col']}")
    print(
        "train / val / test sizes: "
        f"{split_ranges['train']['sample_count']} / "
        f"{split_ranges['val']['sample_count']} / "
        f"{split_ranges['test']['sample_count']}"
    )
    print(
        "train / val / test row ranges: "
        f"train(label_row {split_ranges['train']['label_row_min']}..{split_ranges['train']['label_row_max']}), "
        f"val(label_row {split_ranges['val']['label_row_min']}..{split_ranges['val']['label_row_max']}), "
        f"test(label_row {split_ranges['test']['label_row_min']}..{split_ranges['test']['label_row_max']})"
    )
    print(f"chronological check status: {chrono_check['passed']}")
    print(f"output path: {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
