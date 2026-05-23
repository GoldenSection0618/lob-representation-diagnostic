"""Step 6 entrypoint: train and evaluate reconstruction-only baselines."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.reconstruction_metrics import (
    BEST_ASK_PRICE1_IDX,
    BEST_BID_PRICE1_IDX,
    IMBALANCE_EPS_THRESHOLD,
    IMBALANCE_VALID_RATIO_THRESHOLD,
    WINDOW_LEN,
    class_distribution,
    compute_derived_lob_errors,
    compute_feature_group_errors,
    compute_level_wise_errors,
    compute_per_sample_errors,
    compute_primary_metrics,
    compute_temporal_errors,
    validate_feature_contract,
)
from src.data.prediction_dataset import load_prediction_arrays
from src.models.reconstruction_baselines import (
    MLPAEConfig,
    MLPAutoencoderReconstructor,
    PCAReconstructor,
    LastSnapshotRepeatReconstructor,
    TrainMeanWindowReconstructor,
    compute_compression_ratio,
    count_parameters,
    flatten_windows,
    unflatten_windows,
)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _safe_label(model: str, latent_dim: int | None) -> str:
    if latent_dim is None:
        return model
    return f"{model}@{latent_dim}"


def _model_variant(model: str, latent_dim: int | None) -> str:
    if latent_dim is None:
        return model
    return f"{model}@{int(latent_dim)}"


def _select_best_model_row(test_rows: pd.DataFrame) -> pd.Series:
    candidates = test_rows.copy()
    candidates = candidates.sort_values(
        by=["normalized_mse", "compression_ratio_for_rank", "original_mae"], ascending=[True, True, True]
    )
    return candidates.iloc[0]


def _plot_rate_distortion(rate_df: pd.DataFrame, metrics_df: pd.DataFrame, fig_path: Path) -> None:
    test_rate = rate_df[rate_df["split"] == "test"].copy()

    plt.figure(figsize=(9, 5))
    for model in ["pca", "mlp_ae"]:
        part = test_rate[test_rate["model"] == model].copy().sort_values("latent_dim")
        if part.empty:
            continue
        plt.plot(part["latent_dim"], part["normalized_mse"], marker="o", label=model)

    ls_row = metrics_df[(metrics_df["model"] == "last_snapshot_repeat") & (metrics_df["split"] == "test")]
    if not ls_row.empty:
        x = float(ls_row.iloc[0]["latent_dim"])
        y = float(ls_row.iloc[0]["normalized_mse"])
        plt.scatter([x], [y], marker="x", s=80, label="last_snapshot_repeat")

    plt.xlabel("latent_dim")
    plt.ylabel("test normalized_mse (lower is better)")
    plt.title("Rate-Distortion Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def _plot_scorecard(metrics_df: pd.DataFrame, fig_path: Path) -> None:
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    test_df["label"] = test_df.apply(lambda r: _safe_label(str(r["model"]), None if pd.isna(r["latent_dim"]) else int(r["latent_dim"])), axis=1)
    test_df = test_df.sort_values("normalized_mse", ascending=True)

    labels = test_df["label"].tolist()
    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(12, 5))
    plt.bar(x - width, test_df["normalized_mse"].to_numpy(), width=width, label="normalized_mse")
    plt.bar(x, test_df["normalized_mae"].to_numpy(), width=width, label="normalized_mae")
    plt.bar(x + width, test_df["relative_mse_vs_last_snapshot"].to_numpy(), width=width, label="relative_mse_vs_last_snapshot")

    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("score")
    plt.title("Reconstruction Scorecard by Model (Test)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def _plot_feature_group(feature_df: pd.DataFrame, derived_df: pd.DataFrame, selected: List[Tuple[str, int | None]], fig_path: Path) -> None:
    # Upper: normalized_mae for feature groups. Lower: original MAE for midprice/spread.
    selected_labels = [_safe_label(m, d) for m, d in selected]

    feature_groups = ["price", "volume", "top_of_book", "last_step"]
    bar_data = []
    for model, latent in selected:
        part = feature_df[
            (feature_df["split"] == "test")
            & (feature_df["model"] == model)
            & (feature_df["latent_dim"].isna() if latent is None else feature_df["latent_dim"] == latent)
            & (feature_df["group"].isin(feature_groups))
        ]
        row = {g: float("nan") for g in feature_groups}
        for _, r in part.iterrows():
            row[str(r["group"])] = float(r["normalized_mae"])
        bar_data.append(row)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    x = np.arange(len(feature_groups))
    width = 0.8 / max(len(selected_labels), 1)

    for i, row in enumerate(bar_data):
        vals = [row[g] for g in feature_groups]
        axes[0].bar(x - 0.4 + width / 2 + i * width, vals, width=width, label=selected_labels[i])

    axes[0].set_xticks(x, feature_groups)
    axes[0].set_ylabel("normalized_mae")
    axes[0].set_title("Feature-Group Error by Model (Test)")
    axes[0].legend(fontsize=8)

    derived_groups = ["midprice", "spread"]
    x2 = np.arange(len(derived_groups))
    for i, (model, latent) in enumerate(selected):
        part = derived_df[
            (derived_df["split"] == "test")
            & (derived_df["model"] == model)
            & (derived_df["latent_dim"].isna() if latent is None else derived_df["latent_dim"] == latent)
        ]
        if part.empty:
            vals = [float("nan"), float("nan")]
        else:
            r = part.iloc[0]
            vals = [float(r["midprice_mae"]), float(r["spread_mae"])]
        axes[1].bar(x2 - 0.4 + width / 2 + i * width, vals, width=width, label=selected_labels[i])

    axes[1].set_xticks(x2, derived_groups)
    axes[1].set_ylabel("original MAE")
    axes[1].set_title("Derived LOB Error (Test)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)


def _plot_derived_lob_error_by_model(
    derived_df: pd.DataFrame, selected: List[Tuple[str, int | None]], fig_path: Path
) -> None:
    selected_labels = [_safe_label(m, d) for m, d in selected]
    metrics = [
        "midprice_mae",
        "spread_mae",
        "top1_volume_sum_mae",
        "top5_volume_sum_mae",
        "top1_volume_diff_mae",
        "top5_volume_diff_mae",
    ]
    x = np.arange(len(metrics))
    width = 0.8 / max(len(selected_labels), 1)

    plt.figure(figsize=(13, 5))
    for i, (model, latent) in enumerate(selected):
        part = derived_df[
            (derived_df["split"] == "test")
            & (derived_df["model"] == model)
            & (derived_df["latent_dim"].isna() if latent is None else derived_df["latent_dim"] == latent)
        ]
        if part.empty:
            vals = [float("nan")] * len(metrics)
        else:
            r = part.iloc[0]
            vals = [float(r[m]) for m in metrics]
        plt.bar(x - 0.4 + width / 2 + i * width, vals, width=width, label=selected_labels[i])

    plt.xticks(x, metrics, rotation=20, ha="right")
    plt.ylabel("original MAE")
    plt.title("Derived LOB Error by Model (Test)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def _plot_level_heatmap(level_df: pd.DataFrame, best_model: str, best_latent: int | None, fig_path: Path) -> None:
    part = level_df[
        (level_df["split"] == "test")
        & (level_df["model"] == best_model)
        & (level_df["latent_dim"].isna() if best_latent is None else level_df["latent_dim"] == best_latent)
    ]

    cols = [("bid", "price"), ("ask", "price"), ("bid", "volume"), ("ask", "volume")]
    mat = np.zeros((10, 4), dtype=np.float64)
    for level in range(1, 11):
        for j, (side, field_type) in enumerate(cols):
            row = part[(part["level"] == level) & (part["side"] == side) & (part["field_type"] == field_type)].iloc[0]
            mat[level - 1, j] = float(row["normalized_mse"])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(4), ["bid_price", "ask_price", "bid_volume", "ask_volume"])
    ax.set_yticks(np.arange(10), [str(x) for x in range(1, 11)])
    ax.set_xlabel("field")
    ax.set_ylabel("level (1=top-of-book)")
    ax.set_title(f"Level-wise Error Heatmap (Best: {_safe_label(best_model, best_latent)})")

    for i in range(10):
        for j in range(4):
            ax.text(j, i, f"{mat[i, j]:.4f}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close(fig)


def _plot_temporal_profile(temporal_df: pd.DataFrame, model_specs: List[Tuple[str, int | None]], fig_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for model, latent in model_specs:
        part = temporal_df[
            (temporal_df["split"] == "test")
            & (temporal_df["model"] == model)
            & (temporal_df["latent_dim"].isna() if latent is None else temporal_df["latent_dim"] == latent)
        ].sort_values("timestep")
        if part.empty:
            continue
        label = _safe_label(model, latent)
        plt.plot(part["timestep"], part["normalized_mse"], label=label)

    plt.axvline(WINDOW_LEN - 1, linestyle="--", linewidth=1.0)
    plt.xlabel("timestep")
    plt.ylabel("test normalized_mse")
    plt.title("Temporal Error Profile")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def _model_eval(
    model_name: str,
    latent_dim: int | None,
    reconstructor,
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    sample_tables: Dict[str, pd.DataFrame],
    scaler: StandardScaler,
    scaled_flat: Dict[str, np.ndarray],
    baseline_mse_train_mean: Dict[str, float],
    baseline_mse_last_snapshot: Dict[str, float],
    compression_ratio: float | None,
    num_parameters: int,
    train_seconds: float,
) -> Dict[str, object]:
    variant = _model_variant(model_name, latent_dim)
    metrics_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    level_rows: List[Dict[str, object]] = []
    temporal_rows: List[Dict[str, object]] = []
    derived_rows: List[Dict[str, object]] = []
    per_sample_tables: List[pd.DataFrame] = []

    for split in ["train", "val", "test"]:
        X_orig, y_split = splits[split]
        X_true_scaled_flat = scaled_flat[split]
        X_true_scaled = unflatten_windows(X_true_scaled_flat, window_len=X_orig.shape[1], feature_dim=X_orig.shape[2])

        t0 = time.perf_counter()
        if model_name == "last_snapshot_repeat":
            X_hat_orig = reconstructor.reconstruct(X_orig)
            X_hat_scaled_flat = scaler.transform(flatten_windows(X_hat_orig))
        else:
            X_hat_scaled_flat = reconstructor.reconstruct(X_true_scaled_flat)
            X_hat_orig_flat = scaler.inverse_transform(X_hat_scaled_flat)
            X_hat_orig = unflatten_windows(X_hat_orig_flat, window_len=X_orig.shape[1], feature_dim=X_orig.shape[2])

        infer_seconds = time.perf_counter() - t0
        infer_ms_per_1000 = float((infer_seconds / max(len(X_orig), 1)) * 1000.0 * 1000.0)

        X_hat_scaled = unflatten_windows(X_hat_scaled_flat, window_len=X_orig.shape[1], feature_dim=X_orig.shape[2])

        primary = compute_primary_metrics(
            X_true_scaled_flat=X_true_scaled_flat,
            X_hat_scaled_flat=X_hat_scaled_flat,
            X_true_original=X_orig,
            X_hat_original=X_hat_orig,
            baseline_train_mean_mse=baseline_mse_train_mean[split],
            baseline_last_snapshot_mse=baseline_mse_last_snapshot[split],
            compression_ratio=compression_ratio,
            num_parameters=num_parameters,
            train_seconds=train_seconds,
            inference_ms_per_1000_samples=infer_ms_per_1000,
        )

        metrics_rows.append(
            {
                "model": model_name,
                "model_variant": variant,
                "latent_dim": latent_dim,
                "split": split,
                **primary,
            }
        )

        for row in compute_feature_group_errors(
            X_true_scaled=X_true_scaled,
            X_hat_scaled=X_hat_scaled,
            X_true_original=X_orig,
            X_hat_original=X_hat_orig,
        ):
            feature_rows.append(
                {"model": model_name, "model_variant": variant, "latent_dim": latent_dim, "split": split, **row}
            )

        for row in compute_level_wise_errors(
            X_true_scaled=X_true_scaled,
            X_hat_scaled=X_hat_scaled,
            X_true_original=X_orig,
            X_hat_original=X_hat_orig,
        ):
            level_rows.append(
                {"model": model_name, "model_variant": variant, "latent_dim": latent_dim, "split": split, **row}
            )

        for row in compute_temporal_errors(X_true_scaled=X_true_scaled, X_hat_scaled=X_hat_scaled):
            temporal_rows.append(
                {"model": model_name, "model_variant": variant, "latent_dim": latent_dim, "split": split, **row}
            )

        derived = compute_derived_lob_errors(X_true_original=X_orig, X_hat_original=X_hat_orig)
        derived_rows.append(
            {"model": model_name, "model_variant": variant, "latent_dim": latent_dim, "split": split, **derived}
        )

        per_sample = compute_per_sample_errors(
            X_true_scaled_flat=X_true_scaled_flat,
            X_hat_scaled_flat=X_hat_scaled_flat,
            X_true_scaled=X_true_scaled,
            X_hat_scaled=X_hat_scaled,
            X_true_original=X_orig,
            X_hat_original=X_hat_orig,
        )

        sample_table = sample_tables[split][["sample_id", "original_sample_id", "label_row"]].copy()
        sample_table["split"] = split
        sample_table["y_true"] = y_split.astype(int)
        sample_table["model"] = model_name
        sample_table["model_variant"] = variant
        sample_table["latent_dim"] = latent_dim
        for k, arr in per_sample.items():
            sample_table[k] = arr
        per_sample_tables.append(sample_table)

    return {
        "metrics_rows": metrics_rows,
        "feature_rows": feature_rows,
        "level_rows": level_rows,
        "temporal_rows": temporal_rows,
        "derived_rows": derived_rows,
        "per_sample_df": pd.concat(per_sample_tables, ignore_index=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 6 reconstruction-only baselines")
    parser.add_argument("--subset-dir", default="data/processed/minimal_subset")
    parser.add_argument("--output-dir", default="results/step6_reconstruction_baselines")
    parser.add_argument("--figures-dir", default="figures/step6_reconstruction_baselines")
    parser.add_argument("--artifact-dir", default="artifacts/step6_reconstruction_baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", default="train_mean_window,last_snapshot_repeat,pca,mlp_ae")
    parser.add_argument("--pca-latent-dims", default="8,16,32,64,128")
    parser.add_argument("--mlp-latent-dims", default="16,32,64")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--save-latents", action="store_true")
    parser.add_argument("--save-model-artifacts", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    pca_dims = parse_int_list(args.pca_latent_dims)
    mlp_dims = parse_int_list(args.mlp_latent_dims)

    subset = load_prediction_arrays(args.subset_dir)
    splits = {k: subset[k] for k in ["train", "val", "test"]}
    sample_tables = subset["sample_tables"]

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    validate_feature_contract(feature_dim=X_train.shape[2], window_len=X_train.shape[1])

    print("=== Step 6 Subset Split Sizes ===")
    print(f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print(f"X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")
    print(f"y_train_dist={class_distribution(y_train)}")
    print(f"y_val_dist={class_distribution(y_val)}")
    print(f"y_test_dist={class_distribution(y_test)}")

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    artifact_dir = Path(args.artifact_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    flat = {split: flatten_windows(arr[0]) for split, arr in splits.items()}

    scaler = StandardScaler()
    scaler.fit(flat["train"])
    scaled_flat = {split: scaler.transform(flat_arr) for split, flat_arr in flat.items()}

    input_dim = flat["train"].shape[1]

    # Always compute both baseline references for relative metrics.
    baseline_train_mean = TrainMeanWindowReconstructor()
    baseline_train_mean.fit(scaled_flat["train"])

    baseline_last_snapshot = LastSnapshotRepeatReconstructor()
    baseline_last_snapshot.fit(X_train)

    baseline_mse_train_mean: Dict[str, float] = {}
    baseline_mse_last_snapshot: Dict[str, float] = {}

    for split in ["train", "val", "test"]:
        X_orig, _ = splits[split]
        X_scaled_flat = scaled_flat[split]

        tm_hat_scaled_flat = baseline_train_mean.reconstruct(X_scaled_flat)
        tm_mse = float(np.mean(np.square(tm_hat_scaled_flat - X_scaled_flat)))
        baseline_mse_train_mean[split] = tm_mse

        ls_hat_orig = baseline_last_snapshot.reconstruct(X_orig)
        ls_hat_scaled_flat = scaler.transform(flatten_windows(ls_hat_orig))
        ls_mse = float(np.mean(np.square(ls_hat_scaled_flat - X_scaled_flat)))
        baseline_mse_last_snapshot[split] = ls_mse

    metrics_rows: List[Dict[str, object]] = []
    feature_rows: List[Dict[str, object]] = []
    level_rows: List[Dict[str, object]] = []
    temporal_rows: List[Dict[str, object]] = []
    derived_rows: List[Dict[str, object]] = []
    per_sample_parts: List[pd.DataFrame] = []
    model_manifest: List[Dict[str, object]] = []
    latent_manifest_entries: List[Dict[str, object]] = []

    def record_variant(
        model_name: str,
        latent_dim: int | None,
        reconstructor,
        train_seconds: float,
        compression_ratio: float | None,
        num_parameters: int,
        artifact_path: str | None,
    ) -> None:
        variant = _model_variant(model_name, latent_dim)
        result = _model_eval(
            model_name=model_name,
            latent_dim=latent_dim,
            reconstructor=reconstructor,
            splits=splits,
            sample_tables=sample_tables,
            scaler=scaler,
            scaled_flat=scaled_flat,
            baseline_mse_train_mean=baseline_mse_train_mean,
            baseline_mse_last_snapshot=baseline_mse_last_snapshot,
            compression_ratio=compression_ratio,
            num_parameters=num_parameters,
            train_seconds=train_seconds,
        )
        metrics_rows.extend(result["metrics_rows"])
        feature_rows.extend(result["feature_rows"])
        level_rows.extend(result["level_rows"])
        temporal_rows.extend(result["temporal_rows"])
        derived_rows.extend(result["derived_rows"])
        per_sample_parts.append(result["per_sample_df"])

        manifest_entry = {
            "model": model_name,
            "model_variant": variant,
            "latent_dim": latent_dim,
            "input_dim": int(input_dim),
            "compression_ratio": compression_ratio,
            "num_parameters": int(num_parameters),
            "scaler_fitted_on_train_only": True,
            "train_seconds": float(train_seconds),
            "artifact_saved": bool(artifact_path is not None),
            "artifact_path": artifact_path,
        }
        model_manifest.append(manifest_entry)

        for split in ["train", "val", "test"]:
            lat_shape = None
            lat_path = None
            if args.save_latents:
                X_split = splits[split][0]
                if model_name == "last_snapshot_repeat":
                    lat = reconstructor.encode(X_split)
                elif model_name == "train_mean_window":
                    lat = reconstructor.encode(scaled_flat[split])
                else:
                    lat = reconstructor.encode(scaled_flat[split])
                if lat is not None:
                    lat_shape = list(lat.shape)
                    lat_dir = artifact_dir / "latents"
                    lat_dir.mkdir(parents=True, exist_ok=True)
                    name = f"{model_name}_latent{latent_dim if latent_dim is not None else 'none'}_{split}.npy"
                    lat_path = str((lat_dir / name).resolve())
                    np.save(lat_path, lat)

            latent_manifest_entries.append(
                {
                    "latents_saved": bool(args.save_latents and lat_path is not None),
                    "save_latents_flag": bool(args.save_latents),
                    "model": model_name,
                    "latent_dim": latent_dim,
                    "split": split,
                    "latent_shape": lat_shape,
                    "artifact_path": lat_path,
                    "committed": False,
                }
            )

    if "train_mean_window" in selected_models:
        art = None
        if args.save_model_artifacts:
            model_dir = artifact_dir / "models"
            model_dir.mkdir(parents=True, exist_ok=True)
            path = model_dir / "train_mean_window_mean.npy"
            np.save(path, baseline_train_mean.mean_vector_)
            art = str(path.resolve())

        record_variant(
            model_name="train_mean_window",
            latent_dim=None,
            reconstructor=baseline_train_mean,
            train_seconds=0.0,
            compression_ratio=None,
            num_parameters=0,
            artifact_path=art,
        )

    if "last_snapshot_repeat" in selected_models:
        record_variant(
            model_name="last_snapshot_repeat",
            latent_dim=40,
            reconstructor=baseline_last_snapshot,
            train_seconds=0.0,
            compression_ratio=compute_compression_ratio(input_dim=input_dim, latent_dim=40),
            num_parameters=0,
            artifact_path=None,
        )

    if "pca" in selected_models:
        for latent_dim in pca_dims:
            model = PCAReconstructor(latent_dim=latent_dim, random_state=args.seed)
            t0 = time.perf_counter()
            model.fit(scaled_flat["train"])
            train_seconds = time.perf_counter() - t0

            art = None
            if args.save_model_artifacts:
                model_dir = artifact_dir / "models"
                model_dir.mkdir(parents=True, exist_ok=True)
                path = model_dir / f"pca_latent{latent_dim}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(model.model, f)
                art = str(path.resolve())

            params = int(model.model.components_.size) if model.model is not None else 0
            record_variant(
                model_name="pca",
                latent_dim=latent_dim,
                reconstructor=model,
                train_seconds=train_seconds,
                compression_ratio=compute_compression_ratio(input_dim=input_dim, latent_dim=latent_dim),
                num_parameters=params,
                artifact_path=art,
            )

    if "mlp_ae" in selected_models:
        for latent_dim in mlp_dims:
            config = MLPAEConfig(
                input_dim=input_dim,
                latent_dim=latent_dim,
                max_epochs=args.max_epochs,
                batch_size=args.batch_size,
                patience=10,
                lr=1e-3,
                weight_decay=1e-5,
                dropout=0.1,
                device=device,
                random_state=args.seed,
            )
            model = MLPAutoencoderReconstructor(config=config)
            model.fit(scaled_flat["train"], scaled_flat["val"])

            art = None
            if args.save_model_artifacts and model.model is not None:
                model_dir = artifact_dir / "models"
                model_dir.mkdir(parents=True, exist_ok=True)
                path = model_dir / f"mlp_ae_latent{latent_dim}.pt"
                torch.save(model.model.state_dict(), path)
                art = str(path.resolve())

            params = count_parameters(model.model) if model.model is not None else 0
            train_seconds = float(model.train_seconds_ or 0.0)
            record_variant(
                model_name="mlp_ae",
                latent_dim=latent_dim,
                reconstructor=model,
                train_seconds=train_seconds,
                compression_ratio=compute_compression_ratio(input_dim=input_dim, latent_dim=latent_dim),
                num_parameters=params,
                artifact_path=art,
            )

    metrics_df = pd.DataFrame(metrics_rows)
    feature_df = pd.DataFrame(feature_rows)
    level_df = pd.DataFrame(level_rows)
    temporal_df = pd.DataFrame(temporal_rows)
    derived_df = pd.DataFrame(derived_rows)
    per_sample_df = pd.concat(per_sample_parts, ignore_index=True)

    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    rate_df = metrics_df[
        ["model", "model_variant", "latent_dim", "split", "compression_ratio", "normalized_mse", "normalized_mae", "original_mae"]
    ].copy()
    rate_df = rate_df[(rate_df["compression_ratio"].notna()) & (rate_df["compression_ratio"] > 0)]
    rate_df.to_csv(output_dir / "rate_distortion.csv", index=False)

    feature_df.to_csv(output_dir / "feature_group_errors.csv", index=False)
    level_df.to_csv(output_dir / "level_wise_errors.csv", index=False)
    temporal_df.to_csv(output_dir / "temporal_errors.csv", index=False)
    derived_df.to_csv(output_dir / "derived_lob_errors.csv", index=False)
    per_sample_df.to_csv(output_dir / "per_sample_reconstruction_errors.csv", index=False)

    (output_dir / "model_manifest.json").write_text(json.dumps(to_jsonable(model_manifest), indent=2), encoding="utf-8")
    latent_manifest = {
        "latents_saved": bool(args.save_latents),
        "save_latents_flag": bool(args.save_latents),
        "entries": latent_manifest_entries,
    }
    (output_dir / "latent_manifest.json").write_text(json.dumps(to_jsonable(latent_manifest), indent=2), encoding="utf-8")

    run_config = {
        "subset_dir": str(Path(args.subset_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "figures_dir": str(figures_dir.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "seed": args.seed,
        "selected_models": selected_models,
        "pca_latent_dims": pca_dims,
        "mlp_ae_latent_dims": mlp_dims,
        "max_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "device": device,
        "scaler_policy": "train-only StandardScaler on flattened windows",
        "class_label_usage": "y labels copied only into per-sample diagnostics",
        "step4_protocol_note": "boundary-purged chronological split",
        "step5_relation": "prediction-only baseline already complete; Step 6 does not train prediction heads",
        "imbalance_gate": {
            "eps_threshold": IMBALANCE_EPS_THRESHOLD,
            "valid_ratio_threshold": IMBALANCE_VALID_RATIO_THRESHOLD,
            "valid_condition": "bid>=0 and ask>=0 and bid+ask>eps_threshold for true and reconstructed volumes",
            "invalid_policy": "set imbalance_mae to null and prefer volume_sum/diff diagnostics",
        },
        "step3_metadata_summary": subset["metadata"].get("summary", {}),
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    (output_dir / "run_config.json").write_text(json.dumps(to_jsonable(run_config), indent=2), encoding="utf-8")

    test_rows = metrics_df[metrics_df["split"] == "test"].copy()
    rank_col = test_rows["compression_ratio"].copy()
    rank_col = rank_col.fillna(np.inf)
    test_rows = test_rows.assign(compression_ratio_for_rank=rank_col)
    best_row = _select_best_model_row(test_rows)

    constrained = test_rows[(test_rows["latent_dim"].notna()) & (test_rows["latent_dim"] <= 40)].copy()
    best_constrained = None if constrained.empty else _select_best_model_row(constrained)

    best_pca = None
    pca_test = test_rows[test_rows["model"] == "pca"]
    if not pca_test.empty:
        best_pca = _select_best_model_row(pca_test)

    best_mlp = None
    mlp_test = test_rows[test_rows["model"] == "mlp_ae"]
    if not mlp_test.empty:
        best_mlp = _select_best_model_row(mlp_test)

    _plot_rate_distortion(rate_df, metrics_df, figures_dir / "rate_distortion_curve.png")
    _plot_scorecard(metrics_df, figures_dir / "reconstruction_scorecard_by_model.png")

    selected_for_group: List[Tuple[str, int | None]] = [
        ("last_snapshot_repeat", 40),
        ("train_mean_window", None),
    ]
    if best_pca is not None:
        selected_for_group.append(("pca", int(best_pca["latent_dim"])))
    if best_mlp is not None:
        selected_for_group.append(("mlp_ae", int(best_mlp["latent_dim"])))

    # Keep order unique
    seen = set()
    selected_unique = []
    for item in selected_for_group:
        if item not in seen:
            selected_unique.append(item)
            seen.add(item)

    _plot_feature_group(feature_df, derived_df, selected_unique, figures_dir / "feature_group_error_by_model.png")
    _plot_derived_lob_error_by_model(derived_df, selected_unique, figures_dir / "derived_lob_error_by_model.png")
    best_latent = None if pd.isna(best_row["latent_dim"]) else int(best_row["latent_dim"])
    _plot_level_heatmap(level_df, str(best_row["model"]), best_latent, figures_dir / "level_wise_error_heatmap_best_model.png")

    temporal_models = [("last_snapshot_repeat", 40)]
    if best_pca is not None:
        temporal_models.append(("pca", int(best_pca["latent_dim"])))
    if best_mlp is not None:
        temporal_models.append(("mlp_ae", int(best_mlp["latent_dim"])))
    _plot_temporal_profile(temporal_df, temporal_models, figures_dir / "temporal_error_profile.png")

    lines: List[str] = []
    lines.append("# Step 6 Reconstruction Baseline Summary")
    lines.append("")
    lines.append("## Split Sizes")
    lines.append(f"- train: {len(y_train)}")
    lines.append(f"- val: {len(y_val)}")
    lines.append(f"- test: {len(y_test)}")
    lines.append("")
    lines.append("## Models Run")
    lines.append(f"- selected: {selected_models}")
    lines.append(f"- pca latent dims: {pca_dims}")
    lines.append(f"- mlp_ae latent dims: {mlp_dims}")
    lines.append(
        f"- imbalance gate: eps_threshold={IMBALANCE_EPS_THRESHOLD}, "
        f"valid_ratio_threshold={IMBALANCE_VALID_RATIO_THRESHOLD}"
    )
    lines.append("")
    lines.append("## Best Test Reconstruction")
    lines.append(
        f"- best by normalized_mse: {_safe_label(str(best_row['model']), best_latent)}; "
        f"normalized_mse={float(best_row['normalized_mse']):.6f}, normalized_mae={float(best_row['normalized_mae']):.6f}, "
        f"original_mae={float(best_row['original_mae']):.6f}"
    )
    if best_constrained is not None:
        c_lat = None if pd.isna(best_constrained["latent_dim"]) else int(best_constrained["latent_dim"])
        lines.append(
            f"- best with latent_dim<=40: {_safe_label(str(best_constrained['model']), c_lat)}; "
            f"normalized_mse={float(best_constrained['normalized_mse']):.6f}"
        )
    lines.append("")

    ls_test = metrics_df[(metrics_df["model"] == "last_snapshot_repeat") & (metrics_df["split"] == "test")].iloc[0]
    for model_name in ["pca", "mlp_ae"]:
        part = metrics_df[(metrics_df["model"] == model_name) & (metrics_df["split"] == "test")]
        if part.empty:
            continue
        best_part = part.sort_values("normalized_mse").iloc[0]
        verdict = "beat" if float(best_part["normalized_mse"]) < float(ls_test["normalized_mse"]) else "did not beat"
        lines.append(
            f"- best {model_name} ({_safe_label(model_name, int(best_part['latent_dim']))}) {verdict} last_snapshot_repeat on test normalized_mse"
        )

    lines.append("")
    lines.append("## Error Concentration (Test)")
    best_feature_rows = feature_df[
        (feature_df["split"] == "test")
        & (feature_df["model"] == str(best_row["model"]))
        & (feature_df["latent_dim"].isna() if best_latent is None else feature_df["latent_dim"] == best_latent)
    ]
    if not best_feature_rows.empty:
        pivot = {str(r["group"]): float(r["normalized_mse"]) for _, r in best_feature_rows.iterrows()}
        lines.append(
            "- normalized_mse by group (best model): "
            f"price={pivot.get('price', math.nan):.6f}, volume={pivot.get('volume', math.nan):.6f}, "
            f"top_of_book={pivot.get('top_of_book', math.nan):.6f}, last_step={pivot.get('last_step', math.nan):.6f}"
        )

    best_derived = derived_df[
        (derived_df["split"] == "test")
        & (derived_df["model"] == str(best_row["model"]))
        & (derived_df["latent_dim"].isna() if best_latent is None else derived_df["latent_dim"] == best_latent)
    ]
    if not best_derived.empty:
        d = best_derived.iloc[0]
        top1_valid = bool(d.get("top1_imbalance_valid", False))
        top5_valid = bool(d.get("top5_imbalance_valid", False))
        lines.append(
            f"- derived MAE (best model): midprice={float(d['midprice_mae']):.6f}, "
            f"spread={float(d['spread_mae']):.6f}, "
            f"top1_volume_sum={float(d['top1_volume_sum_mae']):.6f}, "
            f"top5_volume_sum={float(d['top5_volume_sum_mae']):.6f}, "
            f"top1_volume_diff={float(d['top1_volume_diff_mae']):.6f}, "
            f"top5_volume_diff={float(d['top5_volume_diff_mae']):.6f}"
        )
        lines.append(
            f"- imbalance validity (best model): top1_valid={top1_valid}, top5_valid={top5_valid}, "
            f"top1_valid_ratio={float(d['top1_imbalance_valid_ratio']):.4f}, "
            f"top5_valid_ratio={float(d['top5_imbalance_valid_ratio']):.4f}"
        )
        if top1_valid:
            lines.append(f"- top1_imbalance_mae(valid-only)={float(d['top1_imbalance_mae']):.6f}")
        else:
            lines.append("- top1_imbalance_mae(valid-only)=null (valid ratio below threshold)")
        if top5_valid:
            lines.append(f"- top5_imbalance_mae(valid-only)={float(d['top5_imbalance_mae']):.6f}")
        else:
            lines.append("- top5_imbalance_mae(valid-only)=null (valid ratio below threshold)")

    lines.append("")
    lines.append("## Scope Guard")
    lines.append("- Step 6 measures reconstruction quality only.")
    lines.append("- Step 6 does not train prediction heads.")
    lines.append("- Step 6 does not run reconstruction-prediction alignment.")
    lines.append("- Step 7 will use per_sample_reconstruction_errors.csv for alignment analysis.")
    lines.append("- Imbalance metrics are reported only when non-negative volume and denominator-validity checks pass.")
    lines.append("- When imbalance validity is weak, volume-sum and volume-difference diagnostics are preferred.")

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== Step 6 Reconstruction Baselines Complete ===")
    print(f"models: {selected_models}")
    print(f"best test reconstruction: {_safe_label(str(best_row['model']), best_latent)}")
    print(f"results: {output_dir.resolve()}")
    print(f"figures: {figures_dir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
