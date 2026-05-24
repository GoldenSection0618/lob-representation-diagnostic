"""Step 7 entrypoint: reconstruction-prediction alignment and frozen latent transfer."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.alignment_metrics import (
    CLASS_ORDER,
    assign_quantile_bins,
    auroc_safe,
    class_aligned_proba,
    cliffs_delta,
    multiclass_bin_metrics,
    point_biserial_safe,
    spearmanr_safe,
)
from src.analysis.prediction_metrics import compute_prediction_metrics


SAMPLE_DIAGNOSTICS = [
    "normalized_mse",
    "normalized_mae",
    "top_of_book_mse",
    "last_step_mse",
    "midprice_mae",
    "spread_mae",
    "top1_volume_sum_mae",
    "top5_volume_sum_mae",
    "top1_volume_diff_mae",
    "top5_volume_diff_mae",
]
QUANTILE_DIAGNOSTICS = [
    "normalized_mse",
    "top_of_book_mse",
    "last_step_mse",
    "midprice_mae",
    "top1_volume_sum_mae",
]
FAILURE_DIAGNOSTICS = [
    "normalized_mse",
    "top_of_book_mse",
    "last_step_mse",
    "midprice_mae",
    "top1_volume_sum_mae",
    "top5_volume_sum_mae",
]
OUTCOMES = [
    "correct",
    "confidence",
    "proba_true",
    "opposite_direction_error",
    "direction_correct_non_neutral",
]


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


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def proba_margin(proba: np.ndarray) -> np.ndarray:
    sorted_proba = np.sort(proba, axis=1)
    return sorted_proba[:, -1] - sorted_proba[:, -2]


def add_prediction_derived_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    proba = out[["proba_0", "proba_1", "proba_2"]].to_numpy(dtype=float)
    y_true = out["y_true"].astype(int).to_numpy()
    out["proba_true"] = proba[np.arange(len(out)), y_true]
    out["proba_margin"] = proba_margin(proba)
    return out


def normalize_bool_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = out[col].astype(str).str.lower().map({"true": True, "false": False}).fillna(out[col])
    return out


def load_inputs(step5_dir: Path, step6_dir: Path) -> Dict[str, object]:
    required = [
        step5_dir / "metrics.csv",
        step5_dir / "per_sample_predictions.csv",
        step5_dir / "run_config.json",
        step6_dir / "metrics.csv",
        step6_dir / "lobench_compatible_reconstruction_metrics.csv",
        step6_dir / "per_sample_reconstruction_errors.csv",
        step6_dir / "latent_manifest.json",
        step6_dir / "run_config.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Step 7 inputs: {missing}")

    return {
        "step5_metrics": pd.read_csv(step5_dir / "metrics.csv"),
        "step5_predictions": pd.read_csv(step5_dir / "per_sample_predictions.csv"),
        "step5_config": json.loads((step5_dir / "run_config.json").read_text(encoding="utf-8")),
        "step6_metrics": pd.read_csv(step6_dir / "metrics.csv"),
        "step6_lobench": pd.read_csv(step6_dir / "lobench_compatible_reconstruction_metrics.csv"),
        "step6_per_sample": pd.read_csv(step6_dir / "per_sample_reconstruction_errors.csv"),
        "latent_manifest": json.loads((step6_dir / "latent_manifest.json").read_text(encoding="utf-8")),
        "step6_config": json.loads((step6_dir / "run_config.json").read_text(encoding="utf-8")),
    }


def build_sample_alignment_panel(pred_raw: pd.DataFrame, recon_raw: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    pred = pred_raw.rename(columns={"model": "prediction_model"}).copy()
    recon = recon_raw.rename(columns={"model": "reconstruction_model", "model_variant": "reconstruction_variant"}).copy()

    pred = normalize_bool_columns(pred, ["correct", "opposite_direction_error"])
    recon = normalize_bool_columns(recon, ["top1_imbalance_valid", "top5_imbalance_valid"])
    pred = add_prediction_derived_fields(pred)

    recon = recon[recon["split"].isin(["val", "test"])].copy()

    pred_key = ["sample_id", "split", "prediction_model"]
    recon_key = ["sample_id", "split", "reconstruction_variant"]
    duplicate_key_count = int(pred.duplicated(pred_key).sum() + recon.duplicated(recon_key).sum())

    merged = pred.merge(
        recon,
        on=["sample_id", "split", "y_true"],
        how="inner",
        suffixes=("_pred", "_recon"),
        validate="many_to_many",
    )

    sample_id_match = bool(len(merged) > 0)
    label_row_match = bool((merged["label_row_pred"] == merged["label_row_recon"]).all())
    original_sample_id_match = bool((merged["original_sample_id_pred"] == merged["original_sample_id_recon"]).all())
    y_true_match = True
    split_match = True

    if not label_row_match or not original_sample_id_match:
        raise ValueError("Step 5 and Step 6 sample metadata do not match after join.")

    merged["original_sample_id"] = merged["original_sample_id_pred"].astype(int)
    merged["label_row"] = merged["label_row_pred"].astype(int)

    merged["error_percentile_within_variant"] = (
        merged.groupby(["split", "reconstruction_variant"])["normalized_mse"].rank(pct=True, method="average")
    )
    merged["error_quartile_within_variant"] = (
        merged.groupby(["split", "reconstruction_variant"])["normalized_mse"]
        .transform(lambda s: assign_quantile_bins(s))
    )

    panel_cols = [
        "sample_id",
        "original_sample_id",
        "label_row",
        "split",
        "y_true",
        "prediction_model",
        "y_pred",
        "correct",
        "confidence",
        "proba_true",
        "proba_margin",
        "direction_correct_non_neutral",
        "opposite_direction_error",
        "reconstruction_model",
        "reconstruction_variant",
        "latent_dim",
        "compression_ratio",
        "normalized_mse",
        "normalized_mae",
        "top_of_book_mse",
        "last_step_mse",
        "midprice_mae",
        "spread_mae",
        "top1_volume_sum_mae",
        "top5_volume_sum_mae",
        "top1_volume_diff_mae",
        "top5_volume_diff_mae",
        "error_percentile_within_variant",
        "error_quartile_within_variant",
    ]
    panel = merged[panel_cols].copy()

    contract_bits = {
        "sample_id_match": sample_id_match,
        "label_row_match": label_row_match,
        "original_sample_id_match": original_sample_id_match,
        "y_true_match": y_true_match,
        "split_match": split_match,
        "duplicate_key_count": duplicate_key_count,
    }
    return panel, contract_bits


def build_join_contract(
    step5_config: Dict[str, object],
    step6_config: Dict[str, object],
    pred_raw: pd.DataFrame,
    recon_raw: pd.DataFrame,
    panel: pd.DataFrame,
    contract_bits: Dict[str, object],
) -> Dict[str, object]:
    step3_summary = step6_config["step3_metadata_summary"]
    step3_total = int(step3_summary["total_samples"])
    step3_stride = int(step3_summary["sample_stride"])
    pred_models = sorted(pred_raw["model"].astype(str).unique().tolist())
    recon_variants = sorted(recon_raw["model_variant"].astype(str).unique().tolist())
    expected_pred_rows = int((step3_summary["actual_counts"]["val"] + step3_summary["actual_counts"]["test"]) * len(pred_models))
    expected_recon_rows = int(step3_total * len(recon_variants))
    expected_joined_rows = int((step3_summary["actual_counts"]["val"] + step3_summary["actual_counts"]["test"]) * len(pred_models) * len(recon_variants))

    actual_joined = int(len(panel))
    status = "passed" if (
        len(pred_raw) == expected_pred_rows
        and len(recon_raw) == expected_recon_rows
        and actual_joined == expected_joined_rows
        and contract_bits["duplicate_key_count"] == 0
        and contract_bits["sample_id_match"]
        and contract_bits["label_row_match"]
        and contract_bits["y_true_match"]
        and contract_bits["split_match"]
    ) else "failed"

    return {
        "step3_total_samples": step3_total,
        "step3_sample_stride": step3_stride,
        "step5_prediction_rows": int(len(pred_raw)),
        "step6_reconstruction_rows": int(len(recon_raw)),
        "expected_prediction_rows": expected_pred_rows,
        "expected_reconstruction_rows": expected_recon_rows,
        "expected_joined_rows": expected_joined_rows,
        "actual_joined_rows": actual_joined,
        "join_keys": ["sample_id", "split", "y_true"],
        "sample_id_match": bool(contract_bits["sample_id_match"]),
        "label_row_match": bool(contract_bits["label_row_match"]),
        "y_true_match": bool(contract_bits["y_true_match"]),
        "split_match": bool(contract_bits["split_match"]),
        "duplicate_key_count": int(contract_bits["duplicate_key_count"]),
        "prediction_models": pred_models,
        "reconstruction_variants": recon_variants,
        "status": status,
    }


def binary_failure_for_outcome(df: pd.DataFrame, outcome: str) -> pd.Series | None:
    if outcome == "correct":
        return (~df["correct"].astype(bool)).astype(float)
    if outcome == "opposite_direction_error":
        return df["opposite_direction_error"].astype(bool).astype(float)
    if outcome == "direction_correct_non_neutral":
        valid = df["direction_correct_non_neutral"].notna()
        out = pd.Series(np.nan, index=df.index, dtype="float64")
        out.loc[valid] = (~df.loc[valid, "direction_correct_non_neutral"].astype(bool)).astype(float)
        return out
    return None


def sample_diagnostic_association(panel: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (split, pred_model, recon_variant), group in panel.groupby(["split", "prediction_model", "reconstruction_variant"]):
        for diag in SAMPLE_DIAGNOSTICS:
            score = group[diag]
            for outcome in OUTCOMES:
                if outcome in ["confidence", "proba_true"]:
                    outcome_values = group[outcome]
                    failure = None
                else:
                    failure = binary_failure_for_outcome(group, outcome)
                    outcome_values = failure

                valid = score.notna() & outcome_values.notna()
                n = int(valid.sum())
                spearman = spearmanr_safe(score[valid], outcome_values[valid])
                point_biserial = point_biserial_safe(score[valid], outcome_values[valid]) if failure is not None else float("nan")
                auc = auroc_safe(score[valid], failure[valid]) if failure is not None else float("nan")

                if failure is not None:
                    fail_mask = failure == 1.0
                    ref_mask = failure == 0.0
                    median_failure = float(score[fail_mask].median()) if fail_mask.any() else float("nan")
                    median_success = float(score[ref_mask].median()) if ref_mask.any() else float("nan")
                else:
                    median_failure = float("nan")
                    median_success = float("nan")

                rows.append(
                    {
                        "split": split,
                        "prediction_model": pred_model,
                        "reconstruction_variant": recon_variant,
                        "diagnostic_name": diag,
                        "outcome_name": outcome,
                        "n_samples": n,
                        "spearman_r": spearman,
                        "point_biserial_r": point_biserial,
                        "auroc_error_predicts_failure": auc,
                        "median_error_when_success": median_success,
                        "median_error_when_failure": median_failure,
                        "median_delta_failure_minus_success": (
                            float(median_failure - median_success)
                            if not math.isnan(median_failure) and not math.isnan(median_success)
                            else float("nan")
                        ),
                    }
                )
    return pd.DataFrame(rows)


def error_quantile_response(panel: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, group in panel.groupby(["split", "prediction_model", "reconstruction_variant"]):
        split, pred_model, recon_variant = keys
        for diag in QUANTILE_DIAGNOSTICS:
            work = group.copy()
            work["quantile_bin"] = assign_quantile_bins(work[diag])
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                part = work[work["quantile_bin"] == q]
                metrics = multiclass_bin_metrics(part)
                rows.append(
                    {
                        "split": split,
                        "prediction_model": pred_model,
                        "reconstruction_variant": recon_variant,
                        "diagnostic_name": diag,
                        "quantile_bin": q,
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def failure_mode_error_delta(panel: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, group in panel.groupby(["split", "prediction_model", "reconstruction_variant"]):
        split, pred_model, recon_variant = keys
        low_thr = group["confidence"].quantile(0.25)
        high_thr = group["confidence"].quantile(0.75)
        modes = {
            "incorrect_vs_correct": (~group["correct"].astype(bool), group["correct"].astype(bool)),
            "opposite_direction_vs_not": (
                group["opposite_direction_error"].astype(bool),
                ~group["opposite_direction_error"].astype(bool),
            ),
            "low_confidence_vs_high_confidence": (group["confidence"] <= low_thr, group["confidence"] >= high_thr),
            "non_neutral_missed_vs_detected": (
                group["y_true"].isin([0, 2]) & (group["y_pred"] == 1),
                group["y_true"].isin([0, 2]) & group["y_pred"].isin([0, 2]),
            ),
        }
        for diag in FAILURE_DIAGNOSTICS:
            for mode, (failure_mask, reference_mask) in modes.items():
                failure_values = group.loc[failure_mask, diag]
                reference_values = group.loc[reference_mask, diag]
                failure_median = float(failure_values.median()) if len(failure_values.dropna()) else float("nan")
                reference_median = float(reference_values.median()) if len(reference_values.dropna()) else float("nan")
                rows.append(
                    {
                        "split": split,
                        "prediction_model": pred_model,
                        "reconstruction_variant": recon_variant,
                        "diagnostic_name": diag,
                        "failure_mode": mode,
                        "n_failure": int(failure_values.notna().sum()),
                        "n_reference": int(reference_values.notna().sum()),
                        "failure_median": failure_median,
                        "reference_median": reference_median,
                        "median_delta": (
                            float(failure_median - reference_median)
                            if not math.isnan(failure_median) and not math.isnan(reference_median)
                            else float("nan")
                        ),
                        "median_ratio": (
                            float(failure_median / reference_median)
                            if not math.isnan(failure_median)
                            and not math.isnan(reference_median)
                            and abs(reference_median) > 1e-12
                            else float("nan")
                        ),
                        "cliffs_delta": cliffs_delta(failure_values, reference_values, seed=seed),
                    }
                )
    return pd.DataFrame(rows)


def latent_entries_by_variant(latent_manifest: Dict[str, object]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for entry in latent_manifest["entries"]:
        if not entry.get("latents_saved"):
            continue
        variant = entry["model"] if entry["latent_dim"] is None else f"{entry['model']}@{int(entry['latent_dim'])}"
        out.setdefault(variant, {})[entry["split"]] = entry
    return out


def sample_meta_from_recon(recon_raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    cols = ["sample_id", "original_sample_id", "label_row", "split", "y_true"]
    dedup = recon_raw[cols].drop_duplicates().sort_values(["split", "sample_id"]).reset_index(drop=True)
    return {split: part.sort_values("sample_id").reset_index(drop=True) for split, part in dedup.groupby("split")}


def fit_logistic_head(
    features: Dict[str, np.ndarray],
    sample_meta: Dict[str, pd.DataFrame],
    c_grid: Sequence[float],
    selection_metric: str,
    seed: int,
) -> Tuple[Pipeline, float]:
    y = {split: sample_meta[split]["y_true"].astype(int).to_numpy() for split in ["train", "val", "test"]}
    best = None
    for c in c_grid:
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=float(c),
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        pipe.fit(features["train"], y["train"])
        val_pred = pipe.predict(features["val"]).astype(int)
        val_proba = class_aligned_proba(pipe.predict_proba(features["val"]), pipe.named_steps["clf"].classes_)
        val_metrics = compute_prediction_metrics(y["val"], val_pred, val_proba)
        candidate = (
            float(val_metrics[selection_metric]),
            float(val_metrics["mcc"]),
            -float(val_metrics["log_loss"]),
            float(c),
            pipe,
        )
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        raise RuntimeError("No logistic head candidate was fitted.")
    return best[4], float(best[3])


def evaluate_logistic_head(
    pipe: Pipeline,
    features: Dict[str, np.ndarray],
    sample_meta: Dict[str, pd.DataFrame],
    representation_model: str,
    representation_variant: str,
    latent_dim: int | None,
    compression_ratio: float | None,
    head_c: float,
    selected_by: str,
) -> Tuple[List[Dict[str, object]], List[pd.DataFrame]]:
    metrics_rows: List[Dict[str, object]] = []
    pred_parts: List[pd.DataFrame] = []
    y = {split: sample_meta[split]["y_true"].astype(int).to_numpy() for split in ["train", "val", "test"]}

    for split in ["train", "val", "test"]:
        y_pred = pipe.predict(features[split]).astype(int)
        y_proba = class_aligned_proba(pipe.predict_proba(features[split]), pipe.named_steps["clf"].classes_)
        metrics = compute_prediction_metrics(y[split], y_pred, y_proba)
        row = {
            "representation_model": representation_model,
            "representation_variant": representation_variant,
            "latent_dim": latent_dim,
            "compression_ratio": compression_ratio,
            "head_model": "logistic_regression",
            "head_C": head_c,
            "head_class_weight": "balanced",
            "selected_by": selected_by,
            "split": split,
        }
        for key in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "mcc",
            "log_loss",
            "weighted_f1",
            "non_neutral_recall",
            "non_neutral_precision",
            "directional_accuracy_non_neutral",
            "up_down_macro_f1",
            "opposite_direction_rate",
        ]:
            row[key] = metrics[key]
        metrics_rows.append(row)

        meta = sample_meta[split].copy()
        meta["representation_variant"] = representation_variant
        meta["latent_dim"] = latent_dim
        meta["head_model"] = "logistic_regression"
        meta["head_C"] = head_c
        meta["y_pred"] = y_pred
        meta["correct"] = y_pred == y[split]
        meta["confidence"] = np.max(y_proba, axis=1)
        meta["proba_0"] = y_proba[:, 0]
        meta["proba_1"] = y_proba[:, 1]
        meta["proba_2"] = y_proba[:, 2]
        meta["proba_true"] = y_proba[np.arange(len(meta)), y[split]]
        meta["proba_margin"] = proba_margin(y_proba)
        true_non_neutral = np.isin(y[split], [0, 2])
        meta["direction_correct_non_neutral"] = np.where(true_non_neutral, y_pred == y[split], np.nan)
        meta["opposite_direction_error"] = (
            ((y[split] == 0) & (y_pred == 2)) | ((y[split] == 2) & (y_pred == 0))
        )
        pred_parts.append(
            meta[
                [
                    "sample_id",
                    "original_sample_id",
                    "label_row",
                    "split",
                    "y_true",
                    "representation_variant",
                    "latent_dim",
                    "head_model",
                    "head_C",
                    "y_pred",
                    "correct",
                    "confidence",
                    "proba_0",
                    "proba_1",
                    "proba_2",
                    "proba_true",
                    "proba_margin",
                    "direction_correct_non_neutral",
                    "opposite_direction_error",
                ]
            ]
        )
    return metrics_rows, pred_parts


def fit_latent_heads(
    latent_manifest: Dict[str, object],
    latent_artifact_dir: Path,
    step6_metrics: pd.DataFrame,
    sample_meta: Dict[str, pd.DataFrame],
    c_grid: Sequence[float],
    selection_metric: str,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    metrics_rows: List[Dict[str, object]] = []
    pred_parts: List[pd.DataFrame] = []
    variants_used: List[str] = []
    entries = latent_entries_by_variant(latent_manifest)

    for variant in sorted(entries):
        split_entries = entries[variant]
        if set(split_entries) != {"train", "val", "test"}:
            raise FileNotFoundError(f"Latent variant {variant} does not have train/val/test entries.")
        arrays = {}
        for split, entry in split_entries.items():
            manifest_path = Path(entry["artifact_path"])
            fallback_path = latent_artifact_dir / manifest_path.name
            if manifest_path.exists():
                path = manifest_path
            elif fallback_path.exists():
                path = fallback_path
            else:
                raise FileNotFoundError(
                    f"Missing latent file for {variant}/{split}: "
                    f"manifest path {manifest_path} and fallback path {fallback_path} do not exist."
                )
            arrays[split] = np.load(path)
            if list(arrays[split].shape) != entry["latent_shape"]:
                raise ValueError(f"Latent shape mismatch for {path}: {arrays[split].shape} vs {entry['latent_shape']}")

        pipe, head_c = fit_logistic_head(arrays, sample_meta, c_grid, selection_metric, seed)
        variants_used.append(variant)

        variant_row = step6_metrics[(step6_metrics["model_variant"] == variant) & (step6_metrics["split"] == "test")]
        latent_dim = None if variant_row.empty or pd.isna(variant_row.iloc[0]["latent_dim"]) else int(variant_row.iloc[0]["latent_dim"])
        compression_ratio = None if variant_row.empty or pd.isna(variant_row.iloc[0]["compression_ratio"]) else float(variant_row.iloc[0]["compression_ratio"])
        representation_model = str(variant_row.iloc[0]["model"]) if not variant_row.empty else variant.split("@")[0]
        variant_metrics, variant_predictions = evaluate_logistic_head(
            pipe,
            arrays,
            sample_meta,
            representation_model=representation_model,
            representation_variant=variant,
            latent_dim=latent_dim,
            compression_ratio=compression_ratio,
            head_c=head_c,
            selected_by=f"val_{selection_metric}_tie_mcc_log_loss",
        )
        metrics_rows.extend(variant_metrics)
        pred_parts.extend(variant_predictions)

    return pd.DataFrame(metrics_rows), pd.concat(pred_parts, ignore_index=True), variants_used


def fit_matched_raw_window_head(
    subset_dir: Path,
    sample_meta: Dict[str, pd.DataFrame],
    c_grid: Sequence[float],
    selection_metric: str,
    seed: int,
) -> pd.DataFrame:
    required = [subset_dir / "X.npy", subset_dir / "y.npy", subset_dir / "samples.csv"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw-window subset files for matched baseline: {missing}")

    x_all = np.load(subset_dir / "X.npy")
    y_all = np.load(subset_dir / "y.npy").astype(int)
    samples = pd.read_csv(subset_dir / "samples.csv")
    if not np.array_equal(samples["sample_id"].to_numpy(), np.arange(len(samples))):
        raise ValueError("samples.csv sample_id is not contiguous, so raw-window indexing is ambiguous.")
    if len(x_all) != len(samples) or len(y_all) != len(samples):
        raise ValueError("X.npy, y.npy, and samples.csv row counts do not match.")

    features: Dict[str, np.ndarray] = {}
    for split in ["train", "val", "test"]:
        meta = sample_meta[split].sort_values("sample_id").reset_index(drop=True)
        sample_ids = meta["sample_id"].astype(int).to_numpy()
        if not np.array_equal(y_all[sample_ids], meta["y_true"].astype(int).to_numpy()):
            raise ValueError(f"y.npy labels do not match Step 6 sample metadata for split {split}.")
        features[split] = x_all[sample_ids].reshape(len(sample_ids), -1)

    pipe, head_c = fit_logistic_head(features, sample_meta, c_grid, selection_metric, seed)
    metrics_rows, _ = evaluate_logistic_head(
        pipe,
        features,
        sample_meta,
        representation_model="raw_window",
        representation_variant="raw_window_logistic_tuned",
        latent_dim=int(np.prod(x_all.shape[1:])),
        compression_ratio=1.0,
        head_c=head_c,
        selected_by=f"val_{selection_metric}_tie_mcc_log_loss",
    )
    return pd.DataFrame(metrics_rows)


def transfer_baseline_comparison(
    step5_metrics: pd.DataFrame,
    latent_metrics: pd.DataFrame,
    matched_raw_metrics: pd.DataFrame,
) -> pd.DataFrame:
    raw = step5_metrics.copy()
    raw["source"] = "raw_window_baseline"
    raw["variant"] = raw["model"]
    raw["head_or_model"] = raw["model"]
    raw["latent_dim"] = np.nan
    raw["compression_ratio"] = np.nan
    raw["head_C"] = np.nan
    raw["selected_by"] = "step5_fixed_baseline_policy"
    raw_cols = [
        "source",
        "variant",
        "head_or_model",
        "latent_dim",
        "compression_ratio",
        "head_C",
        "selected_by",
        "split",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "mcc",
        "log_loss",
        "directional_accuracy_non_neutral",
        "opposite_direction_rate",
    ]

    latent = latent_metrics.copy()
    latent["source"] = "frozen_latent_head"
    latent["variant"] = latent["representation_variant"]
    latent["head_or_model"] = latent["head_model"]

    matched = matched_raw_metrics.copy()
    matched["source"] = "matched_raw_window_head"
    matched["variant"] = matched["representation_variant"]
    matched["head_or_model"] = matched["head_model"]
    return pd.concat([raw[raw_cols], matched[raw_cols], latent[raw_cols]], ignore_index=True)


def model_level_rank_alignment(
    step6_metrics: pd.DataFrame,
    lobench: pd.DataFrame,
    derived: pd.DataFrame,
    feature: pd.DataFrame,
    per_sample: pd.DataFrame,
    latent_metrics: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    recon = step6_metrics[step6_metrics["split"] == "test"].copy()
    recon = recon[recon["model_variant"] != "train_mean_window"].copy()
    recon = recon.rename(columns={"model_variant": "representation_variant", "model": "representation_model"})

    lob = lobench[lobench["split"] == "test"].rename(columns={"model_variant": "representation_variant"})
    der = derived[derived["split"] == "test"].rename(columns={"model_variant": "representation_variant"})
    top = feature[(feature["split"] == "test") & (feature["group"] == "top_of_book")].rename(
        columns={"model_variant": "representation_variant", "normalized_mse": "test_recon_top_of_book_mse"}
    )
    last = per_sample[per_sample["split"] == "test"].groupby("model_variant", as_index=False)["last_step_mse"].mean()
    last = last.rename(columns={"model_variant": "representation_variant", "last_step_mse": "test_recon_last_step_mse"})
    pred = latent_metrics[latent_metrics["split"] == "test"].copy()

    out = recon[
        [
            "representation_variant",
            "representation_model",
            "latent_dim",
            "compression_ratio",
            "normalized_mse",
            "normalized_mae",
        ]
    ].rename(
        columns={
            "normalized_mse": "test_recon_normalized_mse",
            "normalized_mae": "test_recon_normalized_mae",
        }
    )
    out = out.merge(top[["representation_variant", "test_recon_top_of_book_mse"]], on="representation_variant", how="left")
    out = out.merge(last, on="representation_variant", how="left")
    out = out.merge(
        der[["representation_variant", "midprice_mae", "top1_volume_sum_mae"]].rename(
            columns={"midprice_mae": "test_midprice_mae", "top1_volume_sum_mae": "test_top1_volume_sum_mae"}
        ),
        on="representation_variant",
        how="left",
    )
    out = out.merge(
        lob[["representation_variant", "lobench_weighted_mse_loss", "lobench_all_loss"]].rename(
            columns={
                "lobench_weighted_mse_loss": "test_lobench_weighted_mse_loss",
                "lobench_all_loss": "test_lobench_all_loss",
            }
        ),
        on="representation_variant",
        how="left",
    )
    out = out.merge(
        pred[
            [
                "representation_variant",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "mcc",
                "log_loss",
                "directional_accuracy_non_neutral",
                "opposite_direction_rate",
            ]
        ].rename(
            columns={
                "accuracy": "test_pred_accuracy",
                "balanced_accuracy": "test_pred_balanced_accuracy",
                "macro_f1": "test_pred_macro_f1",
                "mcc": "test_pred_mcc",
                "log_loss": "test_pred_log_loss",
                "directional_accuracy_non_neutral": "test_pred_directional_accuracy_non_neutral",
                "opposite_direction_rate": "test_pred_opposite_direction_rate",
            }
        ),
        on="representation_variant",
        how="inner",
    )

    out["rank_recon_normalized_mse"] = out["test_recon_normalized_mse"].rank(method="min", ascending=True).astype(int)
    out["rank_recon_top_of_book_mse"] = out["test_recon_top_of_book_mse"].rank(method="min", ascending=True).astype(int)
    out["rank_recon_lobench_weighted_mse"] = out["test_lobench_weighted_mse_loss"].rank(method="min", ascending=True).astype(int)
    out["rank_pred_macro_f1"] = out["test_pred_macro_f1"].rank(method="min", ascending=False).astype(int)
    out["rank_pred_mcc"] = out["test_pred_mcc"].rank(method="min", ascending=False).astype(int)
    out["rank_gap_macro_f1_vs_recon_mse"] = out["rank_pred_macro_f1"] - out["rank_recon_normalized_mse"]
    out["rank_gap_macro_f1_vs_lobench_weighted_mse"] = out["rank_pred_macro_f1"] - out["rank_recon_lobench_weighted_mse"]
    out["rank_gap_mcc_vs_recon_mse"] = out["rank_pred_mcc"] - out["rank_recon_normalized_mse"]
    out["interpretation"] = np.where(
        out["rank_gap_macro_f1_vs_recon_mse"].abs() <= 1,
        "aligned",
        np.where(
            out["rank_gap_macro_f1_vs_recon_mse"] > 1,
            "better_reconstruction_than_prediction",
            "better_prediction_than_reconstruction",
        ),
    )

    corr_specs = [
        ("test_recon_normalized_mse", "test_pred_macro_f1"),
        ("test_recon_top_of_book_mse", "test_pred_macro_f1"),
        ("test_recon_last_step_mse", "test_pred_macro_f1"),
        ("test_lobench_weighted_mse_loss", "test_pred_macro_f1"),
        ("test_lobench_weighted_mse_loss", "test_pred_mcc"),
    ]
    corr_rows = []
    for x, y in corr_specs:
        corr_rows.append(
            {
                "x_metric": x,
                "y_metric": y,
                "n_variants": int(out[[x, y]].dropna().shape[0]),
                "spearman_r": spearmanr_safe(out[x], out[y]),
                "note": "descriptive only; reconstruction variant count is small",
            }
        )
    return out, pd.DataFrame(corr_rows)


def plot_transfer(comparison: pd.DataFrame, fig_path: Path) -> None:
    test = comparison[comparison["split"] == "test"].copy()
    test = test.sort_values(["source", "macro_f1"], ascending=[True, False])
    plt.figure(figsize=(12, 5))
    x = np.arange(len(test))
    for source in ["raw_window_baseline", "matched_raw_window_head", "frozen_latent_head"]:
        mask = test["source"] == source
        plt.bar(x[mask], test.loc[mask, "macro_f1"], label=source)
    plt.xticks(x, test["variant"].tolist(), rotation=35, ha="right")
    plt.ylabel("test macro_f1")
    plt.title("Frozen Latent Heads vs Raw-Window Baselines")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def plot_latent_metrics(latent_metrics: pd.DataFrame, fig_path: Path) -> None:
    test = latent_metrics[latent_metrics["split"] == "test"].sort_values("macro_f1", ascending=False)
    x = np.arange(len(test))
    width = 0.25
    plt.figure(figsize=(12, 5))
    plt.bar(x - width, test["macro_f1"], width=width, label="macro_f1")
    plt.bar(x, test["balanced_accuracy"], width=width, label="balanced_accuracy")
    plt.bar(x + width, test["mcc"], width=width, label="mcc")
    plt.xticks(x, test["representation_variant"], rotation=35, ha="right")
    plt.ylabel("metric value")
    plt.title("Frozen Latent Head Primary Metrics (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def plot_rank_alignment(rank_df: pd.DataFrame, fig_path: Path) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(rank_df["rank_recon_normalized_mse"], rank_df["rank_pred_macro_f1"])
    for _, row in rank_df.iterrows():
        plt.annotate(row["representation_variant"], (row["rank_recon_normalized_mse"], row["rank_pred_macro_f1"]), fontsize=8)
    lim = [1, max(rank_df["rank_recon_normalized_mse"].max(), rank_df["rank_pred_macro_f1"].max())]
    plt.plot(lim, lim, linestyle="--")
    plt.xlabel("rank_recon_normalized_mse (lower is better)")
    plt.ylabel("rank_pred_macro_f1 (lower is better)")
    plt.title("Reconstruction vs Prediction Rank Alignment")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def plot_compression_tradeoff(rank_df: pd.DataFrame, fig_path: Path) -> None:
    data = rank_df.dropna(subset=["compression_ratio"]).copy()
    plt.figure(figsize=(8, 5))
    plt.scatter(data["compression_ratio"], data["test_pred_macro_f1"])
    for _, row in data.iterrows():
        plt.annotate(row["representation_variant"], (row["compression_ratio"], row["test_pred_macro_f1"]), fontsize=8)
    plt.xscale("log")
    plt.xlabel("compression_ratio")
    plt.ylabel("test macro_f1")
    plt.title("Compression vs Frozen-Head Prediction")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def plot_association_heatmap(assoc: pd.DataFrame, primary_model: str, fig_path: Path) -> None:
    diagnostics = [
        "normalized_mse",
        "top_of_book_mse",
        "last_step_mse",
        "midprice_mae",
        "spread_mae",
        "top1_volume_sum_mae",
        "top5_volume_sum_mae",
        "top1_volume_diff_mae",
        "top5_volume_diff_mae",
    ]
    columns = {
        "incorrect": ("correct", "auroc_error_predicts_failure"),
        "low_proba_true": ("proba_true", "spearman_r"),
        "opposite_direction_error": ("opposite_direction_error", "auroc_error_predicts_failure"),
        "non_neutral_missed": ("direction_correct_non_neutral", "auroc_error_predicts_failure"),
    }
    test = assoc[(assoc["split"] == "test") & (assoc["prediction_model"] == primary_model)].copy()
    matrix = np.full((len(diagnostics), len(columns)), np.nan)
    for i, diag in enumerate(diagnostics):
        for j, (_label, (outcome, metric)) in enumerate(columns.items()):
            vals = test[(test["diagnostic_name"] == diag) & (test["outcome_name"] == outcome)][metric].dropna()
            if len(vals):
                matrix[i, j] = float(vals.abs().mean()) if metric == "spearman_r" else float(vals.mean())

    plt.figure(figsize=(8, 6))
    im = plt.imshow(matrix, aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, label="association value")
    plt.xticks(range(len(columns)), list(columns.keys()), rotation=25, ha="right")
    plt.yticks(range(len(diagnostics)), diagnostics)
    plt.title(f"Diagnostic Association with Failure ({primary_model}, Test)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def plot_error_quantile_curve(qdf: pd.DataFrame, primary_model: str, fig_path: Path) -> None:
    diagnostics = ["normalized_mse", "top_of_book_mse", "last_step_mse"]
    data = qdf[(qdf["split"] == "test") & (qdf["prediction_model"] == primary_model)]
    data = data.groupby(["diagnostic_name", "quantile_bin"], as_index=False).agg(
        accuracy=("accuracy", "mean"),
        opposite_direction_rate=("opposite_direction_rate", "mean"),
    )
    plt.figure(figsize=(8, 5))
    for diag in diagnostics:
        part = data[data["diagnostic_name"] == diag].set_index("quantile_bin").reindex(["Q1", "Q2", "Q3", "Q4"])
        plt.plot(["Q1", "Q2", "Q3", "Q4"], 1.0 - part["accuracy"], marker="o", label=f"{diag} error_rate")
        plt.plot(["Q1", "Q2", "Q3", "Q4"], part["opposite_direction_rate"], marker="x", linestyle="--", label=f"{diag} opposite")
    plt.xlabel("reconstruction error quartile")
    plt.ylabel("rate")
    plt.title(f"Failure Rates by Error Quartile ({primary_model}, Test)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def write_summary(
    output_dir: Path,
    join_contract: Dict[str, object],
    comparison: pd.DataFrame,
    rank_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    assoc: pd.DataFrame,
    primary_model: str,
) -> None:
    test_comp = comparison[comparison["split"] == "test"].copy()
    raw_best = test_comp[test_comp["source"] == "raw_window_baseline"].sort_values("macro_f1", ascending=False).iloc[0]
    matched_raw = test_comp[test_comp["source"] == "matched_raw_window_head"].sort_values("macro_f1", ascending=False).iloc[0]
    latent_best = test_comp[test_comp["source"] == "frozen_latent_head"].sort_values("macro_f1", ascending=False).iloc[0]
    recon_best = rank_df.sort_values("test_recon_normalized_mse").iloc[0]
    same_variant = str(latent_best["variant"]) == str(recon_best["representation_variant"])
    corr_mse = corr_df[
        (corr_df["x_metric"] == "test_recon_normalized_mse") & (corr_df["y_metric"] == "test_pred_macro_f1")
    ].iloc[0]
    assoc_test = assoc[(assoc["split"] == "test") & (assoc["prediction_model"] == primary_model)]
    assoc_focus = assoc_test[assoc_test["outcome_name"] == "correct"].copy()
    assoc_focus = assoc_focus.groupby("diagnostic_name", as_index=False)["auroc_error_predicts_failure"].mean()
    best_assoc = assoc_focus.sort_values("auroc_error_predicts_failure", ascending=False).iloc[0]

    lines = [
        "# Step 7 Reconstruction-Prediction Alignment Summary",
        "",
        "## Scope",
        "- Step 7 uses the locked stride-4 boundary-purged chronological protocol.",
        "- Step 7 does not run random split or no-purge split ablations.",
        "- Reconstruction encoders are frozen; Step 7 trains only logistic heads on saved latent arrays and one matched flattened raw-window control.",
        "- Evidence is limited to one symbol (`sz000001`), one horizon (`trend5`), and one subset.",
        "",
        "## Join Contract",
        f"- status: `{join_contract['status']}`",
        f"- expected joined rows: `{join_contract['expected_joined_rows']}`",
        f"- actual joined rows: `{join_contract['actual_joined_rows']}`",
        f"- duplicate key count: `{join_contract['duplicate_key_count']}`",
        "",
        "## Sample-Level Difficulty Alignment",
        f"- Primary sample-analysis prediction model: `{primary_model}`.",
        "- Diagnostic-outcome associations are descriptive and computed across sample-level reconstruction diagnostics.",
        "- The heatmap uses AUROC where failure is binary; for low `proba_true`, it uses absolute Spearman association.",
        "",
        "## Frozen Latent Transfer",
        f"- Best fixed Step 5 raw-window baseline by test macro-F1: `{raw_best['variant']}` (`{raw_best['macro_f1']:.6f}`).",
        f"- Matched raw-window logistic head by test macro-F1: `{matched_raw['variant']}` (`{matched_raw['macro_f1']:.6f}`, selected C=`{matched_raw['head_C']}`).",
        f"- Best frozen latent head by test macro-F1: `{latent_best['variant']}` (`{latent_best['macro_f1']:.6f}`).",
        f"- Frozen latent head beat fixed Step 5 raw-window baseline: `{bool(latent_best['macro_f1'] > raw_best['macro_f1'])}`.",
        f"- Frozen latent head beat matched tuned raw-window head: `{bool(latent_best['macro_f1'] > matched_raw['macro_f1'])}`.",
        "",
        "## Reconstruction-Prediction Rank Alignment",
        f"- Best test reconstruction normalized_mse variant: `{recon_best['representation_variant']}`.",
        f"- Best test frozen-head macro-F1 variant: `{latent_best['variant']}`.",
        f"- Same variant: `{same_variant}`.",
        f"- Spearman(test_recon_normalized_mse, test_pred_macro_f1): `{corr_mse['spearman_r']:.6f}`.",
        "",
        "## Failure Diagnostics",
        f"- Highest mean AUROC for incorrect prediction under `{primary_model}`: `{best_assoc['diagnostic_name']}` (`{best_assoc['auroc_error_predicts_failure']:.6f}`).",
        "- Compare `normalized_mse`, `top_of_book_mse`, and `last_step_mse` in the quantile curve before treating overall reconstruction MSE as a downstream proxy.",
        "",
        "## Limits",
        "- This is one-symbol, one-horizon, one-subset evidence.",
        "- Rank correlations use only nine frozen latent variants and should not be read as significance tests.",
        "- Step 7 does not claim cross-symbol, cross-horizon, or trading-PnL generality.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 7 reconstruction-prediction alignment")
    parser.add_argument("--subset-dir", default="data/processed/minimal_subset")
    parser.add_argument("--step5-dir", default="results/step5_prediction_baselines")
    parser.add_argument("--step6-dir", default="results/step6_reconstruction_baselines")
    parser.add_argument("--latent-artifact-dir", default="artifacts/step6_reconstruction_baselines/latents")
    parser.add_argument("--output-dir", default="results/step7_alignment")
    parser.add_argument("--figures-dir", default="figures/step7_alignment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--head-c-grid", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--primary-prediction-model-for-sample-analysis", default="logistic_regression")
    parser.add_argument("--selection-metric", default="macro_f1")
    args = parser.parse_args()

    subset_dir = Path(args.subset_dir)
    step5_dir = Path(args.step5_dir)
    step6_dir = Path(args.step6_dir)
    latent_artifact_dir = Path(args.latent_artifact_dir)
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(step5_dir, step6_dir)
    recon_for_panel = inputs["step6_per_sample"].merge(
        inputs["step6_metrics"][["model_variant", "split", "compression_ratio"]],
        on=["model_variant", "split"],
        how="left",
        validate="many_to_one",
    )
    panel, bits = build_sample_alignment_panel(inputs["step5_predictions"], recon_for_panel)
    join_contract = build_join_contract(
        inputs["step5_config"],
        inputs["step6_config"],
        inputs["step5_predictions"],
        inputs["step6_per_sample"],
        panel,
        bits,
    )
    if join_contract["status"] != "passed":
        raise RuntimeError(f"Join contract failed: {join_contract}")

    sample_meta = sample_meta_from_recon(inputs["step6_per_sample"])
    c_grid = parse_float_list(args.head_c_grid)
    latent_metrics, latent_predictions, latent_variants = fit_latent_heads(
        inputs["latent_manifest"],
        latent_artifact_dir,
        inputs["step6_metrics"],
        sample_meta,
        c_grid,
        args.selection_metric,
        args.seed,
    )
    matched_raw_metrics = fit_matched_raw_window_head(
        subset_dir,
        sample_meta,
        c_grid,
        args.selection_metric,
        args.seed,
    )

    assoc = sample_diagnostic_association(panel)
    qdf = error_quantile_response(panel)
    failure_delta = failure_mode_error_delta(panel, seed=args.seed)
    comparison = transfer_baseline_comparison(inputs["step5_metrics"], latent_metrics, matched_raw_metrics)
    rank_df, corr_df = model_level_rank_alignment(
        inputs["step6_metrics"],
        inputs["step6_lobench"],
        pd.read_csv(step6_dir / "derived_lob_errors.csv"),
        pd.read_csv(step6_dir / "feature_group_errors.csv"),
        inputs["step6_per_sample"],
        latent_metrics,
    )

    panel.to_csv(output_dir / "sample_alignment_panel.csv", index=False)
    assoc.to_csv(output_dir / "sample_diagnostic_association.csv", index=False)
    qdf.to_csv(output_dir / "error_quantile_response.csv", index=False)
    failure_delta.to_csv(output_dir / "failure_mode_error_delta.csv", index=False)
    latent_metrics.to_csv(output_dir / "latent_head_metrics.csv", index=False)
    latent_predictions.to_csv(output_dir / "latent_head_predictions.csv", index=False)
    comparison.to_csv(output_dir / "transfer_baseline_comparison.csv", index=False)
    rank_df.to_csv(output_dir / "model_level_rank_alignment.csv", index=False)
    corr_df.to_csv(output_dir / "model_level_correlations.csv", index=False)
    (output_dir / "join_contract.json").write_text(json.dumps(to_jsonable(join_contract), indent=2), encoding="utf-8")

    run_config = {
        "step": "step7_alignment",
        "sample_stride": int(inputs["step6_config"]["step3_metadata_summary"]["sample_stride"]),
        "split_protocol": "boundary-purged chronological",
        "subset_dir": args.subset_dir,
        "step5_dir": args.step5_dir,
        "step6_dir": args.step6_dir,
        "latent_artifact_dir": args.latent_artifact_dir,
        "output_dir": args.output_dir,
        "figures_dir": args.figures_dir,
        "seed": args.seed,
        "prediction_models": join_contract["prediction_models"],
        "reconstruction_variants": join_contract["reconstruction_variants"],
        "latent_head_variants": latent_variants,
        "head_model": "logistic_regression",
        "head_c_grid": c_grid,
        "head_selection_metric": args.selection_metric,
        "latent_path_resolution": "manifest absolute path first, fallback to --latent-artifact-dir / basename",
        "matched_raw_window_head": {
            "source": "matched_raw_window_head",
            "variant": "raw_window_logistic_tuned",
            "features": "flattened Step 3 raw windows from X.npy",
            "scaler": "train-only StandardScaler inside sklearn Pipeline",
            "class_weight": "balanced",
            "c_grid": c_grid,
            "selection": f"validation {args.selection_metric}, tie-broken by validation MCC then validation log_loss",
            "test_policy": "test split used only for final evaluation",
        },
        "primary_prediction_model_for_sample_analysis": args.primary_prediction_model_for_sample_analysis,
        "join_keys": ["sample_id", "split", "y_true"],
        "sample_level_diagnostics": SAMPLE_DIAGNOSTICS,
        "sample_level_outcomes": OUTCOMES,
        "rank_alignment_metrics": [
            "test_recon_normalized_mse",
            "test_recon_top_of_book_mse",
            "test_lobench_weighted_mse_loss",
            "test_pred_macro_f1",
            "test_pred_mcc",
        ],
        "cliffs_delta_policy": "exact unless a group exceeds 5000 rows; then deterministic capped sample with seed",
        "notes": [
            "no random split",
            "no no-purge split",
            "reconstruction encoders are frozen",
            "Step 7 does not retrain reconstruction models",
            "evidence limited to sz000001 trend5 stride-4 subset",
        ],
    }
    (output_dir / "run_config.json").write_text(json.dumps(to_jsonable(run_config), indent=2), encoding="utf-8")

    plot_transfer(comparison, figures_dir / "transfer_vs_raw_baselines.png")
    plot_latent_metrics(latent_metrics, figures_dir / "latent_head_primary_metrics.png")
    plot_rank_alignment(rank_df, figures_dir / "reconstruction_prediction_rank_alignment.png")
    plot_compression_tradeoff(rank_df, figures_dir / "compression_prediction_tradeoff.png")
    plot_association_heatmap(assoc, args.primary_prediction_model_for_sample_analysis, figures_dir / "diagnostic_outcome_association_heatmap.png")
    plot_error_quantile_curve(qdf, args.primary_prediction_model_for_sample_analysis, figures_dir / "error_quantile_failure_curve.png")

    write_summary(output_dir, join_contract, comparison, rank_df, corr_df, assoc, args.primary_prediction_model_for_sample_analysis)

    print("=== Step 7 Alignment Complete ===")
    print(f"joined rows: {join_contract['actual_joined_rows']}")
    print(f"latent variants: {latent_variants}")
    print(f"results: {output_dir.resolve()}")
    print(f"figures: {figures_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
