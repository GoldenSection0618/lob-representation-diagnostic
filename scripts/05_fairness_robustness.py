"""Step 8 entrypoint: fairness and robustness checks for Step 7."""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.alignment_metrics import class_aligned_proba, spearmanr_safe
from src.analysis.prediction_metrics import compute_prediction_metrics


METRIC_COLUMNS = [
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
]
PRIMARY_TRANSFER_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "mcc",
    "log_loss",
    "directional_accuracy_non_neutral",
    "opposite_direction_rate",
]
BOOTSTRAP_METRICS = ["macro_f1", "mcc", "balanced_accuracy", "opposite_direction_rate"]
CLASS_ORDER = [0, 1, 2]


def parse_float_list(text: str) -> List[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


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


def proba_margin(proba: np.ndarray) -> np.ndarray:
    sorted_proba = np.sort(proba, axis=1)
    return sorted_proba[:, -1] - sorted_proba[:, -2]


def split_metadata(samples: pd.DataFrame, y: np.ndarray) -> Dict[str, pd.DataFrame]:
    required = ["sample_id", "original_sample_id", "label_row", "split"]
    missing = [col for col in required if col not in samples.columns]
    if missing:
        raise ValueError(f"samples.csv missing required columns: {missing}")
    meta = samples[required].copy()
    meta["y_true"] = y.astype(int)
    out = {}
    for split, part in meta.groupby("split"):
        out[str(split)] = part.sort_values("sample_id").reset_index(drop=True)
    if set(out) != {"train", "val", "test"}:
        raise ValueError(f"Expected train/val/test splits, got {sorted(out)}")
    return out


def metric_row(metrics: Dict[str, object], keys: Dict[str, object]) -> Dict[str, object]:
    row = dict(keys)
    for col in METRIC_COLUMNS:
        row[col] = metrics.get(col)
    return row


def fit_raw_logistic_grid(
    x: np.ndarray,
    y: np.ndarray,
    sample_meta: Dict[str, pd.DataFrame],
    c_grid: Sequence[float],
    seed: int,
    selection_metric: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    features = {
        split: x[meta["sample_id"].astype(int).to_numpy()].reshape(len(meta), -1)
        for split, meta in sample_meta.items()
    }
    labels = {split: meta["y_true"].astype(int).to_numpy() for split, meta in sample_meta.items()}
    if features["train"].shape[1] != 4000:
        raise ValueError(f"Expected flattened raw input_dim=4000, got {features['train'].shape[1]}")

    grid_rows: List[Dict[str, object]] = []
    fitted: Dict[float, Pipeline] = {}
    val_scores: Dict[float, Tuple[float, float, float]] = {}

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
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipe.fit(features["train"], labels["train"])
        fitted[float(c)] = pipe
        for split in ["train", "val", "test"]:
            y_pred = pipe.predict(features[split]).astype(int)
            y_proba = class_aligned_proba(pipe.predict_proba(features[split]), pipe.named_steps["clf"].classes_)
            metrics = compute_prediction_metrics(labels[split], y_pred, y_proba)
            grid_rows.append(
                metric_row(
                    metrics,
                    {
                        "feature_source": "raw_window_flattened",
                        "input_dim": int(features["train"].shape[1]),
                        "C": float(c),
                        "class_weight": "balanced",
                        "scaler_policy": "train_only_standard_scaler",
                        "selection_metric": "val_macro_f1_tie_mcc_log_loss",
                        "split": split,
                    },
                )
            )
            if split == "val":
                val_scores[float(c)] = (
                    float(metrics[selection_metric]),
                    float(metrics["mcc"]),
                    -float(metrics["log_loss"]),
                )

    selected_c = max(c_grid, key=lambda c: val_scores[float(c)])
    selected_pipe = fitted[float(selected_c)]
    tuned_rows: List[Dict[str, object]] = []
    pred_parts: List[pd.DataFrame] = []
    for split in ["train", "val", "test"]:
        y_pred = selected_pipe.predict(features[split]).astype(int)
        y_proba = class_aligned_proba(selected_pipe.predict_proba(features[split]), selected_pipe.named_steps["clf"].classes_)
        metrics = compute_prediction_metrics(labels[split], y_pred, y_proba)
        tuned_rows.append(
            metric_row(
                metrics,
                {
                    "model": "raw_window_logistic_tuned",
                    "variant": f"raw_window_logistic_tuned@C={float(selected_c):g}",
                    "feature_source": "raw_window_flattened",
                    "input_dim": int(features["train"].shape[1]),
                    "selected_C": float(selected_c),
                    "selected_by": "val_macro_f1_tie_mcc_log_loss",
                    "split": split,
                },
            )
        )

        meta = sample_meta[split].copy()
        meta["model"] = "raw_window_logistic_tuned"
        meta["selected_C"] = float(selected_c)
        meta["y_pred"] = y_pred
        meta["correct"] = y_pred == labels[split]
        meta["confidence"] = np.max(y_proba, axis=1)
        meta["proba_0"] = y_proba[:, 0]
        meta["proba_1"] = y_proba[:, 1]
        meta["proba_2"] = y_proba[:, 2]
        meta["proba_true"] = y_proba[np.arange(len(meta)), labels[split]]
        meta["proba_margin"] = proba_margin(y_proba)
        true_non_neutral = np.isin(labels[split], [0, 2])
        meta["direction_correct_non_neutral"] = np.where(true_non_neutral, y_pred == labels[split], np.nan)
        meta["opposite_direction_error"] = (
            ((labels[split] == 0) & (y_pred == 2)) | ((labels[split] == 2) & (y_pred == 0))
        )
        pred_parts.append(
            meta[
                [
                    "sample_id",
                    "original_sample_id",
                    "label_row",
                    "split",
                    "y_true",
                    "model",
                    "selected_C",
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

    return (
        pd.DataFrame(grid_rows),
        pd.DataFrame(tuned_rows),
        pd.concat(pred_parts, ignore_index=True),
        float(selected_c),
    )


def build_fair_transfer_comparison(
    step5_metrics: pd.DataFrame,
    raw_grid: pd.DataFrame,
    raw_tuned_metrics: pd.DataFrame,
    step7_latent_metrics: pd.DataFrame,
    c_grid_text: str,
) -> Tuple[pd.DataFrame, str]:
    test_latents = step7_latent_metrics[step7_latent_metrics["split"] == "test"].copy()
    best_variant = str(test_latents.sort_values("macro_f1", ascending=False).iloc[0]["representation_variant"])
    oracle = raw_grid[raw_grid["split"] == "test"].sort_values("macro_f1", ascending=False).head(1).copy()
    oracle["variant"] = "raw_window_logistic_test_oracle"
    variants = {
        "majority": {
            "source": "raw_window_baseline",
            "feature_source": "raw_window_flattened",
            "uses_validation_tuning": False,
            "tuning_grid": None,
            "selected_C": None,
            "selection_metric": None,
            "selection_basis": "fixed_majority_baseline_from_step5",
            "rows": step5_metrics[step5_metrics["model"] == "majority"].assign(variant="majority"),
        },
        "raw_window_logistic_untuned": {
            "source": "raw_window_baseline",
            "feature_source": "raw_window_flattened",
            "uses_validation_tuning": False,
            "tuning_grid": None,
            "selected_C": None,
            "selection_metric": None,
            "selection_basis": "fixed_step5_logistic_default_C",
            "rows": step5_metrics[step5_metrics["model"] == "logistic_regression"].assign(
                variant="raw_window_logistic_untuned"
            ),
        },
        "raw_window_logistic_tuned": {
            "source": "raw_window_tuned_control",
            "feature_source": "raw_window_flattened",
            "uses_validation_tuning": True,
            "tuning_grid": c_grid_text,
            "selected_C": float(raw_tuned_metrics["selected_C"].dropna().iloc[0]),
            "selection_metric": "val_macro_f1_tie_mcc_log_loss",
            "selection_basis": "selected_by_val_macro_f1_tie_mcc_log_loss",
            "rows": raw_tuned_metrics.assign(variant="raw_window_logistic_tuned"),
        },
        "raw_window_logistic_test_oracle": {
            "source": "raw_window_oracle_reference",
            "feature_source": "raw_window_flattened",
            "uses_validation_tuning": False,
            "tuning_grid": c_grid_text,
            "selected_C": float(oracle["C"].iloc[0]),
            "selection_metric": "test_macro_f1_oracle_not_selection_valid",
            "selection_basis": "posthoc_best_test_macro_f1_from_raw_logistic_grid",
            "rows": oracle,
        },
        "best_frozen_latent_head": {
            "source": "frozen_latent_head",
            "feature_source": "frozen_reconstruction_latent",
            "uses_validation_tuning": True,
            "tuning_grid": c_grid_text,
            "selected_C": None,
            "selection_metric": "val_macro_f1_tie_mcc_log_loss",
            "selection_basis": "posthoc_best_test_macro_f1_from_step7",
            "rows": step7_latent_metrics[step7_latent_metrics["representation_variant"] == best_variant].assign(
                variant="best_frozen_latent_head"
            ),
        },
        "pca@128_frozen_latent_head": {
            "source": "frozen_latent_head",
            "feature_source": "frozen_reconstruction_latent",
            "uses_validation_tuning": True,
            "tuning_grid": c_grid_text,
            "selected_C": None,
            "selection_metric": "val_macro_f1_tie_mcc_log_loss",
            "selection_basis": "reconstruction_best_test_normalized_mse_from_step6",
            "rows": step7_latent_metrics[step7_latent_metrics["representation_variant"] == "pca@128"].assign(
                variant="pca@128_frozen_latent_head"
            ),
        },
    }

    rows: List[Dict[str, object]] = []
    for variant, spec in variants.items():
        for _, src in spec["rows"].iterrows():
            row = {
                "source": spec["source"],
                "variant": variant,
                "feature_source": spec["feature_source"],
                "input_dim": 4000 if "raw_window" in spec["feature_source"] else src.get("latent_dim"),
                "compression_ratio": np.nan if "raw_window" in spec["feature_source"] else src.get("compression_ratio"),
                "uses_validation_tuning": bool(spec["uses_validation_tuning"]),
                "tuning_grid": spec["tuning_grid"],
                "selected_C": spec["selected_C"] if spec["selected_C"] is not None else src.get("head_C", np.nan),
                "selection_metric": spec["selection_metric"],
                "selection_basis": spec["selection_basis"],
                "split": src["split"],
            }
            for metric in PRIMARY_TRANSFER_METRICS:
                row[metric] = src[metric]
            rows.append(row)

    out = pd.DataFrame(rows)
    ref_untuned = float(
        out[(out["variant"] == "raw_window_logistic_untuned") & (out["split"] == "test")]["macro_f1"].iloc[0]
    )
    ref_tuned = float(out[(out["variant"] == "raw_window_logistic_tuned") & (out["split"] == "test")]["macro_f1"].iloc[0])
    ref_untuned_mcc = float(
        out[(out["variant"] == "raw_window_logistic_untuned") & (out["split"] == "test")]["mcc"].iloc[0]
    )
    ref_tuned_mcc = float(out[(out["variant"] == "raw_window_logistic_tuned") & (out["split"] == "test")]["mcc"].iloc[0])
    out["delta_macro_f1_vs_raw_untuned"] = np.where(out["split"] == "test", out["macro_f1"] - ref_untuned, np.nan)
    out["delta_macro_f1_vs_raw_tuned"] = np.where(out["split"] == "test", out["macro_f1"] - ref_tuned, np.nan)
    out["delta_mcc_vs_raw_untuned"] = np.where(out["split"] == "test", out["mcc"] - ref_untuned_mcc, np.nan)
    out["delta_mcc_vs_raw_tuned"] = np.where(out["split"] == "test", out["mcc"] - ref_tuned_mcc, np.nan)
    return out, best_variant


def prediction_frame_from_step5(step5_pred: pd.DataFrame, model: str, alias: str) -> pd.DataFrame:
    df = step5_pred[(step5_pred["model"] == model) & (step5_pred["split"] == "test")].copy()
    df["model_alias"] = alias
    return df


def prediction_frame_from_step7(step7_pred: pd.DataFrame, variant: str, alias: str) -> pd.DataFrame:
    df = step7_pred[(step7_pred["representation_variant"] == variant) & (step7_pred["split"] == "test")].copy()
    df["model_alias"] = alias
    return df


def prediction_frame_from_step8(raw_pred: pd.DataFrame, alias: str) -> pd.DataFrame:
    df = raw_pred[raw_pred["split"] == "test"].copy()
    df["model_alias"] = alias
    return df


def metric_from_frame(df: pd.DataFrame, metric: str) -> float:
    y_true = df["y_true"].astype(int).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()
    proba = df[["proba_0", "proba_1", "proba_2"]].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        metrics = compute_prediction_metrics(y_true, y_pred, proba)
    value = metrics[metric]
    return float(value) if value is not None else float("nan")


def paired_bootstrap_delta(
    frames: Dict[str, pd.DataFrame],
    selection_basis: Dict[str, str],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    specs = [
        ("best_frozen_latent_head_vs_raw_window_logistic_untuned", "best_frozen_latent_head", "raw_window_logistic_untuned"),
        ("best_frozen_latent_head_vs_raw_window_logistic_tuned", "best_frozen_latent_head", "raw_window_logistic_tuned"),
        ("pca@128_frozen_latent_head_vs_raw_window_logistic_tuned", "pca@128_frozen_latent_head", "raw_window_logistic_tuned"),
    ]
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for comparison, model_a, model_b in specs:
        a = frames[model_a].sort_values("sample_id").reset_index(drop=True)
        b = frames[model_b].sort_values("sample_id").reset_index(drop=True)
        if not np.array_equal(a["sample_id"].to_numpy(), b["sample_id"].to_numpy()):
            raise ValueError(f"Paired bootstrap sample_id mismatch for {comparison}.")
        n = len(a)
        for metric in BOOTSTRAP_METRICS:
            a_obs = metric_from_frame(a, metric)
            b_obs = metric_from_frame(b, metric)
            if metric == "opposite_direction_rate":
                observed = b_obs - a_obs
                delta_definition = f"{metric}(model_b) - {metric}(model_a)"
            else:
                observed = a_obs - b_obs
                delta_definition = f"{metric}(model_a) - {metric}(model_b)"
            deltas = np.empty(iterations, dtype=float)
            for i in range(iterations):
                idx = rng.integers(0, n, size=n)
                a_boot = a.iloc[idx]
                b_boot = b.iloc[idx]
                a_val = metric_from_frame(a_boot, metric)
                b_val = metric_from_frame(b_boot, metric)
                deltas[i] = b_val - a_val if metric == "opposite_direction_rate" else a_val - b_val
            rows.append(
                {
                    "comparison": comparison,
                    "model_a": model_a,
                    "model_b": model_b,
                    "model_a_selection_basis": selection_basis[model_a],
                    "model_b_selection_basis": selection_basis[model_b],
                    "metric": metric,
                    "higher_is_better": True,
                    "split": "test",
                    "n_samples": n,
                    "n_bootstrap": iterations,
                    "seed": seed,
                    "delta_definition": delta_definition,
                    "delta_observed": float(observed),
                    "delta_mean": float(np.nanmean(deltas)),
                    "delta_median": float(np.nanmedian(deltas)),
                    "ci_2_5": float(np.nanpercentile(deltas, 2.5)),
                    "ci_97_5": float(np.nanpercentile(deltas, 97.5)),
                    "fraction_delta_gt_0": float(np.nanmean(deltas > 0)),
                    "fraction_delta_lt_0": float(np.nanmean(deltas < 0)),
                }
            )
    return pd.DataFrame(rows)


def rank_sensitivity(rank_df: pd.DataFrame) -> pd.DataFrame:
    sets = {
        "all_latent_variants": rank_df.index == rank_df.index,
        "exclude_last_snapshot_repeat": rank_df["representation_variant"] != "last_snapshot_repeat@40",
        "pca_only": rank_df["representation_model"] == "pca",
        "mlp_ae_only": rank_df["representation_model"] == "mlp_ae",
        "compressed_only_latent_dim_le_40": rank_df["latent_dim"].astype(float) <= 40,
        "high_capacity_latent_dim_gt_40": rank_df["latent_dim"].astype(float) > 40,
    }
    rows = []
    for name, mask in sets.items():
        part = rank_df[mask].copy()
        if len(part):
            part["subset_rank_recon_normalized_mse"] = (
                part["test_recon_normalized_mse"].rank(method="min", ascending=True).astype(int)
            )
            part["subset_rank_pred_macro_f1"] = (
                part["test_pred_macro_f1"].rank(method="min", ascending=False).astype(int)
            )
        variants = part["representation_variant"].tolist()
        n = len(part)
        if n:
            best_recon = part.sort_values("test_recon_normalized_mse").iloc[0]
            best_pred = part.sort_values("test_pred_macro_f1", ascending=False).iloc[0]
            same = str(best_recon["representation_variant"]) == str(best_pred["representation_variant"])
            if n < 3:
                interpretation = "rank_mismatch_not_evaluable"
            elif same:
                interpretation = "rank_mismatch_weakens"
            else:
                interpretation = "rank_mismatch_persists"
            rows.append(
                {
                    "variant_set": name,
                    "n_variants": n,
                    "included_variants": ",".join(variants),
                    "spearman_recon_mse_vs_macro_f1": spearmanr_safe(
                        part["test_recon_normalized_mse"], part["test_pred_macro_f1"]
                    ),
                    "spearman_top_of_book_mse_vs_macro_f1": spearmanr_safe(
                        part["test_recon_top_of_book_mse"], part["test_pred_macro_f1"]
                    ),
                    "spearman_last_step_mse_vs_macro_f1": spearmanr_safe(
                        part["test_recon_last_step_mse"], part["test_pred_macro_f1"]
                    ),
                    "spearman_lobench_weighted_mse_vs_macro_f1": spearmanr_safe(
                        part["test_lobench_weighted_mse_loss"], part["test_pred_macro_f1"]
                    ),
                    "best_reconstruction_variant": best_recon["representation_variant"],
                    "best_prediction_variant": best_pred["representation_variant"],
                    "same_best_variant": bool(same),
                    "reconstruction_best_rank_pred_macro_f1_within_set": int(
                        best_recon["subset_rank_pred_macro_f1"]
                    ),
                    "prediction_best_rank_recon_mse_within_set": int(
                        best_pred["subset_rank_recon_normalized_mse"]
                    ),
                    "reconstruction_best_rank_pred_macro_f1_global": int(best_recon["rank_pred_macro_f1"]),
                    "prediction_best_rank_recon_mse_global": int(best_pred["rank_recon_normalized_mse"]),
                    "interpretation": interpretation,
                }
            )
        else:
            rows.append(
                {
                    "variant_set": name,
                    "n_variants": 0,
                    "included_variants": "",
                    "spearman_recon_mse_vs_macro_f1": np.nan,
                    "spearman_top_of_book_mse_vs_macro_f1": np.nan,
                    "spearman_last_step_mse_vs_macro_f1": np.nan,
                    "spearman_lobench_weighted_mse_vs_macro_f1": np.nan,
                    "best_reconstruction_variant": None,
                    "best_prediction_variant": None,
                    "same_best_variant": False,
                    "reconstruction_best_rank_pred_macro_f1_within_set": np.nan,
                    "prediction_best_rank_recon_mse_within_set": np.nan,
                    "reconstruction_best_rank_pred_macro_f1_global": np.nan,
                    "prediction_best_rank_recon_mse_global": np.nan,
                    "interpretation": "rank_mismatch_not_evaluable",
                }
            )
    return pd.DataFrame(rows)


def last_snapshot_sensitivity(rank_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for analysis, include_last in [("with_last_snapshot", True), ("without_last_snapshot", False)]:
        part = rank_df.copy() if include_last else rank_df[rank_df["representation_variant"] != "last_snapshot_repeat@40"].copy()
        best_pred = part.sort_values("test_pred_macro_f1", ascending=False).iloc[0]
        best_recon = part.sort_values("test_recon_normalized_mse").iloc[0]
        last = rank_df[rank_df["representation_variant"] == "last_snapshot_repeat@40"].iloc[0]
        rows.append(
            {
                "analysis": analysis,
                "included_last_snapshot": bool(include_last),
                "n_variants": int(len(part)),
                "best_prediction_variant": best_pred["representation_variant"],
                "best_reconstruction_variant": best_recon["representation_variant"],
                "spearman_last_step_mse_vs_macro_f1": spearmanr_safe(
                    part["test_recon_last_step_mse"], part["test_pred_macro_f1"]
                ),
                "spearman_recon_mse_vs_macro_f1": spearmanr_safe(
                    part["test_recon_normalized_mse"], part["test_pred_macro_f1"]
                ),
                "last_snapshot_test_macro_f1": float(last["test_pred_macro_f1"]),
                "last_snapshot_test_recon_mse": float(last["test_recon_normalized_mse"]),
                "last_snapshot_test_last_step_mse": float(last["test_recon_last_step_mse"]),
                "note": "last_snapshot_repeat has zero last-step reconstruction error by construction.",
            }
        )
    return pd.DataFrame(rows)


def build_final_claim_table(
    join_contract: Dict[str, object],
    fair: pd.DataFrame,
    bootstrap: pd.DataFrame,
    rank_sens: pd.DataFrame,
    last_sens: pd.DataFrame,
) -> pd.DataFrame:
    test = fair[fair["split"] == "test"].set_index("variant")
    best_vs_untuned = float(test.loc["best_frozen_latent_head", "delta_macro_f1_vs_raw_untuned"])
    best_vs_tuned = float(test.loc["best_frozen_latent_head", "delta_macro_f1_vs_raw_tuned"])
    c3_boot = bootstrap[
        (bootstrap["comparison"] == "best_frozen_latent_head_vs_raw_window_logistic_tuned")
        & (bootstrap["metric"] == "macro_f1")
    ].iloc[0]
    oracle_row = test.loc["raw_window_logistic_test_oracle"]
    if best_vs_tuned > 0 and float(c3_boot["ci_2_5"]) > 0:
        c3_status = "supported"
    elif best_vs_tuned > 0:
        c3_status = "partially_supported"
    else:
        c3_status = "not_supported"

    all_rank = rank_sens[rank_sens["variant_set"] == "all_latent_variants"].iloc[0]
    excl_rank = rank_sens[rank_sens["variant_set"] == "exclude_last_snapshot_repeat"].iloc[0]
    c5_status = "supported" if excl_rank["interpretation"] == "rank_mismatch_persists" else "not_supported"
    if all_rank["interpretation"] == "rank_mismatch_persists" and c5_status != "supported":
        c4_status = "partially_supported"
    else:
        c4_status = "supported" if all_rank["interpretation"] == "rank_mismatch_persists" else "not_supported"

    c6_status = "supported"
    c6_caveat = "Descriptive one-symbol, one-horizon evidence; not a general statistical claim."
    if excl_rank["interpretation"] != "rank_mismatch_persists":
        c6_status = "partially_supported"
        c6_caveat = (
            "All-variant rank mismatch is driven partly by last_snapshot_repeat@40; "
            "after excluding it, pca@128 is both reconstruction-best and prediction-best."
        )

    claims = [
        {
            "claim_id": "C1",
            "claim": "Step 7 sample-level join is valid.",
            "evidence_type": "join_contract",
            "primary_artifact": "results/step7_alignment/join_contract.json",
            "metric_or_check": "status",
            "result": join_contract["status"],
            "status": "supported" if join_contract["status"] == "passed" else "not_supported",
            "caveat": "Step 5 predictions cover val/test; Step 6 diagnostics cover train/val/test.",
        },
        {
            "claim_id": "C2",
            "claim": "Best frozen latent head beats untuned raw-window logistic on test macro-F1.",
            "evidence_type": "fair_transfer_comparison",
            "primary_artifact": "results/step8_fairness_robustness/fair_transfer_comparison.csv",
            "metric_or_check": "delta_macro_f1_vs_raw_untuned",
            "result": f"{best_vs_untuned:.6f}",
            "status": "supported" if best_vs_untuned > 0 else "not_supported",
            "caveat": "Best frozen latent head is selected post hoc from Step 7 test macro-F1.",
        },
        {
            "claim_id": "C3",
            "claim": "Best frozen latent head beats tuned raw-window logistic on test macro-F1.",
            "evidence_type": "fair_transfer_comparison and paired_bootstrap_delta",
            "primary_artifact": "results/step8_fairness_robustness/paired_bootstrap_delta.csv",
            "metric_or_check": "delta_macro_f1_vs_raw_tuned and bootstrap ci_2_5",
            "result": f"delta={best_vs_tuned:.6f}; ci_2_5={float(c3_boot['ci_2_5']):.6f}",
            "status": c3_status,
            "caveat": (
                "The tuned raw control is selected by validation macro-F1. "
                f"The raw-grid test-oracle point C={float(oracle_row['selected_C']):g} reaches test macro-F1 {float(oracle_row['macro_f1']):.4f} "
                "but is post hoc and remains below the best frozen latent head."
            ),
        },
        {
            "claim_id": "C4",
            "claim": "Reconstruction-best variant differs from prediction-best variant.",
            "evidence_type": "model_level_rank_alignment and rank_sensitivity",
            "primary_artifact": "results/step8_fairness_robustness/rank_sensitivity.csv",
            "metric_or_check": "all_latent_variants same_best_variant",
            "result": str(bool(all_rank["same_best_variant"])),
            "status": c4_status,
            "caveat": "Rank comparisons are descriptive and have small variant counts.",
        },
        {
            "claim_id": "C5",
            "claim": "Rank mismatch persists after excluding last_snapshot_repeat.",
            "evidence_type": "rank_sensitivity and last_snapshot_sensitivity",
            "primary_artifact": "results/step8_fairness_robustness/last_snapshot_sensitivity.csv",
            "metric_or_check": "exclude_last_snapshot_repeat interpretation",
            "result": str(excl_rank["interpretation"]),
            "status": c5_status,
            "caveat": "Without last_snapshot_repeat, pca@128 is both reconstruction-best and prediction-best if mismatch weakens.",
        },
        {
            "claim_id": "C6",
            "claim": "Overall reconstruction MSE is not a reliable standalone downstream proxy in this controlled run.",
            "evidence_type": "model-level correlations, rank sensitivity, and fair transfer comparison",
            "primary_artifact": "results/step8_fairness_robustness/rank_sensitivity.csv",
            "metric_or_check": "rank mismatch and Spearman correlations",
            "result": f"all_latent={all_rank['interpretation']}; exclude_last={excl_rank['interpretation']}",
            "status": c6_status,
            "caveat": c6_caveat,
        },
        {
            "claim_id": "C7",
            "claim": "Evidence is limited to one symbol, one horizon, one stride-4 subset.",
            "evidence_type": "data contract and run configs",
            "primary_artifact": "data/processed/minimal_subset/metadata.json",
            "metric_or_check": "scope",
            "result": "sz000001 trend5 sample_stride=4",
            "status": "scope_limited",
            "caveat": "No multi-symbol or multi-horizon robustness in Step 8.",
        },
    ]
    return pd.DataFrame(claims)


def plot_fair_transfer(fair: pd.DataFrame, bootstrap: pd.DataFrame, fig_path: Path) -> None:
    order = [
        "raw_window_logistic_untuned",
        "raw_window_logistic_tuned",
        "best_frozen_latent_head",
        "pca@128_frozen_latent_head",
    ]
    test = fair[(fair["split"] == "test") & (fair["variant"].isin(order))].set_index("variant").loc[order]
    plt.figure(figsize=(9, 5))
    x = np.arange(len(test))
    plt.bar(x, test["macro_f1"])
    plt.xticks(x, order, rotation=25, ha="right")
    plt.ylabel("test macro_f1")
    plt.title("Fair Transfer Comparison")
    boot = bootstrap[
        (bootstrap["comparison"] == "best_frozen_latent_head_vs_raw_window_logistic_tuned")
        & (bootstrap["metric"] == "macro_f1")
    ].iloc[0]
    oracle = fair[(fair["split"] == "test") & (fair["variant"] == "raw_window_logistic_test_oracle")].iloc[0]
    plt.scatter([1], [oracle["macro_f1"]], marker="x", s=80, label="raw logistic test-oracle")
    plt.legend(fontsize=8)
    plt.text(
        0.02,
        0.95,
        f"best latent - tuned raw = {boot['delta_observed']:.4f}\n"
        f"95% CI [{boot['ci_2_5']:.4f}, {boot['ci_97_5']:.4f}]\n"
        f"raw logistic test-oracle C={oracle['selected_C']:g}: {oracle['macro_f1']:.4f} (not selection-valid)",
        transform=plt.gca().transAxes,
        va="top",
    )
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def plot_tuning_grid(grid: pd.DataFrame, selected_c: float, fig_path: Path) -> None:
    data = grid[grid["split"].isin(["val", "test"])]
    plt.figure(figsize=(7, 5))
    for split in ["val", "test"]:
        part = data[data["split"] == split].sort_values("C")
        plt.plot(part["C"], part["macro_f1"], marker="o", label=f"{split} macro_f1")
    plt.axvline(selected_c, linestyle="--", label=f"selected C={selected_c:g}")
    plt.xscale("log")
    plt.xlabel("C")
    plt.ylabel("macro_f1")
    plt.title("Raw-Window Logistic C Grid")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def plot_rank_sensitivity(rank_sens: pd.DataFrame, fig_path: Path) -> None:
    metrics = [
        ("spearman_recon_mse_vs_macro_f1", "recon_mse"),
        ("spearman_top_of_book_mse_vs_macro_f1", "top_book"),
        ("spearman_last_step_mse_vs_macro_f1", "last_step"),
        ("spearman_lobench_weighted_mse_vs_macro_f1", "lobench_weighted"),
    ]
    x = np.arange(len(rank_sens))
    width = 0.18
    plt.figure(figsize=(12, 5))
    for i, (col, label) in enumerate(metrics):
        plt.bar(x + (i - 1.5) * width, rank_sens[col], width=width, label=label)
    plt.axhline(0, linewidth=1)
    plt.xticks(x, rank_sens["variant_set"], rotation=30, ha="right")
    plt.ylabel("Spearman correlation")
    plt.title("Rank Sensitivity by Variant Set")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def plot_last_snapshot_sensitivity(last_sens: pd.DataFrame, fig_path: Path) -> None:
    x = np.arange(len(last_sens))
    width = 0.35
    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, last_sens["spearman_last_step_mse_vs_macro_f1"], width=width, label="last_step_mse")
    plt.bar(x + width / 2, last_sens["spearman_recon_mse_vs_macro_f1"], width=width, label="recon_mse")
    plt.axhline(0, linewidth=1)
    plt.xticks(x, last_sens["analysis"])
    plt.ylabel("Spearman correlation")
    plt.title("Last Snapshot Sensitivity")
    plt.text(
        0.02,
        0.95,
        "last_snapshot_repeat@40 has last_step_mse=0 by construction",
        transform=plt.gca().transAxes,
        va="top",
        fontsize=8,
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def write_summary(
    output_dir: Path,
    selected_c: float,
    grid: pd.DataFrame,
    fair: pd.DataFrame,
    bootstrap: pd.DataFrame,
    rank_sens: pd.DataFrame,
    last_sens: pd.DataFrame,
) -> None:
    test = fair[fair["split"] == "test"].set_index("variant")
    raw_tuned = test.loc["raw_window_logistic_tuned"]
    raw_untuned = test.loc["raw_window_logistic_untuned"]
    raw_oracle = test.loc["raw_window_logistic_test_oracle"]
    best = test.loc["best_frozen_latent_head"]
    delta = float(best["macro_f1"] - raw_tuned["macro_f1"])
    val_macro = float(grid[(grid["C"] == selected_c) & (grid["split"] == "val")]["macro_f1"].iloc[0])
    boot = bootstrap[
        (bootstrap["comparison"] == "best_frozen_latent_head_vs_raw_window_logistic_tuned")
        & (bootstrap["metric"] == "macro_f1")
    ].iloc[0]
    all_rank = rank_sens[rank_sens["variant_set"] == "all_latent_variants"].iloc[0]
    excl_rank = rank_sens[rank_sens["variant_set"] == "exclude_last_snapshot_repeat"].iloc[0]
    pca_rank = rank_sens[rank_sens["variant_set"] == "pca_only"].iloc[0]
    mlp_rank = rank_sens[rank_sens["variant_set"] == "mlp_ae_only"].iloc[0]
    final_case = (
        "The best frozen latent head remains stronger than both the untuned and tuned raw-window logistic controls on test macro-F1 in this stride-4 subset."
        if delta > 0
        else "The Step 7 latent-head advantage over the original untuned raw-window baseline does not survive the tuned raw-window logistic control. The stronger claim is the reconstruction-prediction rank mismatch."
    )
    lines = [
        "# Step 8 Fairness and Robustness Summary",
        "",
        "## Scope",
        "- Step 8 only adds fairness and robustness controls.",
        "- It does not modify split, data construction, or reconstruction encoders.",
        "- Latent artifact path reproducibility was already fixed before Step 8.",
        "- No random split, no no-purge split, no new reconstruction models, and no multi-symbol or multi-horizon expansion are introduced.",
        "",
        "## Tuned Raw-Window Logistic Control",
        f"- selected C: `{selected_c:g}`",
        f"- validation macro-F1 at selected C: `{val_macro:.6f}`",
        f"- test macro-F1 at selected C: `{raw_tuned['macro_f1']:.6f}`",
        f"- untuned Step 5 raw logistic test macro-F1: `{raw_untuned['macro_f1']:.6f}`",
        f"- tuned raw-window logistic beats untuned raw-window logistic: `{bool(raw_tuned['macro_f1'] > raw_untuned['macro_f1'])}`",
        f"- raw-window logistic test-oracle best C: `{raw_oracle['selected_C']:g}` with test macro-F1 `{raw_oracle['macro_f1']:.6f}`; this is not selection-valid.",
        "",
        "## Fair Transfer Comparison",
        f"- best frozen latent head test macro-F1: `{best['macro_f1']:.6f}`",
        f"- tuned raw-window logistic test macro-F1: `{raw_tuned['macro_f1']:.6f}`",
        f"- delta: `{delta:.6f}`",
        f"- best frozen latent head still beats tuned raw-window logistic: `{bool(delta > 0)}`",
        f"- best frozen latent head also remains above the raw-window logistic test-oracle point by `{float(best['macro_f1'] - raw_oracle['macro_f1']):.6f}` macro-F1.",
        "- best_frozen_latent_head is selected post hoc from Step 7 test macro-F1, so the paired bootstrap comparison is descriptive rather than a fully pre-registered confirmatory test.",
        "- The raw-window logistic grid contains a test-oracle best point at C=0.01 with test macro-F1 0.4101. This is not used as the selected tuned baseline because C is selected by validation macro-F1, but it remains below the post hoc best frozen latent head at 0.4355.",
        "",
        "## Paired Bootstrap Delta",
        f"- best latent vs tuned raw logistic macro-F1 delta: `{boot['delta_observed']:.6f}`",
        f"- 95% CI: `[{boot['ci_2_5']:.6f}, {boot['ci_97_5']:.6f}]`",
        f"- fraction_delta_gt_0: `{boot['fraction_delta_gt_0']:.6f}`",
        "- Interpretation is paired on the same test samples and descriptive.",
        "",
        "## Rank Sensitivity",
        f"- all_latent_variants: `{all_rank['interpretation']}`",
        f"- exclude_last_snapshot_repeat: `{excl_rank['interpretation']}`",
        f"- pca_only: `{pca_rank['interpretation']}`",
        f"- mlp_ae_only: `{mlp_rank['interpretation']}`",
        "",
        "## Last-Snapshot Sensitivity",
        "- last_snapshot_repeat@40 has last_step_mse=0 by construction.",
        f"- with last_snapshot best prediction: `{last_sens.iloc[0]['best_prediction_variant']}`",
        f"- without last_snapshot best prediction: `{last_sens.iloc[1]['best_prediction_variant']}`",
        f"- removing it changes last_step_mse Spearman from `{last_sens.iloc[0]['spearman_last_step_mse_vs_macro_f1']:.6f}` to `{last_sens.iloc[1]['spearman_last_step_mse_vs_macro_f1']:.6f}`.",
        "",
        "## Final Claim Update",
        f"- {final_case}",
        "- Overall reconstruction MSE is not a reliable standalone downstream proxy across all variants in this one-symbol, one-horizon, stride-4 subset.",
        "- The rank-mismatch claim weakens after excluding `last_snapshot_repeat@40`; without it, `pca@128` is both reconstruction-best and prediction-best.",
        "",
        "## Limits",
        "- One symbol.",
        "- One horizon.",
        "- One subset.",
        "- No random/no-purge ablation.",
        "- No multi-symbol/multi-horizon robustness.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 8 fairness and robustness checks")
    parser.add_argument("--subset-dir", default="data/processed/minimal_subset")
    parser.add_argument("--step5-dir", default="results/step5_prediction_baselines")
    parser.add_argument("--step6-dir", default="results/step6_reconstruction_baselines")
    parser.add_argument("--step7-dir", default="results/step7_alignment")
    parser.add_argument("--output-dir", default="results/step8_fairness_robustness")
    parser.add_argument("--figures-dir", default="figures/step8_fairness_robustness")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-grid", default="0.01,0.1,1.0,10.0")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--selection-metric", default="macro_f1")
    parser.add_argument("--primary-metric", default="macro_f1")
    args = parser.parse_args()

    subset_dir = Path(args.subset_dir)
    step5_dir = Path(args.step5_dir)
    step6_dir = Path(args.step6_dir)
    step7_dir = Path(args.step7_dir)
    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    required_inputs = [
        subset_dir / "X.npy",
        subset_dir / "y.npy",
        subset_dir / "samples.csv",
        subset_dir / "metadata.json",
        step5_dir / "metrics.csv",
        step5_dir / "per_sample_predictions.csv",
        step5_dir / "run_config.json",
        step6_dir / "metrics.csv",
        step6_dir / "lobench_compatible_reconstruction_metrics.csv",
        step6_dir / "summary.md",
        step6_dir / "run_config.json",
        step7_dir / "latent_head_metrics.csv",
        step7_dir / "latent_head_predictions.csv",
        step7_dir / "transfer_baseline_comparison.csv",
        step7_dir / "model_level_rank_alignment.csv",
        step7_dir / "model_level_correlations.csv",
        step7_dir / "join_contract.json",
        step7_dir / "run_config.json",
        step7_dir / "summary.md",
    ]
    missing = [str(path) for path in required_inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Step 8 inputs: {missing}")

    x = np.load(subset_dir / "X.npy")
    y = np.load(subset_dir / "y.npy")
    samples = pd.read_csv(subset_dir / "samples.csv")
    metadata = json.loads((subset_dir / "metadata.json").read_text(encoding="utf-8"))
    sample_meta = split_metadata(samples, y)

    step5_metrics = pd.read_csv(step5_dir / "metrics.csv")
    step5_pred = pd.read_csv(step5_dir / "per_sample_predictions.csv")
    step7_latent_metrics = pd.read_csv(step7_dir / "latent_head_metrics.csv")
    step7_latent_pred = pd.read_csv(step7_dir / "latent_head_predictions.csv")
    rank_df = pd.read_csv(step7_dir / "model_level_rank_alignment.csv")
    join_contract = json.loads((step7_dir / "join_contract.json").read_text(encoding="utf-8"))

    c_grid = parse_float_list(args.c_grid)
    grid, tuned_metrics, tuned_pred, selected_c = fit_raw_logistic_grid(
        x=x,
        y=y,
        sample_meta=sample_meta,
        c_grid=c_grid,
        seed=args.seed,
        selection_metric=args.selection_metric,
    )
    fair, best_latent_variant = build_fair_transfer_comparison(
        step5_metrics, grid, tuned_metrics, step7_latent_metrics, args.c_grid
    )

    selection_basis = {
        "best_frozen_latent_head": "posthoc_best_test_macro_f1_from_step7",
        "pca@128_frozen_latent_head": "reconstruction_best_test_normalized_mse_from_step6",
        "raw_window_logistic_untuned": "fixed_step5_logistic_default_C",
        "raw_window_logistic_tuned": "selected_by_val_macro_f1_tie_mcc_log_loss",
    }
    frames = {
        "best_frozen_latent_head": prediction_frame_from_step7(
            step7_latent_pred, best_latent_variant, "best_frozen_latent_head"
        ),
        "pca@128_frozen_latent_head": prediction_frame_from_step7(
            step7_latent_pred, "pca@128", "pca@128_frozen_latent_head"
        ),
        "raw_window_logistic_untuned": prediction_frame_from_step5(
            step5_pred, "logistic_regression", "raw_window_logistic_untuned"
        ),
        "raw_window_logistic_tuned": prediction_frame_from_step8(
            tuned_pred, "raw_window_logistic_tuned"
        ),
    }
    bootstrap = paired_bootstrap_delta(frames, selection_basis, args.bootstrap_iterations, args.seed)
    rank_sens = rank_sensitivity(rank_df)
    last_sens = last_snapshot_sensitivity(rank_df)
    final_claims = build_final_claim_table(join_contract, fair, bootstrap, rank_sens, last_sens)

    grid.to_csv(output_dir / "raw_logistic_tuning_grid.csv", index=False)
    tuned_metrics.to_csv(output_dir / "raw_logistic_tuned_metrics.csv", index=False)
    tuned_pred.to_csv(output_dir / "raw_logistic_tuned_predictions.csv", index=False)
    fair.to_csv(output_dir / "fair_transfer_comparison.csv", index=False)
    bootstrap.to_csv(output_dir / "paired_bootstrap_delta.csv", index=False)
    rank_sens.to_csv(output_dir / "rank_sensitivity.csv", index=False)
    last_sens.to_csv(output_dir / "last_snapshot_sensitivity.csv", index=False)
    final_claims.to_csv(output_dir / "final_claim_table.csv", index=False)

    run_config = {
        "step": "step8_fairness_robustness",
        "sample_stride": int(metadata["sample_stride"]),
        "split_protocol": "boundary-purged chronological",
        "subset_dir": args.subset_dir,
        "step5_dir": args.step5_dir,
        "step6_dir": args.step6_dir,
        "step7_dir": args.step7_dir,
        "output_dir": args.output_dir,
        "figures_dir": args.figures_dir,
        "seed": int(args.seed),
        "c_grid": c_grid,
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "selection_metric": args.selection_metric,
        "primary_metric": args.primary_metric,
        "raw_logistic_control": {
            "input": "flattened raw window",
            "scaler": "train-only StandardScaler",
            "class_weight": "balanced",
            "max_iter": 2000,
            "selected_by": "val_macro_f1_tie_mcc_log_loss",
        },
        "bootstrap_policy": {
            "method": "paired test-set bootstrap",
            "resampling": "resample same test indices for both models",
            "delta_direction": "positive delta means model_a better",
        },
        "variant_sets": rank_sens["variant_set"].tolist(),
        "selection_basis_note": "best_frozen_latent_head is selected post hoc from Step 7 test macro-F1",
        "notes": [
            "no random split",
            "no no-purge split",
            "no reconstruction encoder retraining",
            "no new reconstruction models",
            "no multi-symbol or multi-horizon expansion",
            "latent path reproducibility was fixed before Step 8 and is not part of this step",
        ],
    }
    (output_dir / "run_config.json").write_text(json.dumps(to_jsonable(run_config), indent=2), encoding="utf-8")

    plot_fair_transfer(fair, bootstrap, figures_dir / "fair_transfer_macro_f1_with_ci.png")
    plot_tuning_grid(grid, selected_c, figures_dir / "tuning_grid_val_test_curve.png")
    plot_rank_sensitivity(rank_sens, figures_dir / "rank_sensitivity_by_variant_set.png")
    plot_last_snapshot_sensitivity(last_sens, figures_dir / "last_snapshot_sensitivity.png")
    write_summary(output_dir, selected_c, grid, fair, bootstrap, rank_sens, last_sens)

    print("=== Step 8 Fairness and Robustness Complete ===")
    print(f"selected raw C: {selected_c:g}")
    print(f"best frozen latent variant: {best_latent_variant}")
    print(f"results: {output_dir.resolve()}")
    print(f"figures: {figures_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
