"""Step 9: validation-selected frozen latent transfer audit."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.prediction_metrics import compute_prediction_metrics


BOOTSTRAP_METRICS = ["macro_f1", "mcc", "balanced_accuracy", "opposite_direction_rate"]
PRIMARY_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "mcc",
    "log_loss",
    "opposite_direction_rate",
]


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj):
        return None
    return obj


def metric_from_frame(df: pd.DataFrame, metric: str) -> float:
    y_true = df["y_true"].astype(int).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()
    y_proba = df[["proba_0", "proba_1", "proba_2"]].to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        metrics = compute_prediction_metrics(y_true, y_pred, y_proba)
    value = metrics[metric]
    return float(value) if value is not None else float("nan")


def select_candidates(latent_metrics: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
    val = latent_metrics[latent_metrics["split"] == "val"].copy()
    test = latent_metrics[latent_metrics["split"] == "test"].copy()
    required = {
        "representation_model",
        "representation_variant",
        "latent_dim",
        "compression_ratio",
        "macro_f1",
        "mcc",
        "log_loss",
    }
    missing = sorted(required - set(val.columns))
    if missing:
        raise ValueError(f"latent_head_metrics.csv missing required columns: {missing}")
    if len(val) != len(test):
        raise ValueError("Expected one validation and one test row per latent candidate.")

    val_ranked = val.sort_values(
        ["macro_f1", "mcc", "log_loss", "latent_dim", "representation_variant"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    val_ranked["val_selection_rank"] = np.arange(1, len(val_ranked) + 1)

    test_ranked = test.sort_values(
        ["macro_f1", "mcc", "log_loss", "latent_dim", "representation_variant"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)
    test_ranked["test_rank_by_macro_f1"] = np.arange(1, len(test_ranked) + 1)

    selected_by_validation = str(val_ranked.iloc[0]["representation_variant"])
    selected_by_test = str(test_ranked.iloc[0]["representation_variant"])

    panel = val_ranked[
        [
            "representation_model",
            "representation_variant",
            "latent_dim",
            "compression_ratio",
            "macro_f1",
            "mcc",
            "log_loss",
            "val_selection_rank",
        ]
    ].rename(
        columns={
            "macro_f1": "val_macro_f1",
            "mcc": "val_mcc",
            "log_loss": "val_log_loss",
        }
    )
    test_panel = test_ranked[
        [
            "representation_variant",
            "macro_f1",
            "mcc",
            "log_loss",
            "test_rank_by_macro_f1",
        ]
    ].rename(
        columns={
            "macro_f1": "test_macro_f1",
            "mcc": "test_mcc",
            "log_loss": "test_log_loss",
        }
    )
    panel = panel.merge(test_panel, on="representation_variant", how="inner", validate="one_to_one")
    panel["selected_by_validation"] = panel["representation_variant"] == selected_by_validation
    panel["selected_by_test_posthoc"] = panel["representation_variant"] == selected_by_test

    def note(row: pd.Series) -> str:
        notes = []
        if bool(row["selected_by_validation"]):
            notes.append("validation_selected")
        if bool(row["selected_by_test_posthoc"]):
            notes.append("test_posthoc_best")
        return ";".join(notes) if notes else "candidate"

    panel["selection_note"] = panel.apply(note, axis=1)
    panel = panel.sort_values("val_selection_rank").reset_index(drop=True)
    if int(panel["selected_by_validation"].sum()) != 1:
        raise ValueError("selected_by_validation must have exactly one True row.")
    if int(panel["selected_by_test_posthoc"].sum()) != 1:
        raise ValueError("selected_by_test_posthoc must have exactly one True row.")
    return panel, selected_by_validation, selected_by_test


def best_reconstruction_variant(step7_dir: Path) -> str:
    rank_path = step7_dir / "model_level_rank_alignment.csv"
    if rank_path.exists():
        rank = pd.read_csv(rank_path)
        return str(rank.sort_values("test_recon_normalized_mse", ascending=True).iloc[0]["representation_variant"])
    return "pca@128"


def test_metric_row(latent_metrics: pd.DataFrame, representation_variant: str, alias: str, selection_basis: str) -> Dict[str, object]:
    row = latent_metrics[
        (latent_metrics["split"] == "test") & (latent_metrics["representation_variant"] == representation_variant)
    ].iloc[0]
    return {
        "source": "frozen_latent_head",
        "variant": alias,
        "selection_basis": selection_basis,
        "uses_validation_for_selection": selection_basis.startswith("selected_by_validation"),
        "uses_test_for_selection": selection_basis.startswith("posthoc") or selection_basis.startswith("reconstruction_best_test"),
        "feature_source": "frozen_reconstruction_latent",
        "input_dim": int(row["latent_dim"]),
        "latent_dim": int(row["latent_dim"]),
        "compression_ratio": float(row["compression_ratio"]),
        "test_accuracy": float(row["accuracy"]),
        "test_balanced_accuracy": float(row["balanced_accuracy"]),
        "test_macro_f1": float(row["macro_f1"]),
        "test_mcc": float(row["mcc"]),
        "test_log_loss": float(row["log_loss"]),
        "test_opposite_direction_rate": float(row["opposite_direction_rate"]),
        "underlying_representation_variant": representation_variant,
    }


def build_fair_comparison(
    step8_fair: pd.DataFrame,
    latent_metrics: pd.DataFrame,
    validation_variant: str,
    test_variant: str,
    reconstruction_variant: str,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    raw_variants = [
        "majority",
        "raw_window_logistic_untuned",
        "raw_window_logistic_tuned",
        "raw_window_logistic_test_oracle",
    ]
    test_step8 = step8_fair[step8_fair["split"] == "test"].copy()
    for variant in raw_variants:
        src = test_step8[test_step8["variant"] == variant].iloc[0]
        rows.append(
            {
                "source": src["source"],
                "variant": variant,
                "selection_basis": src["selection_basis"],
                "uses_validation_for_selection": bool(src["uses_validation_tuning"]),
                "uses_test_for_selection": variant == "raw_window_logistic_test_oracle",
                "feature_source": src["feature_source"],
                "input_dim": int(src["input_dim"]) if not pd.isna(src["input_dim"]) else np.nan,
                "latent_dim": np.nan,
                "compression_ratio": np.nan,
                "test_accuracy": float(src["accuracy"]),
                "test_balanced_accuracy": float(src["balanced_accuracy"]),
                "test_macro_f1": float(src["macro_f1"]),
                "test_mcc": float(src["mcc"]),
                "test_log_loss": float(src["log_loss"]),
                "test_opposite_direction_rate": float(src["opposite_direction_rate"]),
                "underlying_representation_variant": "",
            }
        )

    rows.append(
        test_metric_row(
            latent_metrics,
            reconstruction_variant,
            "reconstruction_best_latent_head",
            "reconstruction_best_test_normalized_mse_from_step7",
        )
    )
    rows.append(
        test_metric_row(
            latent_metrics,
            validation_variant,
            "validation_selected_latent_head",
            "selected_by_validation_macro_f1_tie_mcc_log_loss_latent_dim_lexical",
        )
    )
    rows.append(
        test_metric_row(
            latent_metrics,
            test_variant,
            "test_posthoc_best_latent_head",
            "posthoc_best_test_macro_f1_from_step7",
        )
    )

    out = pd.DataFrame(rows)
    raw_tuned = out[out["variant"] == "raw_window_logistic_tuned"].iloc[0]
    raw_untuned = out[out["variant"] == "raw_window_logistic_untuned"].iloc[0]
    out["delta_macro_f1_vs_raw_tuned"] = out["test_macro_f1"].astype(float) - float(raw_tuned["test_macro_f1"])
    out["delta_mcc_vs_raw_tuned"] = out["test_mcc"].astype(float) - float(raw_tuned["test_mcc"])
    out["delta_macro_f1_vs_raw_untuned"] = out["test_macro_f1"].astype(float) - float(raw_untuned["test_macro_f1"])
    out["delta_mcc_vs_raw_untuned"] = out["test_mcc"].astype(float) - float(raw_untuned["test_mcc"])

    tags = {
        "majority": "raw_baseline",
        "raw_window_logistic_untuned": "raw_control_fixed",
        "raw_window_logistic_tuned": "raw_control_validation_selected",
        "raw_window_logistic_test_oracle": "test_oracle_reference",
        "reconstruction_best_latent_head": "reconstruction_reference",
        "validation_selected_latent_head": "fair_validation_selected_latent",
        "test_posthoc_best_latent_head": "test_posthoc_reference",
    }
    out["interpretation_tag"] = out["variant"].map(tags)
    return out[
        [
            "source",
            "variant",
            "selection_basis",
            "uses_validation_for_selection",
            "uses_test_for_selection",
            "feature_source",
            "input_dim",
            "latent_dim",
            "compression_ratio",
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_f1",
            "test_mcc",
            "test_log_loss",
            "test_opposite_direction_rate",
            "delta_macro_f1_vs_raw_tuned",
            "delta_mcc_vs_raw_tuned",
            "delta_macro_f1_vs_raw_untuned",
            "delta_mcc_vs_raw_untuned",
            "interpretation_tag",
            "underlying_representation_variant",
        ]
    ]


def prediction_frame_from_step5(step5_pred: pd.DataFrame, model: str, alias: str) -> pd.DataFrame:
    df = step5_pred[(step5_pred["split"] == "test") & (step5_pred["model"] == model)].copy()
    df["model_alias"] = alias
    return df


def prediction_frame_from_step7(step7_pred: pd.DataFrame, representation_variant: str, alias: str) -> pd.DataFrame:
    df = step7_pred[
        (step7_pred["split"] == "test") & (step7_pred["representation_variant"] == representation_variant)
    ].copy()
    df["model_alias"] = alias
    return df


def prediction_frame_from_step8(step8_pred: pd.DataFrame, alias: str) -> pd.DataFrame:
    df = step8_pred[step8_pred["split"] == "test"].copy()
    df["model_alias"] = alias
    return df


def paired_bootstrap_delta(
    frames: Dict[str, pd.DataFrame],
    comparisons: Sequence[tuple[str, str, str]],
    iterations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, object]] = []
    for comparison, model_a, model_b in comparisons:
        a = frames[model_a].sort_values("sample_id").reset_index(drop=True)
        b = frames[model_b].sort_values("sample_id").reset_index(drop=True)
        if len(a) == 0 or len(b) == 0:
            raise ValueError(f"Missing prediction rows for {comparison}.")
        if not np.array_equal(a["sample_id"].astype(int).to_numpy(), b["sample_id"].astype(int).to_numpy()):
            raise ValueError(f"Paired bootstrap sample_id mismatch for {comparison}.")
        n = len(a)
        for metric in BOOTSTRAP_METRICS:
            a_obs = metric_from_frame(a, metric)
            b_obs = metric_from_frame(b, metric)
            if metric == "opposite_direction_rate":
                observed = b_obs - a_obs
            else:
                observed = a_obs - b_obs
            deltas = np.empty(iterations, dtype=float)
            for i in range(iterations):
                idx = rng.integers(0, n, size=n)
                a_val = metric_from_frame(a.iloc[idx], metric)
                b_val = metric_from_frame(b.iloc[idx], metric)
                deltas[i] = b_val - a_val if metric == "opposite_direction_rate" else a_val - b_val
            rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "delta": float(observed),
                    "ci_2_5": float(np.nanpercentile(deltas, 2.5)),
                    "ci_97_5": float(np.nanpercentile(deltas, 97.5)),
                    "fraction_delta_gt_0": float(np.nanmean(deltas > 0)),
                    "n_test_samples": int(n),
                    "n_bootstrap": int(iterations),
                    "seed": int(seed),
                    "method": "paired_test_set_bootstrap; positive_delta_means_first_model_better",
                }
            )
    return pd.DataFrame(rows)


def build_manifest(
    args: argparse.Namespace,
    validation_variant: str,
    test_variant: str,
    reconstruction_variant: str,
    fair: pd.DataFrame,
    bootstrap_status: Dict[str, object],
) -> Dict[str, object]:
    test = fair.set_index("variant")
    validation_row = test.loc["validation_selected_latent_head"]
    raw_tuned = test.loc["raw_window_logistic_tuned"]
    same_variant = validation_variant == test_variant
    if same_variant:
        interpretation = (
            "The post hoc selection caveat is reduced because validation macro-F1 and test macro-F1 select "
            f"the same latent variant ({validation_variant}) in this run."
        )
    else:
        interpretation = (
            "The Step 8 latent advantage depends partly on test-time representation selection because validation "
            f"selects {validation_variant} while test macro-F1 selects {test_variant}."
        )
    return {
        "step": "step9_validation_selected_transfer_audit",
        "purpose": "Audit frozen latent transfer when representation selection is based only on validation split metrics.",
        "input_artifacts": {
            "latent_head_metrics": str(Path(args.step7_dir) / "latent_head_metrics.csv"),
            "latent_head_predictions": str(Path(args.step7_dir) / "latent_head_predictions.csv"),
            "step8_fair_transfer_comparison": str(Path(args.step8_dir) / "fair_transfer_comparison.csv"),
            "step8_raw_logistic_tuned_predictions": str(Path(args.step8_dir) / "raw_logistic_tuned_predictions.csv"),
            "step5_per_sample_predictions": str(Path(args.step5_dir) / "per_sample_predictions.csv"),
            "step7_rank_alignment": str(Path(args.step7_dir) / "model_level_rank_alignment.csv"),
        },
        "selection_policy": {
            "primary": "highest validation macro_f1",
            "tie_1": "highest validation mcc",
            "tie_2": "lowest validation log_loss",
            "tie_3": "smaller latent_dim",
            "tie_4": "lexical representation_variant",
            "test_metrics_used_for_selection": False,
        },
        "selected_results": {
            "validation_selected_variant": validation_variant,
            "test_posthoc_best_variant": test_variant,
            "reconstruction_best_variant": reconstruction_variant,
            "validation_selected_equals_test_posthoc_best": bool(same_variant),
            "validation_selected_test_macro_f1": float(validation_row["test_macro_f1"]),
            "raw_tuned_control_test_macro_f1": float(raw_tuned["test_macro_f1"]),
            "delta_macro_f1_vs_raw_tuned": float(validation_row["delta_macro_f1_vs_raw_tuned"]),
            "validation_selected_test_mcc": float(validation_row["test_mcc"]),
            "raw_tuned_control_test_mcc": float(raw_tuned["test_mcc"]),
            "delta_mcc_vs_raw_tuned": float(validation_row["delta_mcc_vs_raw_tuned"]),
        },
        "bootstrap": bootstrap_status,
        "interpretation": interpretation,
        "remaining_limits": [
            "single symbol",
            "single horizon trend5",
            "single stride-4 chronological subset",
            "candidate set fixed by earlier steps",
            "no multi-symbol robustness",
            "no multi-horizon robustness",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step5-dir", default="results/step5_prediction_baselines")
    parser.add_argument("--step7-dir", default="results/step7_alignment")
    parser.add_argument("--step8-dir", default="results/step8_fairness_robustness")
    parser.add_argument("--output-dir", default="results/step9_validation_selection_audit")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step5_dir = Path(args.step5_dir)
    step7_dir = Path(args.step7_dir)
    step8_dir = Path(args.step8_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    latent_metrics = pd.read_csv(step7_dir / "latent_head_metrics.csv")
    candidate_audit, validation_variant, test_variant = select_candidates(latent_metrics)
    reconstruction_variant = best_reconstruction_variant(step7_dir)
    candidate_audit.loc[
        candidate_audit["representation_variant"] == reconstruction_variant, "selection_note"
    ] = candidate_audit.loc[
        candidate_audit["representation_variant"] == reconstruction_variant, "selection_note"
    ].apply(lambda x: x if "reconstruction_best" in x else f"{x};reconstruction_best")

    step8_fair = pd.read_csv(step8_dir / "fair_transfer_comparison.csv")
    fair = build_fair_comparison(step8_fair, latent_metrics, validation_variant, test_variant, reconstruction_variant)

    bootstrap_status: Dict[str, object]
    try:
        step5_pred = pd.read_csv(step5_dir / "per_sample_predictions.csv")
        step7_pred = pd.read_csv(step7_dir / "latent_head_predictions.csv")
        step8_tuned_pred = pd.read_csv(step8_dir / "raw_logistic_tuned_predictions.csv")
        frames = {
            "validation_selected_latent_head": prediction_frame_from_step7(
                step7_pred, validation_variant, "validation_selected_latent_head"
            ),
            "raw_window_logistic_tuned": prediction_frame_from_step8(
                step8_tuned_pred, "raw_window_logistic_tuned"
            ),
            "raw_window_logistic_untuned": prediction_frame_from_step5(
                step5_pred, "logistic_regression", "raw_window_logistic_untuned"
            ),
            "reconstruction_best_latent_head": prediction_frame_from_step7(
                step7_pred, reconstruction_variant, "reconstruction_best_latent_head"
            ),
        }
        bootstrap = paired_bootstrap_delta(
            frames,
            [
                (
                    "validation_selected_latent_head_vs_raw_window_logistic_tuned",
                    "validation_selected_latent_head",
                    "raw_window_logistic_tuned",
                ),
                (
                    "validation_selected_latent_head_vs_raw_window_logistic_untuned",
                    "validation_selected_latent_head",
                    "raw_window_logistic_untuned",
                ),
                (
                    "validation_selected_latent_head_vs_reconstruction_best_latent_head",
                    "validation_selected_latent_head",
                    "reconstruction_best_latent_head",
                ),
            ],
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        bootstrap.to_csv(output_dir / "paired_bootstrap_delta.csv", index=False)
        bootstrap_status = {
            "status": "completed",
            "artifact": str(output_dir / "paired_bootstrap_delta.csv"),
            "n_bootstrap": int(args.bootstrap_iterations),
            "seed": int(args.seed),
        }
    except Exception as exc:
        bootstrap = pd.DataFrame(
            columns=[
                "comparison",
                "metric",
                "delta",
                "ci_2_5",
                "ci_97_5",
                "fraction_delta_gt_0",
                "n_test_samples",
                "n_bootstrap",
                "seed",
                "method",
            ]
        )
        bootstrap.to_csv(output_dir / "paired_bootstrap_delta.csv", index=False)
        bootstrap_status = {
            "status": "skipped",
            "reason": str(exc),
            "artifact": str(output_dir / "paired_bootstrap_delta.csv"),
        }

    candidate_audit.to_csv(output_dir / "candidate_selection_audit.csv", index=False)
    fair.to_csv(output_dir / "fair_transfer_comparison.csv", index=False)
    manifest = build_manifest(args, validation_variant, test_variant, reconstruction_variant, fair, bootstrap_status)
    with (output_dir / "step9_manifest.json").open("w") as f:
        json.dump(to_jsonable(manifest), f, indent=2)

    print(json.dumps(to_jsonable(manifest["selected_results"]), indent=2))


if __name__ == "__main__":
    main()
