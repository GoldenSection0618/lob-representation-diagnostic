"""Step 5 entrypoint: train and evaluate prediction-only baselines on Step 3 subset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.prediction_metrics import CLASS_NAMES, CLASS_ORDER, compute_prediction_metrics
from src.data.prediction_dataset import load_prediction_arrays
from src.models.prediction_baselines import (
    LogisticRegressionBaseline,
    MLPBaseline,
    MLPTrainConfig,
    MajorityBaseline,
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


def _class_distribution(y: np.ndarray) -> Dict[str, int]:
    vals, cnt = np.unique(y, return_counts=True)
    d = {int(v): int(c) for v, c in zip(vals, cnt)}
    return {str(c): d.get(c, 0) for c in CLASS_ORDER}


def _mean_confidence(proba: np.ndarray) -> float:
    return float(np.max(proba, axis=1).mean()) if len(proba) else float("nan")


def _build_per_sample_prediction_frame(
    sample_table: pd.DataFrame,
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> pd.DataFrame:
    if len(sample_table) != len(y_true):
        raise ValueError(
            f"sample table length mismatch for split={split_name}: "
            f"len(sample_table)={len(sample_table)} vs len(y_true)={len(y_true)}"
        )
    if y_proba.shape[1] != len(CLASS_ORDER):
        raise ValueError(f"Expected probability width {len(CLASS_ORDER)}, got {y_proba.shape[1]}")

    out = sample_table[["sample_id", "original_sample_id", "split", "label_row"]].copy()
    out["split"] = split_name
    out["y_true"] = y_true.astype(int)
    out["model"] = model_name
    out["y_pred"] = y_pred.astype(int)
    out["correct"] = out["y_pred"].to_numpy() == out["y_true"].to_numpy()
    out["confidence"] = np.max(y_proba, axis=1).astype(float)
    out["proba_0"] = y_proba[:, 0].astype(float)
    out["proba_1"] = y_proba[:, 1].astype(float)
    out["proba_2"] = y_proba[:, 2].astype(float)
    out["is_non_neutral_true"] = np.isin(out["y_true"].to_numpy(), [0, 2])
    out["is_non_neutral_pred"] = np.isin(out["y_pred"].to_numpy(), [0, 2])

    direction_correct = np.where(
        out["is_non_neutral_true"].to_numpy(),
        out["y_true"].to_numpy() == out["y_pred"].to_numpy(),
        np.nan,
    )
    out["direction_correct_non_neutral"] = direction_correct
    out["opposite_direction_error"] = (
        ((out["y_true"].to_numpy() == 0) & (out["y_pred"].to_numpy() == 2))
        | ((out["y_true"].to_numpy() == 2) & (out["y_pred"].to_numpy() == 0))
    )
    return out


def _build_model(name: str, args, device: str):
    if name == "majority":
        return MajorityBaseline(class_order=CLASS_ORDER)
    if name == "logistic_regression":
        return LogisticRegressionBaseline(random_state=args.seed, max_iter=2000)
    if name == "mlp":
        cfg = MLPTrainConfig(
            random_state=args.seed,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            patience=10,
            lr=1e-3,
            weight_decay=1e-4,
            dropout=0.2,
            device=device,
        )
        return MLPBaseline(config=cfg)
    raise ValueError(f"Unsupported model name: {name}")


def _select_best_model(test_rows: List[Dict[str, object]]) -> str:
    df = pd.DataFrame(test_rows)
    sort_df = df.sort_values(by=["macro_f1", "mcc", "log_loss"], ascending=[False, False, True])
    return str(sort_df.iloc[0]["model"])


def _plot_primary_metrics(metrics_df: pd.DataFrame, fig_path: Path) -> None:
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    models = test_df["model"].tolist()
    metrics = ["macro_f1", "balanced_accuracy", "mcc", "accuracy"]
    x = np.arange(len(models))
    width = 0.18

    plt.figure(figsize=(10, 5))
    for i, m in enumerate(metrics):
        plt.bar(x + (i - 1.5) * width, test_df[m].to_numpy(), width=width, label=m)

    plt.xticks(x, models, rotation=15)
    min_val = float(test_df[metrics].min().min())
    lower = min(0.0, min_val - 0.05)
    plt.ylim(lower, 1.0)
    plt.axhline(0.0, linewidth=0.8)
    plt.ylabel("score")
    plt.title("Primary Metrics by Model (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def _plot_class_distribution(true_dist, pred_dist_by_model, fig_path: Path) -> None:
    classes = [CLASS_NAMES[c] for c in CLASS_ORDER]
    true_vals = [true_dist[str(c)] for c in CLASS_ORDER]

    series = [("true", true_vals)]
    for model, dist in pred_dist_by_model.items():
        series.append((f"pred:{model}", [dist[str(c)] for c in CLASS_ORDER]))

    x = np.arange(len(classes))
    width = 0.8 / len(series)

    plt.figure(figsize=(10, 5))
    for i, (name, vals) in enumerate(series):
        plt.bar(x - 0.4 + width / 2 + i * width, vals, width=width, label=name)

    plt.xticks(x, classes)
    plt.ylabel("count")
    plt.title("True vs Predicted Class Distribution (Test)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def _plot_confusion_best(best_model: str, cm_raw, cm_norm, fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.array(cm_norm), cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_labels = [CLASS_NAMES[c] for c in CLASS_ORDER]
    ax.set_xticks(np.arange(len(CLASS_ORDER)), labels=tick_labels)
    ax.set_yticks(np.arange(len(CLASS_ORDER)), labels=tick_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Normalized Confusion Matrix (Best: {best_model})")

    raw = np.array(cm_raw)
    norm = np.array(cm_norm)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            ax.text(j, i, f"{norm[i,j]*100:.1f}%\n({raw[i,j]})", ha="center", va="center", color="black", fontsize=9)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def _plot_directional(metrics_df: pd.DataFrame, fig_path: Path) -> None:
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    models = test_df["model"].tolist()
    metrics = [
        "non_neutral_recall",
        "non_neutral_precision",
        "directional_accuracy_non_neutral",
        "up_down_macro_f1",
        "opposite_direction_rate",
    ]

    x = np.arange(len(models))
    width = 0.15

    plt.figure(figsize=(11, 5))
    for i, m in enumerate(metrics):
        vals = test_df[m].astype(float).to_numpy()
        plt.bar(x + (i - 2) * width, vals, width=width, label=m)

    plt.xticks(x, models, rotation=15)
    plt.ylabel("score / rate")
    plt.title("Directional Error Summary (Test) - opposite_direction_rate: lower is better")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def _plot_log_loss(metrics_df: pd.DataFrame, fig_path: Path) -> None:
    test_df = metrics_df[metrics_df["split"] == "test"].copy()
    models = test_df["model"].tolist()
    vals = test_df["log_loss"].to_numpy()

    plt.figure(figsize=(8, 4))
    plt.bar(models, vals)
    plt.ylabel("log_loss (lower is better)")
    plt.title("Log Loss by Model (Test)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Step 5 prediction-only baselines")
    parser.add_argument("--subset-dir", default="data/processed/minimal_subset")
    parser.add_argument("--output-dir", default="results/step5_prediction_baselines")
    parser.add_argument("--figures-dir", default="figures/step5_prediction_baselines")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", default="majority,logistic_regression,mlp")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    subset = load_prediction_arrays(args.subset_dir)
    X_train, y_train = subset["train"]
    X_val, y_val = subset["val"]
    X_test, y_test = subset["test"]

    print("=== Step 5 Subset Split Sizes ===")
    print(f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}")
    print(f"X_train={X_train.shape}, X_val={X_val.shape}, X_test={X_test.shape}")

    selected_models = [m.strip() for m in args.models.split(",") if m.strip()]

    output_dir = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows: List[Dict[str, object]] = []
    per_sample_parts: List[pd.DataFrame] = []
    classification_report: Dict[str, Dict[str, object]] = {}
    directional_report: Dict[str, Dict[str, object]] = {}
    confusion_report: Dict[str, Dict[str, object]] = {}
    pred_dist_report: Dict[str, object] = {
        "true_distribution": {
            "train": _class_distribution(y_train),
            "val": _class_distribution(y_val),
            "test": _class_distribution(y_test),
        },
        "predicted_distribution": {},
        "probability_summary": {},
    }

    eval_sets = {
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    for model_name in selected_models:
        model = _build_model(model_name, args=args, device=device)

        if model_name == "mlp":
            model.fit(X_train, y_train, X_val, y_val)
        else:
            model.fit(X_train, y_train)

        classification_report[model_name] = {}
        directional_report[model_name] = {}
        confusion_report[model_name] = {}
        pred_dist_report["predicted_distribution"][model_name] = {}
        pred_dist_report["probability_summary"][model_name] = {}

        for split_name, (X_split, y_split) in eval_sets.items():
            y_pred = model.predict(X_split)
            y_proba = model.predict_proba(X_split)
            m = compute_prediction_metrics(y_split, y_pred, y_proba, class_order=CLASS_ORDER)
            per_sample_parts.append(
                _build_per_sample_prediction_frame(
                    sample_table=subset["sample_tables"][split_name],
                    model_name=model_name,
                    split_name=split_name,
                    y_true=y_split,
                    y_pred=y_pred,
                    y_proba=y_proba,
                )
            )

            metrics_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "accuracy": m["accuracy"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "macro_f1": m["macro_f1"],
                    "mcc": m["mcc"],
                    "log_loss": m["log_loss"],
                    "weighted_f1": m["weighted_f1"],
                    "macro_precision": m["macro_precision"],
                    "macro_recall": m["macro_recall"],
                    "non_neutral_recall": m["non_neutral_recall"],
                    "non_neutral_precision": m["non_neutral_precision"],
                    "directional_accuracy_non_neutral": m["directional_accuracy_non_neutral"],
                    "up_down_macro_f1": m["up_down_macro_f1"],
                    "opposite_direction_rate": m["opposite_direction_rate"],
                }
            )

            classification_report[model_name][split_name] = {
                "per_class_precision": m["per_class_precision"],
                "per_class_recall": m["per_class_recall"],
                "per_class_f1": m["per_class_f1"],
                "macro_precision": m["macro_precision"],
                "macro_recall": m["macro_recall"],
                "weighted_f1": m["weighted_f1"],
                "support": m["support"],
                "true_class_distribution": m["true_class_distribution"],
                "pred_class_distribution": m["pred_class_distribution"],
            }

            directional_report[model_name][split_name] = {
                "non_neutral_recall": m["non_neutral_recall"],
                "non_neutral_precision": m["non_neutral_precision"],
                "directional_accuracy_non_neutral": m["directional_accuracy_non_neutral"],
                "up_down_macro_f1": m["up_down_macro_f1"],
                "opposite_direction_rate": m["opposite_direction_rate"],
            }

            confusion_report[model_name][split_name] = {
                "raw_confusion_matrix": m["raw_confusion_matrix"],
                "row_normalized_confusion_matrix": m["row_normalized_confusion_matrix"],
                "class_order": CLASS_ORDER,
                "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
            }

            pred_dist_report["predicted_distribution"][model_name][split_name] = m["pred_class_distribution"]
            pred_dist_report["probability_summary"][model_name][split_name] = {
                "mean_confidence": _mean_confidence(y_proba)
            }

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    per_sample_df = pd.concat(per_sample_parts, ignore_index=True)
    per_sample_df.to_csv(output_dir / "per_sample_predictions.csv", index=False)

    (output_dir / "classification_report.json").write_text(
        json.dumps(to_jsonable(classification_report), indent=2), encoding="utf-8"
    )
    (output_dir / "directional_metrics.json").write_text(
        json.dumps(to_jsonable(directional_report), indent=2), encoding="utf-8"
    )
    (output_dir / "confusion_matrices.json").write_text(
        json.dumps(to_jsonable(confusion_report), indent=2), encoding="utf-8"
    )
    (output_dir / "prediction_distributions.json").write_text(
        json.dumps(to_jsonable(pred_dist_report), indent=2), encoding="utf-8"
    )

    run_config = {
        "subset_dir": str(Path(args.subset_dir).resolve()),
        "output_dir": str(output_dir.resolve()),
        "figures_dir": str(figures_dir.resolve()),
        "seed": args.seed,
        "selected_models": selected_models,
        "model_hyperparameters": {
            "majority": {},
            "logistic_regression": {"max_iter": 2000, "class_weight": "balanced"},
            "mlp": {
                "max_epochs": args.max_epochs,
                "batch_size": args.batch_size,
                "patience": 10,
                "hidden": [256, 64],
                "dropout": 0.2,
                "class_weight": "train_distribution_inverse_freq",
            },
        },
        "class_order": CLASS_ORDER,
        "class_names": {str(k): v for k, v in CLASS_NAMES.items()},
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "step3_metadata_summary": subset["metadata"].get("summary", {}),
        "step4_protocol_note": "boundary-purged chronological split",
        "per_sample_prediction_output": {
            "file": "per_sample_predictions.csv",
            "splits": ["val", "test"],
            "join_keys_for_step7": ["sample_id", "split"],
            "prediction_model_column": "model",
            "prediction_model_column_note": (
                "In Step 5 outputs, model means prediction model. "
                "Rename to prediction_model before joining with Step 6."
            ),
            "direction_correct_non_neutral_semantics": (
                "1.0/0.0 for true non-neutral samples; null for neutral samples"
            ),
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(to_jsonable(run_config), indent=2), encoding="utf-8")

    test_rows = metrics_df[metrics_df["split"] == "test"].to_dict(orient="records")
    best_model = _select_best_model(test_rows)

    _plot_primary_metrics(metrics_df, figures_dir / "primary_metrics_by_model.png")
    _plot_class_distribution(
        true_dist=pred_dist_report["true_distribution"]["test"],
        pred_dist_by_model={m: pred_dist_report["predicted_distribution"][m]["test"] for m in selected_models},
        fig_path=figures_dir / "class_distribution_true_vs_pred.png",
    )

    best_cm_raw = confusion_report[best_model]["test"]["raw_confusion_matrix"]
    best_cm_norm = confusion_report[best_model]["test"]["row_normalized_confusion_matrix"]
    _plot_confusion_best(
        best_model=best_model,
        cm_raw=best_cm_raw,
        cm_norm=best_cm_norm,
        fig_path=figures_dir / "confusion_matrix_best_model_normalized.png",
    )
    _plot_directional(metrics_df, figures_dir / "directional_error_summary.png")
    _plot_log_loss(metrics_df, figures_dir / "log_loss_by_model.png")

    majority_test = metrics_df[(metrics_df["model"] == "majority") & (metrics_df["split"] == "test")].iloc[0]

    lines = []
    lines.append("# Step 5 Prediction-Only Baseline Summary")
    lines.append("")
    lines.append("## Split Sizes")
    lines.append(f"- train: {len(y_train)}")
    lines.append(f"- val: {len(y_val)}")
    lines.append(f"- test: {len(y_test)}")
    lines.append("")
    lines.append("## Best Test Model")
    best_row = metrics_df[(metrics_df["model"] == best_model) & (metrics_df["split"] == "test")].iloc[0]
    lines.append(f"- best model by test macro_f1 tie-broken by mcc then log_loss: `{best_model}`")
    lines.append(
        f"- test macro_f1={best_row['macro_f1']:.6f}, balanced_accuracy={best_row['balanced_accuracy']:.6f}, "
        f"mcc={best_row['mcc']:.6f}, log_loss={best_row['log_loss']:.6f}"
    )
    lines.append("")
    lines.append("## Probability Quality Warning")
    lines.append(
        f"- {best_model} has the best test macro-F1, but its test log_loss is high "
        f"({best_row['log_loss']:.6f}). This suggests poor probability calibration or overconfident errors."
    )
    lines.append("- It should not be treated as the best calibrated predictor.")
    lines.append("")
    lines.append("## Majority Baseline Comparison (Test)")
    for model in selected_models:
        if model == "majority":
            continue
        row = metrics_df[(metrics_df["model"] == model) & (metrics_df["split"] == "test")].iloc[0]
        lines.append(
            f"- {model}: "
            f"macro_f1 {'beat' if row['macro_f1'] > majority_test['macro_f1'] else 'not beat'} majority, "
            f"balanced_accuracy {'beat' if row['balanced_accuracy'] > majority_test['balanced_accuracy'] else 'not beat'} majority, "
            f"mcc {'beat' if row['mcc'] > majority_test['mcc'] else 'not beat'} majority"
        )

    lines.append("")
    lines.append("## Class-Coverage and Collapse Check")
    min_class_ratio = 0.02
    for model in selected_models:
        dist = pred_dist_report["predicted_distribution"][model]["test"]
        total = sum(dist.values())
        neutral_ratio = dist.get("1", 0) / total if total > 0 else 0.0
        missing_classes = [c for c, count in dist.items() if count == 0]
        low_coverage_classes = [
            c for c, count in dist.items() if total > 0 and (count / total) < min_class_ratio
        ]
        directional_row = metrics_df[(metrics_df["model"] == model) & (metrics_df["split"] == "test")].iloc[0]
        directional_collapse = (
            (directional_row["non_neutral_recall"] is not None and directional_row["non_neutral_recall"] < 0.2)
            or (directional_row["up_down_macro_f1"] is not None and directional_row["up_down_macro_f1"] < 0.2)
        )
        class_name = {str(k): v for k, v in CLASS_NAMES.items()}
        missing_named = [class_name.get(c, c) for c in missing_classes]
        low_cov_named = [class_name.get(c, c) for c in low_coverage_classes]
        dist_text = f"down={dist.get('0', 0)}, neutral={dist.get('1', 0)}, up={dist.get('2', 0)}"
        risk_notes = []
        if neutral_ratio > 0.90:
            risk_notes.append("neutral collapse risk")
        if missing_named:
            risk_notes.append(f"missing predicted classes: {', '.join(missing_named)}")
        if low_cov_named:
            risk_notes.append(f"low-coverage classes(<{min_class_ratio:.0%}): {', '.join(low_cov_named)}")
        if directional_collapse:
            risk_notes.append("directional collapse risk")
        if not risk_notes:
            risk_notes.append("no severe class-collapse signal")
        lines.append(
            f"- {model}: neutral prediction ratio={neutral_ratio:.4f}; "
            f"predicted distribution: {dist_text}; "
            f"{'; '.join(risk_notes)}"
        )

    lines.append("")
    lines.append("## Protocol Scope")
    lines.append("- Step 5 does not use reconstruction models.")
    lines.append("- Step 5 does not use randomized split protocols.")
    lines.append("- Step 5 does not use plain non-purged chronological split.")
    lines.append("- Per-sample prediction outputs are saved for Step 7 alignment in `per_sample_predictions.csv`.")
    lines.append(
        "- Per-sample prediction outputs cover val/test only. For Step 7, join predictions with "
        "reconstruction diagnostics on `sample_id` and `split`; treat Step 5 `model` as `prediction_model`."
    )
    lines.append(
        "- `direction_correct_non_neutral` is encoded as 1.0/0.0 for true non-neutral samples "
        "and null for neutral samples."
    )

    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== Step 5 Prediction Baselines Complete ===")
    print(f"models: {selected_models}")
    print(f"best test model: {best_model}")
    print(f"results: {output_dir.resolve()}")
    print(f"figures: {figures_dir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
