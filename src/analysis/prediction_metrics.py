"""Metrics for multiclass mid-price trend prediction baselines."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


CLASS_ORDER: List[int] = [0, 1, 2]
CLASS_NAMES: Dict[int, str] = {0: "down", 1: "neutral", 2: "up"}


def _distribution(y: np.ndarray, class_order: Sequence[int]) -> Dict[str, int]:
    values, counts = np.unique(y, return_counts=True)
    count_map = {int(v): int(c) for v, c in zip(values, counts)}
    return {str(c): count_map.get(c, 0) for c in class_order}


def _row_normalized_confusion(cm: np.ndarray) -> np.ndarray:
    cm = cm.astype(float)
    row_sum = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
    return norm


def _directional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | None]:
    true_non_neutral = np.isin(y_true, [0, 2])
    pred_non_neutral = np.isin(y_pred, [0, 2])

    true_non_neutral_count = int(true_non_neutral.sum())
    pred_non_neutral_count = int(pred_non_neutral.sum())

    if true_non_neutral_count > 0:
        non_neutral_recall = float((pred_non_neutral[true_non_neutral]).mean())
        directional_accuracy_non_neutral = float((y_true[true_non_neutral] == y_pred[true_non_neutral]).mean())

        true_non_neutral_y = y_true[true_non_neutral]
        pred_non_neutral_y = y_pred[true_non_neutral]
        up_down_macro_f1 = float(
            f1_score(true_non_neutral_y, pred_non_neutral_y, labels=[0, 2], average="macro", zero_division=0)
        )

        opposite_hits = ((y_true == 0) & (y_pred == 2)) | ((y_true == 2) & (y_pred == 0))
        opposite_direction_rate = float(opposite_hits[true_non_neutral].mean())
    else:
        non_neutral_recall = None
        directional_accuracy_non_neutral = None
        up_down_macro_f1 = None
        opposite_direction_rate = None

    if pred_non_neutral_count > 0:
        non_neutral_precision = float((true_non_neutral[pred_non_neutral]).mean())
    else:
        non_neutral_precision = None

    return {
        "non_neutral_recall": non_neutral_recall,
        "non_neutral_precision": non_neutral_precision,
        "directional_accuracy_non_neutral": directional_accuracy_non_neutral,
        "up_down_macro_f1": up_down_macro_f1,
        "opposite_direction_rate": opposite_direction_rate,
    }


def compute_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_order: Sequence[int] = CLASS_ORDER,
) -> Dict[str, object]:
    """Compute required primary, diagnostic, and directional metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    class_order_set = set(class_order)

    if not set(np.unique(y_true)).issubset(class_order_set):
        raise ValueError("y_true contains labels outside class_order.")
    if not set(np.unique(y_pred)).issubset(class_order_set):
        raise ValueError("y_pred contains labels outside class_order.")

    if y_proba.ndim != 2 or y_proba.shape[1] != len(class_order):
        raise ValueError(f"y_proba must have shape (N, {len(class_order)}), got {y_proba.shape}")

    eps = 1e-12
    y_proba = np.clip(y_proba, eps, 1.0)
    y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(class_order),
        average=None,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(class_order))
    cm_norm = _row_normalized_confusion(cm)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(class_order), average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_proba, labels=list(class_order))),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=list(class_order), average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=list(class_order), average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=list(class_order), average="weighted", zero_division=0)
        ),
    }

    metrics.update(
        {
            "per_class_precision": {str(c): float(v) for c, v in zip(class_order, per_p)},
            "per_class_recall": {str(c): float(v) for c, v in zip(class_order, per_r)},
            "per_class_f1": {str(c): float(v) for c, v in zip(class_order, per_f1)},
            "support": {str(c): int(v) for c, v in zip(class_order, support)},
            "true_class_distribution": _distribution(y_true, class_order),
            "pred_class_distribution": _distribution(y_pred, class_order),
            "raw_confusion_matrix": cm.astype(int).tolist(),
            "row_normalized_confusion_matrix": cm_norm.tolist(),
        }
    )

    metrics.update(_directional_metrics(y_true, y_pred))

    return metrics
