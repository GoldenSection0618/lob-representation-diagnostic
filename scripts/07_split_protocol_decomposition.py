"""Step 10: split protocol decomposition diagnostic."""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.prediction_metrics import compute_prediction_metrics


C_GRID = [0.01, 0.1, 1.0, 10.0]
CLASS_ORDER = [0, 1, 2]
PRIMARY_METRIC = "macro_f1"
PCA_DIMS = [32, 128]
MODEL_ORDER = [
    "majority",
    "raw_window_logistic_untuned",
    "raw_window_logistic_tuned",
    "raw_window_logistic_test_oracle",
    "reconstruction_best_latent_head",
    "validation_selected_latent_head",
    "test_posthoc_best_latent_head",
]


@dataclass
class SplitRun:
    run_id: str
    split_protocol: str
    seed: int
    is_deterministic: bool
    assignments: pd.DataFrame
    purge_policy: str
    randomization_policy: str
    block_policy: str
    block_size: int | None
    embargo_size: int
    notes: str


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


def parse_seeds(text: str) -> List[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        metrics = compute_prediction_metrics(y_true, y_pred, y_proba)
    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "mcc": float(metrics["mcc"]),
        "log_loss": float(metrics["log_loss"]),
        "opposite_direction_rate": float(metrics["opposite_direction_rate"]),
    }


def aligned_proba(clf, x: np.ndarray) -> np.ndarray:
    raw = clf.predict_proba(x)
    out = np.zeros((len(x), len(CLASS_ORDER)), dtype=float)
    for i, cls in enumerate(clf.classes_):
        out[:, int(cls)] = raw[:, i]
    eps = 1e-12
    out = np.clip(out, eps, 1.0)
    return out / out.sum(axis=1, keepdims=True)


def labels_distribution(y: np.ndarray) -> str:
    counts = {str(cls): int((y == cls).sum()) for cls in CLASS_ORDER}
    return json.dumps(counts, sort_keys=True)


def split_indices(assignments: pd.DataFrame) -> Dict[str, np.ndarray]:
    return {
        split: assignments.loc[assignments["split"] == split, "sample_id"].astype(int).to_numpy()
        for split in ["train", "val", "test"]
    }


def target_counts(n: int) -> Tuple[int, int, int]:
    # Match the current conservative baseline counts on the 7952-sample universe.
    n_train = int(round(n * 0.704225352112676))
    n_val = int(round(n * 0.15090543259557345))
    n_train = min(max(n_train, 1), n - 2)
    n_val = min(max(n_val, 1), n - n_train - 1)
    return n_train, n_val, n - n_train - n_val


def chronological_purged(samples: pd.DataFrame) -> pd.DataFrame:
    return samples[samples["split"].isin(["train", "val", "test"])][["sample_id", "split"]].copy()


def chronological_no_purge(samples: pd.DataFrame) -> pd.DataFrame:
    n = len(samples)
    n_train, n_val, _ = target_counts(n)
    ordered = samples.sort_values("label_row").reset_index(drop=True)[["sample_id"]].copy()
    split = np.empty(n, dtype=object)
    split[:n_train] = "train"
    split[n_train : n_train + n_val] = "val"
    split[n_train + n_val :] = "test"
    ordered["split"] = split
    return ordered


def random_window_naive(samples: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(samples)
    n_train, n_val, _ = target_counts(n)
    perm = rng.permutation(samples["sample_id"].astype(int).to_numpy())
    rows = []
    for split, ids in [
        ("train", perm[:n_train]),
        ("val", perm[n_train : n_train + n_val]),
        ("test", perm[n_train + n_val :]),
    ]:
        rows.append(pd.DataFrame({"sample_id": ids, "split": split}))
    return pd.concat(rows, ignore_index=True)


def random_block_purged(samples: pd.DataFrame, seed: int, block_size: int, embargo_size: int) -> pd.DataFrame:
    ordered = samples.sort_values("label_row").reset_index(drop=True)[["sample_id"]].copy()
    ordered["pos"] = np.arange(len(ordered))
    ordered["block_id"] = ordered["pos"] // block_size
    blocks = ordered.groupby("block_id").size().reset_index(name="n")
    rng = np.random.default_rng(seed)
    shuffled = blocks.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_train, n_val, _ = target_counts(len(ordered))
    block_split: Dict[int, str] = {}
    counts = {"train": 0, "val": 0, "test": 0}
    for _, row in shuffled.iterrows():
        bid = int(row["block_id"])
        size = int(row["n"])
        deficits = {
            "train": n_train - counts["train"],
            "val": n_val - counts["val"],
            "test": (len(ordered) - n_train - n_val) - counts["test"],
        }
        split = max(deficits, key=deficits.get)
        if deficits[split] <= 0:
            split = min(counts, key=counts.get)
        block_split[bid] = split
        counts[split] += size
    ordered["split"] = ordered["block_id"].map(block_split)

    drop = np.zeros(len(ordered), dtype=bool)
    split_values = ordered["split"].to_numpy()
    for i in range(len(ordered) - 1):
        if split_values[i] != split_values[i + 1]:
            lo = max(0, i - embargo_size + 1)
            hi = min(len(ordered), i + embargo_size + 1)
            drop[lo:hi] = True
    kept = ordered.loc[~drop, ["sample_id", "split"]].copy()
    # If one split became too small due to an unlucky block arrangement, retry deterministically with a derived shuffle.
    if any((kept["split"] == split).sum() < 100 for split in ["train", "val", "test"]):
        return random_block_purged(samples, int(rng.integers(0, 1_000_000)), block_size, embargo_size)
    return kept


def build_split_runs(samples: pd.DataFrame, seeds: Sequence[int], block_size: int, embargo_size: int) -> List[SplitRun]:
    runs = [
        SplitRun(
            run_id="chronological_purged",
            split_protocol="chronological_purged",
            seed=0,
            is_deterministic=True,
            assignments=chronological_purged(samples),
            purge_policy="Step 3 boundary purge already applied",
            randomization_policy="none",
            block_policy="none",
            block_size=None,
            embargo_size=0,
            notes="Conservative baseline protocol used by Steps 3-9.",
        ),
        SplitRun(
            run_id="chronological_no_purge",
            split_protocol="chronological_no_purge",
            seed=0,
            is_deterministic=True,
            assignments=chronological_no_purge(samples),
            purge_policy="no additional boundary purge on current kept sample universe",
            randomization_policy="none",
            block_policy="none",
            block_size=None,
            embargo_size=0,
            notes="Diagnostic approximation on the existing Step 3 kept sample universe; dropped Step 3 boundary samples are not restored.",
        ),
    ]
    for seed in seeds:
        runs.append(
            SplitRun(
                run_id=f"random_window_naive_seed{seed}",
                split_protocol="random_window_naive",
                seed=int(seed),
                is_deterministic=False,
                assignments=random_window_naive(samples, int(seed)),
                purge_policy="none",
                randomization_policy="sample-level random train/val/test split",
                block_policy="none",
                block_size=None,
                embargo_size=0,
                notes="Optimistic leakage-prone random window split.",
            )
        )
        runs.append(
            SplitRun(
                run_id=f"random_block_purged_seed{seed}",
                split_protocol="random_block_purged",
                seed=int(seed),
                is_deterministic=False,
                assignments=random_block_purged(samples, int(seed), block_size, embargo_size),
                purge_policy="embargo around adjacent block split transitions",
                randomization_policy="contiguous block-level random assignment",
                block_policy="contiguous sample-index blocks",
                block_size=block_size,
                embargo_size=embargo_size,
                notes="Blocked random control for separating randomization from near-neighbor exposure.",
            )
        )
    return runs


def min_abs_gap(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a = np.sort(a)
    b = np.sort(b)
    idx = np.searchsorted(a, b)
    vals = []
    valid = idx < len(a)
    vals.extend(np.abs(a[idx[valid]] - b[valid]).tolist())
    valid = idx > 0
    vals.extend(np.abs(a[idx[valid] - 1] - b[valid]).tolist())
    return float(np.min(vals)) if vals else float("nan")


def nearest_abs_distance(train_values: np.ndarray, query_values: np.ndarray) -> np.ndarray:
    train_values = np.sort(train_values)
    idx = np.searchsorted(train_values, query_values)
    out = np.full(len(query_values), np.inf, dtype=float)
    valid = idx < len(train_values)
    out[valid] = np.minimum(out[valid], np.abs(train_values[idx[valid]] - query_values[valid]))
    valid = idx > 0
    out[valid] = np.minimum(out[valid], np.abs(train_values[idx[valid] - 1] - query_values[valid]))
    return out


def split_integrity(run: SplitRun, samples: pd.DataFrame, y: np.ndarray, window_len: int, sample_stride: int) -> Dict[str, object]:
    merged = run.assignments.merge(
        samples.drop(columns=["split"], errors="ignore"),
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    parts = {split: merged[merged["split"] == split].copy() for split in ["train", "val", "test"]}
    train_ids = parts["train"]["sample_id"].astype(int).to_numpy()
    train_label = parts["train"]["label_row"].astype(int).to_numpy()
    train_original = parts["train"]["original_sample_id"].astype(int).to_numpy()
    row = {
        "run_id": run.run_id,
        "split_protocol": run.split_protocol,
        "seed": run.seed,
        "n_train": len(parts["train"]),
        "n_val": len(parts["val"]),
        "n_test": len(parts["test"]),
        "train_label_distribution": labels_distribution(y[train_ids]),
        "val_label_distribution": labels_distribution(y[parts["val"]["sample_id"].astype(int).to_numpy()]),
        "test_label_distribution": labels_distribution(y[parts["test"]["sample_id"].astype(int).to_numpy()]),
        "min_train_val_label_row_gap": min_abs_gap(train_label, parts["val"]["label_row"].astype(int).to_numpy()),
        "min_val_test_label_row_gap": min_abs_gap(parts["val"]["label_row"].astype(int).to_numpy(), parts["test"]["label_row"].astype(int).to_numpy()),
        "min_train_test_label_row_gap": min_abs_gap(train_label, parts["test"]["label_row"].astype(int).to_numpy()),
    }

    def fractions(split: str) -> Dict[str, float]:
        query = parts[split]
        if len(query) == 0:
            return {
                f"fraction_{split}_with_overlapping_train_window": float("nan"),
                f"fraction_{split}_with_near_neighbor_train_k1": float("nan"),
                f"fraction_{split}_with_near_neighbor_train_k5": float("nan"),
                f"fraction_{split}_with_near_neighbor_train_k25": float("nan"),
            }
        label_dist = nearest_abs_distance(train_label, query["label_row"].astype(int).to_numpy())
        sample_dist = nearest_abs_distance(train_original, query["original_sample_id"].astype(int).to_numpy())
        return {
            f"fraction_{split}_with_overlapping_train_window": float(np.mean(label_dist <= (window_len - 1))),
            f"fraction_{split}_with_near_neighbor_train_k1": float(np.mean(sample_dist <= 1)),
            f"fraction_{split}_with_near_neighbor_train_k5": float(np.mean(sample_dist <= 5)),
            f"fraction_{split}_with_near_neighbor_train_k25": float(np.mean(sample_dist <= 25)),
        }

    row.update(fractions("val"))
    row.update(fractions("test"))
    test_original = parts["test"]["original_sample_id"].astype(int).to_numpy()
    test_dist = nearest_abs_distance(train_original, test_original)
    row["fraction_test_with_same_label_row_in_train"] = float(np.mean(test_dist == 0)) if len(test_dist) else float("nan")
    row["fraction_test_with_adjacent_label_row_in_train"] = float(np.mean(test_dist == 1)) if len(test_dist) else float("nan")
    row["boundary_purge_applied"] = run.split_protocol == "chronological_purged"
    row["embargo_applied"] = run.embargo_size > 0
    overlap_risk = row["fraction_test_with_overlapping_train_window"]
    k5_risk = row["fraction_test_with_near_neighbor_train_k5"]
    if overlap_risk >= 0.5 or k5_risk >= 0.25:
        risk = "high"
    elif overlap_risk >= 0.05 or k5_risk >= 0.05:
        risk = "medium"
    else:
        risk = "low"
    row["near_neighbor_risk_level"] = risk
    row["integrity_interpretation"] = (
        f"{risk} train/test near-neighbor risk; test overlap={overlap_risk:.4f}, test k5={k5_risk:.4f}"
    )
    return row


def fit_logistic_grid(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Dict[str, object]:
    candidates = []
    for c in C_GRID:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=float(c),
                        class_weight="balanced",
                        max_iter=300,
                        solver="saga",
                        tol=1e-2,
                        random_state=seed,
                    ),
                ),
            ]
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            pipe.fit(x_train, y_train)
        val_pred = pipe.predict(x_val).astype(int)
        val_proba = aligned_proba(pipe.named_steps["clf"], pipe.named_steps["scaler"].transform(x_val))
        val_metrics = compute_metrics(y_val, val_pred, val_proba)
        test_pred = pipe.predict(x_test).astype(int)
        test_proba = aligned_proba(pipe.named_steps["clf"], pipe.named_steps["scaler"].transform(x_test))
        test_metrics = compute_metrics(y_test, test_pred, test_proba)
        candidates.append(
            {
                "C": float(c),
                "pipe": pipe,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
        )
    selected = max(candidates, key=lambda r: (r["val_metrics"]["macro_f1"], r["val_metrics"]["mcc"], -r["val_metrics"]["log_loss"], -r["C"]))
    test_oracle = max(candidates, key=lambda r: (r["test_metrics"]["macro_f1"], r["test_metrics"]["mcc"], -r["test_metrics"]["log_loss"], -r["C"]))
    c1 = next(r for r in candidates if math.isclose(r["C"], 1.0))
    return {"selected": selected, "test_oracle": test_oracle, "c1": c1, "candidates": candidates}


def majority_metrics(y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    values, counts = np.unique(y_train, return_counts=True)
    majority = int(values[np.argmax(counts)])
    y_pred = np.full(len(y_test), majority, dtype=int)
    prior = np.zeros(3, dtype=float)
    for cls in CLASS_ORDER:
        prior[cls] = max(float((y_train == cls).mean()), 1e-12)
    prior = prior / prior.sum()
    y_proba = np.tile(prior, (len(y_test), 1))
    return compute_metrics(y_test, y_pred, y_proba)


def reconstruct_last_snapshot_scaled(x_raw: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    repeated = np.repeat(x_raw[:, -1:, :], x_raw.shape[1], axis=1).reshape(len(x_raw), -1)
    return scaler.transform(repeated)


def train_and_evaluate_run(run: SplitRun, samples: pd.DataFrame, x: np.ndarray, y: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = split_indices(run.assignments)
    x_flat = x.reshape(len(x), -1)
    x_train, x_val, x_test = x_flat[idx["train"]], x_flat[idx["val"]], x_flat[idx["test"]]
    y_train, y_val, y_test = y[idx["train"]], y[idx["val"]], y[idx["test"]]

    rows: List[Dict[str, object]] = []
    majority = majority_metrics(y_train, y_test)
    rows.append(
        base_perf_row(
            run,
            "baseline",
            "majority",
            "raw_window_flattened",
            4000,
            np.nan,
            np.nan,
            "train_majority_class",
            False,
            False,
            False,
            majority,
            len(y_train),
            len(y_val),
            len(y_test),
        )
    )

    raw = fit_logistic_grid(x_train, y_train, x_val, y_val, x_test, y_test, run.seed)
    rows.append(
        base_perf_row(run, "raw_window_logistic", "raw_window_logistic_untuned", "raw_window_flattened", 4000, np.nan, np.nan, "fixed_C_1", False, False, False, raw["c1"]["test_metrics"], len(y_train), len(y_val), len(y_test))
    )
    rows.append(
        base_perf_row(run, "raw_window_logistic", "raw_window_logistic_tuned", "raw_window_flattened", 4000, np.nan, np.nan, f"validation_macro_f1_tie_mcc_log_loss_C={raw['selected']['C']:g}", True, False, False, raw["selected"]["test_metrics"], len(y_train), len(y_val), len(y_test))
    )
    rows.append(
        base_perf_row(run, "raw_window_logistic", "raw_window_logistic_test_oracle", "raw_window_flattened", 4000, np.nan, np.nan, f"test_macro_f1_oracle_C={raw['test_oracle']['C']:g}", False, True, True, raw["test_oracle"]["test_metrics"], len(y_train), len(y_val), len(y_test))
    )

    raw_scaler = StandardScaler().fit(x_train)
    scaled_train = raw_scaler.transform(x_train)
    scaled_val = raw_scaler.transform(x_val)
    scaled_test = raw_scaler.transform(x_test)
    variant_records: List[Dict[str, object]] = []

    last_latents = {
        "train": x[idx["train"], -1, :],
        "val": x[idx["val"], -1, :],
        "test": x[idx["test"], -1, :],
    }
    last_recon_test = reconstruct_last_snapshot_scaled(x[idx["test"]], raw_scaler)
    variant_records.append(
        evaluate_representation(
            run,
            "last_snapshot_repeat",
            "last_snapshot_repeat@40",
            40,
            100.0,
            last_latents,
            y_train,
            y_val,
            y_test,
            scaled_test,
            last_recon_test,
        )
    )

    for dim in PCA_DIMS:
        pca = PCA(n_components=dim, svd_solver="randomized", random_state=run.seed)
        train_latent = pca.fit_transform(scaled_train)
        val_latent = pca.transform(scaled_val)
        test_latent = pca.transform(scaled_test)
        recon_test = pca.inverse_transform(test_latent)
        variant_records.append(
            evaluate_representation(
                run,
                "pca",
                f"pca@{dim}",
                dim,
                4000.0 / float(dim),
                {"train": train_latent, "val": val_latent, "test": test_latent},
                y_train,
                y_val,
                y_test,
                scaled_test,
                recon_test,
            )
        )

    selection_df = pd.DataFrame(variant_records)
    validation = selection_df.sort_values(
        ["val_macro_f1", "val_mcc", "val_log_loss", "latent_dim", "model_or_variant"],
        ascending=[False, False, True, True, True],
    ).iloc[0]
    test_best = selection_df.sort_values(
        ["test_macro_f1", "test_mcc", "test_log_loss", "latent_dim", "model_or_variant"],
        ascending=[False, False, True, True, True],
    ).iloc[0]
    recon_best = selection_df.sort_values("test_recon_normalized_mse", ascending=True).iloc[0]

    aliases = [
        ("reconstruction_best_latent_head", recon_best, "reconstruction_best_test_normalized_mse", False, True, True),
        ("validation_selected_latent_head", validation, "validation_macro_f1_tie_mcc_log_loss_latent_dim_lexical", True, False, False),
        ("test_posthoc_best_latent_head", test_best, "posthoc_best_test_macro_f1", False, True, True),
    ]
    for alias, src, basis, uses_val, uses_test, oracle in aliases:
        metrics = {col.replace("test_", ""): src[col] for col in ["test_accuracy", "test_balanced_accuracy", "test_macro_f1", "test_mcc", "test_log_loss", "test_opposite_direction_rate"]}
        rows.append(
            base_perf_row(
                run,
                "frozen_latent_head",
                alias,
                "frozen_reconstruction_latent",
                int(src["latent_dim"]),
                int(src["latent_dim"]),
                float(src["compression_ratio"]),
                f"{basis}:{src['model_or_variant']}",
                uses_val,
                uses_test,
                oracle,
                metrics,
                len(y_train),
                len(y_val),
                len(y_test),
            )
        )

    performance = pd.DataFrame(rows)
    raw_tuned_metrics = performance[performance["model_or_variant"] == "raw_window_logistic_tuned"].iloc[0]
    majority_row = performance[performance["model_or_variant"] == "majority"].iloc[0]
    performance["delta_macro_f1_vs_raw_tuned_within_run"] = performance["test_macro_f1"] - float(raw_tuned_metrics["test_macro_f1"])
    performance["delta_mcc_vs_raw_tuned_within_run"] = performance["test_mcc"] - float(raw_tuned_metrics["test_mcc"])
    performance["delta_macro_f1_vs_majority_within_run"] = performance["test_macro_f1"] - float(majority_row["test_macro_f1"])

    selection_row = selection_alignment_row(run, selection_df, validation, test_best, recon_best)
    return performance, pd.DataFrame([selection_row])


def evaluate_representation(
    run: SplitRun,
    model_family: str,
    variant: str,
    latent_dim: int,
    compression_ratio: float,
    latents: Dict[str, np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    scaled_test: np.ndarray,
    recon_test: np.ndarray,
) -> Dict[str, object]:
    grid = fit_logistic_grid(latents["train"], y_train, latents["val"], y_val, latents["test"], y_test, run.seed)
    test_metrics = grid["selected"]["test_metrics"]
    val_metrics = grid["selected"]["val_metrics"]
    mse = float(np.mean((scaled_test - recon_test) ** 2))
    last_mse = float(np.mean((scaled_test.reshape(len(scaled_test), 100, 40)[:, -1, :] - recon_test.reshape(len(recon_test), 100, 40)[:, -1, :]) ** 2))
    return {
        "run_id": run.run_id,
        "split_protocol": run.split_protocol,
        "seed": run.seed,
        "model_family": model_family,
        "model_or_variant": variant,
        "latent_dim": latent_dim,
        "compression_ratio": compression_ratio,
        "selected_C": grid["selected"]["C"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_mcc": val_metrics["mcc"],
        "val_log_loss": val_metrics["log_loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_mcc": test_metrics["mcc"],
        "test_log_loss": test_metrics["log_loss"],
        "test_opposite_direction_rate": test_metrics["opposite_direction_rate"],
        "test_recon_normalized_mse": mse,
        "test_recon_last_step_mse": last_mse,
    }


def base_perf_row(
    run: SplitRun,
    family: str,
    variant: str,
    feature_source: str,
    input_dim: int,
    latent_dim: float,
    compression_ratio: float,
    selection_basis: str,
    uses_validation: bool,
    uses_test: bool,
    oracle: bool,
    metrics: Dict[str, float],
    n_train: int,
    n_val: int,
    n_test: int,
) -> Dict[str, object]:
    return {
        "run_id": run.run_id,
        "split_protocol": run.split_protocol,
        "seed": run.seed,
        "model_family": family,
        "model_or_variant": variant,
        "feature_source": feature_source,
        "input_dim": input_dim,
        "latent_dim": latent_dim,
        "compression_ratio": compression_ratio,
        "selection_basis": selection_basis,
        "uses_validation_for_selection": uses_validation,
        "uses_test_for_selection": uses_test,
        "is_oracle_reference": oracle,
        "test_accuracy": metrics["accuracy"],
        "test_balanced_accuracy": metrics["balanced_accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_mcc": metrics["mcc"],
        "test_log_loss": metrics["log_loss"],
        "test_opposite_direction_rate": metrics["opposite_direction_rate"],
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }


def spearman_safe(x: Iterable[float], y: Iterable[float]) -> float:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if len(x_arr) < 3 or np.nanstd(x_arr) == 0 or np.nanstd(y_arr) == 0:
        return float("nan")
    return float(spearmanr(x_arr, y_arr, nan_policy="omit").correlation)


def selection_alignment_row(run: SplitRun, selection_df: pd.DataFrame, validation: pd.Series, test_best: pd.Series, recon_best: pd.Series) -> Dict[str, object]:
    same_val_test = str(validation["model_or_variant"]) == str(test_best["model_or_variant"])
    same_recon_val = str(recon_best["model_or_variant"]) == str(validation["model_or_variant"])
    same_recon_test = str(recon_best["model_or_variant"]) == str(test_best["model_or_variant"])
    if same_recon_val and same_recon_test:
        status = "aligned"
        interp = "reconstruction-best and prediction-selected variants match within this run"
    elif same_val_test:
        status = "prediction_selection_stable_reconstruction_mismatch"
        interp = "validation and test select the same prediction variant, but reconstruction-best differs"
    else:
        status = "prediction_selection_unstable"
        interp = "validation-selected and test-posthoc variants differ"
    return {
        "run_id": run.run_id,
        "split_protocol": run.split_protocol,
        "seed": run.seed,
        "best_reconstruction_variant": recon_best["model_or_variant"],
        "best_reconstruction_metric": "test_recon_normalized_mse",
        "best_reconstruction_score": recon_best["test_recon_normalized_mse"],
        "validation_selected_latent_variant": validation["model_or_variant"],
        "validation_selected_val_macro_f1": validation["val_macro_f1"],
        "validation_selected_test_macro_f1": validation["test_macro_f1"],
        "test_posthoc_best_latent_variant": test_best["model_or_variant"],
        "test_posthoc_best_test_macro_f1": test_best["test_macro_f1"],
        "validation_selected_equals_test_posthoc_best": bool(same_val_test),
        "reconstruction_best_equals_validation_selected": bool(same_recon_val),
        "reconstruction_best_equals_test_posthoc_best": bool(same_recon_test),
        "spearman_recon_mse_vs_test_macro_f1": spearman_safe(selection_df["test_recon_normalized_mse"], selection_df["test_macro_f1"]),
        "spearman_recon_last_step_mse_vs_test_macro_f1": spearman_safe(selection_df["test_recon_last_step_mse"], selection_df["test_macro_f1"]),
        "rank_mismatch_status": status,
        "rank_mismatch_interpretation": interp,
    }


def protocol_runs_rows(runs: Sequence[SplitRun], samples: pd.DataFrame, sample_stride: int, window_len: int, label: str) -> pd.DataFrame:
    rows = []
    for run in runs:
        counts = run.assignments["split"].value_counts().to_dict()
        rows.append(
            {
                "run_id": run.run_id,
                "split_protocol": run.split_protocol,
                "seed": run.seed,
                "is_deterministic": run.is_deterministic,
                "split_ratio": "70/15/15",
                "sample_stride": sample_stride,
                "window_len": window_len,
                "label": label,
                "n_total_samples": len(run.assignments),
                "n_train": int(counts.get("train", 0)),
                "n_val": int(counts.get("val", 0)),
                "n_test": int(counts.get("test", 0)),
                "purge_policy": run.purge_policy,
                "randomization_policy": run.randomization_policy,
                "block_policy": run.block_policy,
                "block_size": run.block_size,
                "embargo_size": run.embargo_size,
                "notes": run.notes,
            }
        )
    return pd.DataFrame(rows)


def summarize_protocols(perf: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    merged = perf.merge(
        audit[["run_id", "fraction_test_with_overlapping_train_window", "fraction_test_with_near_neighbor_train_k5"]],
        on="run_id",
        how="left",
    )
    rows = []
    for (protocol, variant), part in merged.groupby(["split_protocol", "model_or_variant"], sort=False):
        rows.append(
            {
                "split_protocol": protocol,
                "model_or_variant": variant,
                "n_runs": len(part),
                "mean_test_macro_f1": part["test_macro_f1"].mean(),
                "std_test_macro_f1": part["test_macro_f1"].std(ddof=0),
                "min_test_macro_f1": part["test_macro_f1"].min(),
                "max_test_macro_f1": part["test_macro_f1"].max(),
                "mean_test_mcc": part["test_mcc"].mean(),
                "std_test_mcc": part["test_mcc"].std(ddof=0),
                "mean_test_balanced_accuracy": part["test_balanced_accuracy"].mean(),
                "std_test_balanced_accuracy": part["test_balanced_accuracy"].std(ddof=0),
                "mean_test_log_loss": part["test_log_loss"].mean(),
                "std_test_log_loss": part["test_log_loss"].std(ddof=0),
                "mean_fraction_test_with_overlapping_train_window": part["fraction_test_with_overlapping_train_window"].mean(),
                "mean_fraction_test_with_near_neighbor_train_k5": part["fraction_test_with_near_neighbor_train_k5"].mean(),
                "mean_delta_macro_f1_vs_raw_tuned_within_run": part["delta_macro_f1_vs_raw_tuned_within_run"].mean(),
                "mean_delta_mcc_vs_raw_tuned_within_run": part["delta_mcc_vs_raw_tuned_within_run"].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    contrast_specs = [
        ("random_window_naive_vs_chronological_purged", "chronological_purged", "random_window_naive"),
        ("random_block_purged_vs_chronological_purged", "chronological_purged", "random_block_purged"),
        ("random_window_naive_vs_random_block_purged", "random_block_purged", "random_window_naive"),
        ("chronological_no_purge_vs_chronological_purged", "chronological_purged", "chronological_no_purge"),
    ]
    rows = []
    for cid, baseline, comparison in contrast_specs:
        for variant in MODEL_ORDER:
            base = summary[(summary["split_protocol"] == baseline) & (summary["model_or_variant"] == variant)]
            comp = summary[(summary["split_protocol"] == comparison) & (summary["model_or_variant"] == variant)]
            if base.empty or comp.empty:
                continue
            b = base.iloc[0]
            c = comp.iloc[0]
            delta_macro = c["mean_test_macro_f1"] - b["mean_test_macro_f1"]
            delta_risk = c["mean_fraction_test_with_overlapping_train_window"] - b["mean_fraction_test_with_overlapping_train_window"]
            if cid == "random_window_naive_vs_random_block_purged":
                interpretation = "near-neighbor exposure / overlapping-window effect proxy"
            elif cid == "random_block_purged_vs_chronological_purged":
                interpretation = "temporal or regime mixing effect proxy with near-neighbor risk controlled"
            elif cid == "random_window_naive_vs_chronological_purged":
                interpretation = "mixed randomization effect including temporal mixing, near-neighbor exposure, and purge differences"
            else:
                interpretation = "boundary purge effect proxy on the existing kept sample universe"
            rows.append(
                {
                    "contrast_id": cid,
                    "baseline_protocol": baseline,
                    "comparison_protocol": comparison,
                    "model_or_variant": variant,
                    "baseline_mean_test_macro_f1": b["mean_test_macro_f1"],
                    "comparison_mean_test_macro_f1": c["mean_test_macro_f1"],
                    "delta_test_macro_f1": delta_macro,
                    "baseline_mean_test_mcc": b["mean_test_mcc"],
                    "comparison_mean_test_mcc": c["mean_test_mcc"],
                    "delta_test_mcc": c["mean_test_mcc"] - b["mean_test_mcc"],
                    "baseline_mean_test_balanced_accuracy": b["mean_test_balanced_accuracy"],
                    "comparison_mean_test_balanced_accuracy": c["mean_test_balanced_accuracy"],
                    "delta_test_balanced_accuracy": c["mean_test_balanced_accuracy"] - b["mean_test_balanced_accuracy"],
                    "baseline_mean_overlap_risk": b["mean_fraction_test_with_overlapping_train_window"],
                    "comparison_mean_overlap_risk": c["mean_fraction_test_with_overlapping_train_window"],
                    "delta_overlap_risk": delta_risk,
                    "n_baseline_runs": b["n_runs"],
                    "n_comparison_runs": c["n_runs"],
                    "comparison_seed_std_macro_f1": c["std_test_macro_f1"],
                    "interpretation": interpretation,
                }
            )
    return pd.DataFrame(rows)


def build_manifest(args: argparse.Namespace, runs: Sequence[SplitRun], summary: pd.DataFrame, contrasts: pd.DataFrame) -> Dict[str, object]:
    return {
        "step": "step10_split_protocol_decomposition",
        "purpose": "Treat split protocol as an experimental variable and decompose random split effects into temporal mixing, near-neighbor exposure, and boundary-purge components.",
        "protocols": sorted({run.split_protocol for run in runs}),
        "random_seeds": parse_seeds(args.random_seeds),
        "primary_metric": PRIMARY_METRIC,
        "model_panel": MODEL_ORDER,
        "head_training_policy": {
            "classifier": "LogisticRegression",
            "solver": "saga",
            "max_iter": 300,
            "tol": 0.01,
            "class_weight": "balanced",
            "c_grid": C_GRID,
            "scaler": "train-only StandardScaler inside each protocol run",
            "selection": "validation macro-F1, tie MCC, tie log_loss",
        },
        "protocol_interpretation_rules": {
            "random_window_naive_minus_chronological_purged": "temporal mixing + near-neighbor exposure + purge differences",
            "random_block_purged_minus_chronological_purged": "temporal or regime mixing with near-neighbor risk reduced",
            "random_window_naive_minus_random_block_purged": "near-neighbor exposure / overlapping-window effect proxy",
            "chronological_no_purge_minus_chronological_purged": "boundary purge effect proxy on existing kept sample universe",
        },
        "full_representation_retraining_performed": False,
        "lightweight_panel_used": True,
        "lightweight_panel_note": "Step 10 fits train-only PCA@32/PCA@128 and last_snapshot_repeat heads per split run. It does not repeat the full Step 6 MLP-AE representation panel.",
        "split_universe_note": "All protocols operate on the existing Step 3 kept 7952-sample subset; Step 3 dropped boundary samples are not restored for chronological_no_purge.",
        "main_outputs": {
            "protocol_summary": "results/step10_split_protocol_decomposition/protocol_summary.csv",
            "protocol_contrasts": "results/step10_split_protocol_decomposition/protocol_contrasts.csv",
            "split_integrity_audit": "results/step10_split_protocol_decomposition/split_integrity_audit.csv",
            "model_performance_by_run": "results/step10_split_protocol_decomposition/model_performance_by_run.csv",
            "selection_alignment_by_run": "results/step10_split_protocol_decomposition/selection_alignment_by_run.csv",
        },
        "remaining_limitations": [
            "single symbol",
            "single horizon trend5",
            "single stride-4 sample universe",
            "lightweight representation panel only",
            "no MLP-AE retraining",
            "no trading PnL",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-dir", default="data/processed/minimal_subset")
    parser.add_argument("--output-dir", default="results/step10_split_protocol_decomposition")
    parser.add_argument("--random-seeds", default="42,43,44,45,46")
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--embargo-size", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset_dir = Path(args.subset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.load(subset_dir / "X.npy")
    y = np.load(subset_dir / "y.npy").astype(int)
    samples = pd.read_csv(subset_dir / "samples.csv")
    seeds = parse_seeds(args.random_seeds)
    sample_stride = 4
    window_len = int(x.shape[1])
    runs = build_split_runs(samples, seeds, args.block_size, args.embargo_size)

    run_registry = protocol_runs_rows(runs, samples, sample_stride, window_len, "trend5")
    audits = []
    performance_parts = []
    selection_parts = []
    for run in runs:
        print(f"Running {run.run_id}...", flush=True)
        audits.append(split_integrity(run, samples, y, window_len, sample_stride))
        perf, selection = train_and_evaluate_run(run, samples, x, y)
        performance_parts.append(perf)
        selection_parts.append(selection)

    audit_df = pd.DataFrame(audits)
    perf_df = pd.concat(performance_parts, ignore_index=True)
    selection_df = pd.concat(selection_parts, ignore_index=True)
    summary_df = summarize_protocols(perf_df, audit_df)
    contrasts_df = build_contrasts(summary_df)
    manifest = build_manifest(args, runs, summary_df, contrasts_df)

    run_registry.to_csv(output_dir / "protocol_runs.csv", index=False)
    audit_df.to_csv(output_dir / "split_integrity_audit.csv", index=False)
    perf_df.to_csv(output_dir / "model_performance_by_run.csv", index=False)
    selection_df.to_csv(output_dir / "selection_alignment_by_run.csv", index=False)
    contrasts_df.to_csv(output_dir / "protocol_contrasts.csv", index=False)
    summary_df.to_csv(output_dir / "protocol_summary.csv", index=False)
    with (output_dir / "protocol_manifest.json").open("w") as f:
        json.dump(to_jsonable(manifest), f, indent=2)

    key = contrasts_df[
        (contrasts_df["model_or_variant"] == "raw_window_logistic_tuned")
        & (contrasts_df["contrast_id"].isin([
            "random_window_naive_vs_chronological_purged",
            "random_block_purged_vs_chronological_purged",
            "random_window_naive_vs_random_block_purged",
        ]))
    ][["contrast_id", "delta_test_macro_f1", "delta_overlap_risk"]]
    print(key.to_string(index=False))


if __name__ == "__main__":
    main()
