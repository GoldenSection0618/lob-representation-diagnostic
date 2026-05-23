"""Metric helpers for Step 6 reconstruction-only baselines."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

FEATURE_DIM = 40
WINDOW_LEN = 100
EPS = 1e-12
IMBALANCE_EPS_THRESHOLD = 1e-6
IMBALANCE_VALID_RATIO_THRESHOLD = 0.95

BEST_BID_PRICE1_IDX = 9
BEST_ASK_PRICE1_IDX = 10
BEST_BID_VOLUME1_IDX = 29
BEST_ASK_VOLUME1_IDX = 30


def _to_float(value: float) -> float:
    return float(np.asarray(value, dtype=np.float64))


def _safe_div(a: float, b: float) -> float:
    if abs(b) < EPS:
        return float("nan")
    return float(a / b)


def _assert_window_shapes(X_true: np.ndarray, X_hat: np.ndarray) -> None:
    if X_true.shape != X_hat.shape:
        raise ValueError(f"Shape mismatch: X_true={X_true.shape}, X_hat={X_hat.shape}")
    if X_true.ndim != 3 or X_true.shape[2] != FEATURE_DIM:
        raise ValueError(f"Expected shape (N, T, {FEATURE_DIM}), got {X_true.shape}")


def _assert_flat_shapes(X_true_flat: np.ndarray, X_hat_flat: np.ndarray) -> None:
    if X_true_flat.shape != X_hat_flat.shape:
        raise ValueError(f"Flat shape mismatch: {X_true_flat.shape} vs {X_hat_flat.shape}")
    if X_true_flat.ndim != 2:
        raise ValueError(f"Expected 2D flattened arrays, got {X_true_flat.ndim}D")


def _level_indices(level: int) -> Dict[str, int]:
    if level < 1 or level > 10:
        raise ValueError(f"level must be in [1,10], got {level}")
    return {
        "bid_price": 10 - level,
        "ask_price": 9 + level,
        "bid_volume": 30 - level,
        "ask_volume": 29 + level,
    }


def _imbalance(bid: np.ndarray, ask: np.ndarray) -> np.ndarray:
    return bid / (bid + ask + EPS)


def _imbalance_validity_masks(
    true_bid: np.ndarray,
    true_ask: np.ndarray,
    pred_bid: np.ndarray,
    pred_ask: np.ndarray,
    eps_threshold: float = IMBALANCE_EPS_THRESHOLD,
) -> Dict[str, np.ndarray]:
    """Build validity masks for imbalance diagnostics."""
    nonnegative = (true_bid >= 0.0) & (true_ask >= 0.0) & (pred_bid >= 0.0) & (pred_ask >= 0.0)
    denom_true = true_bid + true_ask
    denom_pred = pred_bid + pred_ask
    denom_small = (denom_true <= eps_threshold) | (denom_pred <= eps_threshold)
    valid = nonnegative & (~denom_small)
    return {
        "nonnegative": nonnegative,
        "denom_small": denom_small,
        "valid": valid,
    }


def _masked_mae(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return _to_float(np.mean(np.abs(values[mask])))


def _top5_volume_indices() -> Dict[str, List[int]]:
    bid = [30 - k for k in range(1, 6)]
    ask = [29 + k for k in range(1, 6)]
    return {"bid": bid, "ask": ask}


def compute_primary_metrics(
    X_true_scaled_flat: np.ndarray,
    X_hat_scaled_flat: np.ndarray,
    X_true_original: np.ndarray,
    X_hat_original: np.ndarray,
    baseline_train_mean_mse: float,
    baseline_last_snapshot_mse: float,
    compression_ratio: float | None,
    num_parameters: int,
    train_seconds: float,
    inference_ms_per_1000_samples: float,
) -> Dict[str, float | int | None]:
    """Compute summary reconstruction metrics for one model/split."""
    _assert_flat_shapes(X_true_scaled_flat, X_hat_scaled_flat)
    _assert_window_shapes(X_true_original, X_hat_original)

    diff_scaled = X_hat_scaled_flat - X_true_scaled_flat
    diff_original = X_hat_original - X_true_original

    normalized_mse = _to_float(np.mean(np.square(diff_scaled)))
    normalized_mae = _to_float(np.mean(np.abs(diff_scaled)))
    original_rmse = _to_float(np.sqrt(np.mean(np.square(diff_original))))
    original_mae = _to_float(np.mean(np.abs(diff_original)))

    return {
        "normalized_mse": normalized_mse,
        "normalized_mae": normalized_mae,
        "original_rmse": original_rmse,
        "original_mae": original_mae,
        "relative_mse_vs_train_mean": _safe_div(normalized_mse, baseline_train_mean_mse),
        "relative_mse_vs_last_snapshot": _safe_div(normalized_mse, baseline_last_snapshot_mse),
        "compression_ratio": None if compression_ratio is None else float(compression_ratio),
        "num_parameters": int(num_parameters),
        "train_seconds": float(train_seconds),
        "inference_ms_per_1000_samples": float(inference_ms_per_1000_samples),
    }


def compute_feature_group_errors(
    X_true_scaled: np.ndarray,
    X_hat_scaled: np.ndarray,
    X_true_original: np.ndarray,
    X_hat_original: np.ndarray,
) -> List[Dict[str, float | str]]:
    """Compute feature-group reconstruction errors."""
    _assert_window_shapes(X_true_scaled, X_hat_scaled)
    _assert_window_shapes(X_true_original, X_hat_original)

    groups = {
        "all": list(range(FEATURE_DIM)),
        "price": list(range(0, 20)),
        "volume": list(range(20, 40)),
        "bid_price": list(range(0, 10)),
        "ask_price": list(range(10, 20)),
        "bid_volume": list(range(20, 30)),
        "ask_volume": list(range(30, 40)),
        "top_of_book": [BEST_BID_PRICE1_IDX, BEST_ASK_PRICE1_IDX, BEST_BID_VOLUME1_IDX, BEST_ASK_VOLUME1_IDX],
        "last_step": list(range(0, 40)),
    }

    out: List[Dict[str, float | str]] = []
    for name, idx in groups.items():
        if name == "last_step":
            t_scaled = X_true_scaled[:, -1, :]
            h_scaled = X_hat_scaled[:, -1, :]
            t_original = X_true_original[:, -1, :]
            h_original = X_hat_original[:, -1, :]
        else:
            t_scaled = X_true_scaled[:, :, idx]
            h_scaled = X_hat_scaled[:, :, idx]
            t_original = X_true_original[:, :, idx]
            h_original = X_hat_original[:, :, idx]

        diff_scaled = h_scaled - t_scaled
        diff_original = h_original - t_original
        out.append(
            {
                "group": name,
                "normalized_mse": _to_float(np.mean(np.square(diff_scaled))),
                "normalized_mae": _to_float(np.mean(np.abs(diff_scaled))),
                "original_mae": _to_float(np.mean(np.abs(diff_original))),
            }
        )
    return out


def compute_level_wise_errors(
    X_true_scaled: np.ndarray,
    X_hat_scaled: np.ndarray,
    X_true_original: np.ndarray,
    X_hat_original: np.ndarray,
) -> List[Dict[str, float | int | str]]:
    """Compute level-wise reconstruction errors for bid/ask and price/volume."""
    _assert_window_shapes(X_true_scaled, X_hat_scaled)
    _assert_window_shapes(X_true_original, X_hat_original)

    out: List[Dict[str, float | int | str]] = []
    for level in range(1, 11):
        idx = _level_indices(level)
        specs = [
            ("bid", "price", idx["bid_price"]),
            ("ask", "price", idx["ask_price"]),
            ("bid", "volume", idx["bid_volume"]),
            ("ask", "volume", idx["ask_volume"]),
        ]
        for side, field_type, f_idx in specs:
            t_scaled = X_true_scaled[:, :, f_idx]
            h_scaled = X_hat_scaled[:, :, f_idx]
            t_original = X_true_original[:, :, f_idx]
            h_original = X_hat_original[:, :, f_idx]
            out.append(
                {
                    "level": int(level),
                    "side": side,
                    "field_type": field_type,
                    "normalized_mse": _to_float(np.mean(np.square(h_scaled - t_scaled))),
                    "normalized_mae": _to_float(np.mean(np.abs(h_scaled - t_scaled))),
                    "original_mae": _to_float(np.mean(np.abs(h_original - t_original))),
                }
            )
    return out


def compute_temporal_errors(
    X_true_scaled: np.ndarray,
    X_hat_scaled: np.ndarray,
) -> List[Dict[str, float | int]]:
    """Compute time-step reconstruction errors."""
    _assert_window_shapes(X_true_scaled, X_hat_scaled)

    out: List[Dict[str, float | int]] = []
    for t in range(X_true_scaled.shape[1]):
        diff = X_hat_scaled[:, t, :] - X_true_scaled[:, t, :]
        out.append(
            {
                "timestep": int(t),
                "normalized_mse": _to_float(np.mean(np.square(diff))),
                "normalized_mae": _to_float(np.mean(np.abs(diff))),
            }
        )
    return out


def compute_derived_lob_errors(
    X_true_original: np.ndarray,
    X_hat_original: np.ndarray,
) -> Dict[str, float]:
    """Compute derived midprice/spread/imbalance errors in original feature space."""
    _assert_window_shapes(X_true_original, X_hat_original)

    true_bid1 = X_true_original[:, :, BEST_BID_PRICE1_IDX]
    true_ask1 = X_true_original[:, :, BEST_ASK_PRICE1_IDX]
    pred_bid1 = X_hat_original[:, :, BEST_BID_PRICE1_IDX]
    pred_ask1 = X_hat_original[:, :, BEST_ASK_PRICE1_IDX]

    true_mid = (true_bid1 + true_ask1) / 2.0
    pred_mid = (pred_bid1 + pred_ask1) / 2.0

    true_spread = true_ask1 - true_bid1
    pred_spread = pred_ask1 - pred_bid1

    true_bid_vol1 = X_true_original[:, :, BEST_BID_VOLUME1_IDX]
    true_ask_vol1 = X_true_original[:, :, BEST_ASK_VOLUME1_IDX]
    pred_bid_vol1 = X_hat_original[:, :, BEST_BID_VOLUME1_IDX]
    pred_ask_vol1 = X_hat_original[:, :, BEST_ASK_VOLUME1_IDX]

    true_top1_imb = _imbalance(true_bid_vol1, true_ask_vol1)
    pred_top1_imb = _imbalance(pred_bid_vol1, pred_ask_vol1)

    top5_idx = _top5_volume_indices()
    true_top5_bid = np.sum(X_true_original[:, :, top5_idx["bid"]], axis=2)
    true_top5_ask = np.sum(X_true_original[:, :, top5_idx["ask"]], axis=2)
    pred_top5_bid = np.sum(X_hat_original[:, :, top5_idx["bid"]], axis=2)
    pred_top5_ask = np.sum(X_hat_original[:, :, top5_idx["ask"]], axis=2)

    true_top5_imb = _imbalance(true_top5_bid, true_top5_ask)
    pred_top5_imb = _imbalance(pred_top5_bid, pred_top5_ask)

    top1_masks = _imbalance_validity_masks(
        true_bid=true_bid_vol1,
        true_ask=true_ask_vol1,
        pred_bid=pred_bid_vol1,
        pred_ask=pred_ask_vol1,
    )
    top5_masks = _imbalance_validity_masks(
        true_bid=true_top5_bid,
        true_ask=true_top5_ask,
        pred_bid=pred_top5_bid,
        pred_ask=pred_top5_ask,
    )

    top1_nonnegative_ratio = _to_float(np.mean(top1_masks["nonnegative"]))
    top1_denom_small_ratio = _to_float(np.mean(top1_masks["denom_small"]))
    top1_valid_ratio = _to_float(np.mean(top1_masks["valid"]))
    top1_imbalance_valid = bool(top1_valid_ratio >= IMBALANCE_VALID_RATIO_THRESHOLD)
    top1_imbalance_mae = (
        _masked_mae(pred_top1_imb - true_top1_imb, top1_masks["valid"]) if top1_imbalance_valid else float("nan")
    )

    top5_nonnegative_ratio = _to_float(np.mean(top5_masks["nonnegative"]))
    top5_denom_small_ratio = _to_float(np.mean(top5_masks["denom_small"]))
    top5_valid_ratio = _to_float(np.mean(top5_masks["valid"]))
    top5_imbalance_valid = bool(top5_valid_ratio >= IMBALANCE_VALID_RATIO_THRESHOLD)
    top5_imbalance_mae = (
        _masked_mae(pred_top5_imb - true_top5_imb, top5_masks["valid"]) if top5_imbalance_valid else float("nan")
    )

    top1_true_sum = true_bid_vol1 + true_ask_vol1
    top1_pred_sum = pred_bid_vol1 + pred_ask_vol1
    top1_true_diff = true_bid_vol1 - true_ask_vol1
    top1_pred_diff = pred_bid_vol1 - pred_ask_vol1

    top5_true_sum = true_top5_bid + true_top5_ask
    top5_pred_sum = pred_top5_bid + pred_top5_ask
    top5_true_diff = true_top5_bid - true_top5_ask
    top5_pred_diff = pred_top5_bid - pred_top5_ask

    return {
        "midprice_mae": _to_float(np.mean(np.abs(pred_mid - true_mid))),
        "spread_mae": _to_float(np.mean(np.abs(pred_spread - true_spread))),
        "best_bid_mae": _to_float(np.mean(np.abs(pred_bid1 - true_bid1))),
        "best_ask_mae": _to_float(np.mean(np.abs(pred_ask1 - true_ask1))),
        "volume_nonnegative_ratio": _to_float((top1_nonnegative_ratio + top5_nonnegative_ratio) / 2.0),
        "imbalance_denominator_small_ratio": _to_float((top1_denom_small_ratio + top5_denom_small_ratio) / 2.0),
        "imbalance_valid_ratio": _to_float((top1_valid_ratio + top5_valid_ratio) / 2.0),
        "imbalance_valid": bool(top1_imbalance_valid and top5_imbalance_valid),
        "top1_volume_nonnegative_ratio": top1_nonnegative_ratio,
        "top1_imbalance_denominator_small_ratio": top1_denom_small_ratio,
        "top1_imbalance_valid_ratio": top1_valid_ratio,
        "top1_imbalance_valid": top1_imbalance_valid,
        "top1_imbalance_mae": top1_imbalance_mae,
        "top5_volume_nonnegative_ratio": top5_nonnegative_ratio,
        "top5_imbalance_denominator_small_ratio": top5_denom_small_ratio,
        "top5_imbalance_valid_ratio": top5_valid_ratio,
        "top5_imbalance_valid": top5_imbalance_valid,
        "top5_imbalance_mae": top5_imbalance_mae,
        "top1_volume_sum_mae": _to_float(np.mean(np.abs(top1_pred_sum - top1_true_sum))),
        "top5_volume_sum_mae": _to_float(np.mean(np.abs(top5_pred_sum - top5_true_sum))),
        "top1_volume_diff_mae": _to_float(np.mean(np.abs(top1_pred_diff - top1_true_diff))),
        "top5_volume_diff_mae": _to_float(np.mean(np.abs(top5_pred_diff - top5_true_diff))),
    }


def compute_per_sample_errors(
    X_true_scaled_flat: np.ndarray,
    X_hat_scaled_flat: np.ndarray,
    X_true_scaled: np.ndarray,
    X_hat_scaled: np.ndarray,
    X_true_original: np.ndarray,
    X_hat_original: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute per-sample reconstruction diagnostics used by Step 7."""
    _assert_flat_shapes(X_true_scaled_flat, X_hat_scaled_flat)
    _assert_window_shapes(X_true_scaled, X_hat_scaled)
    _assert_window_shapes(X_true_original, X_hat_original)

    diff_scaled_flat = X_hat_scaled_flat - X_true_scaled_flat
    normalized_mse = np.mean(np.square(diff_scaled_flat), axis=1)
    normalized_mae = np.mean(np.abs(diff_scaled_flat), axis=1)

    top_idx = [BEST_BID_PRICE1_IDX, BEST_ASK_PRICE1_IDX, BEST_BID_VOLUME1_IDX, BEST_ASK_VOLUME1_IDX]
    top_diff = X_hat_scaled[:, :, top_idx] - X_true_scaled[:, :, top_idx]
    top_of_book_mse = np.mean(np.square(top_diff), axis=(1, 2))

    last_diff = X_hat_scaled[:, -1, :] - X_true_scaled[:, -1, :]
    last_step_mse = np.mean(np.square(last_diff), axis=1)

    true_bid1 = X_true_original[:, :, BEST_BID_PRICE1_IDX]
    true_ask1 = X_true_original[:, :, BEST_ASK_PRICE1_IDX]
    pred_bid1 = X_hat_original[:, :, BEST_BID_PRICE1_IDX]
    pred_ask1 = X_hat_original[:, :, BEST_ASK_PRICE1_IDX]

    true_mid = (true_bid1 + true_ask1) / 2.0
    pred_mid = (pred_bid1 + pred_ask1) / 2.0

    true_spread = true_ask1 - true_bid1
    pred_spread = pred_ask1 - pred_bid1

    true_bid_vol1 = X_true_original[:, :, BEST_BID_VOLUME1_IDX]
    true_ask_vol1 = X_true_original[:, :, BEST_ASK_VOLUME1_IDX]
    pred_bid_vol1 = X_hat_original[:, :, BEST_BID_VOLUME1_IDX]
    pred_ask_vol1 = X_hat_original[:, :, BEST_ASK_VOLUME1_IDX]

    true_top1_imb = _imbalance(true_bid_vol1, true_ask_vol1)
    pred_top1_imb = _imbalance(pred_bid_vol1, pred_ask_vol1)

    top5_idx = _top5_volume_indices()
    true_top5_bid = np.sum(X_true_original[:, :, top5_idx["bid"]], axis=2)
    true_top5_ask = np.sum(X_true_original[:, :, top5_idx["ask"]], axis=2)
    pred_top5_bid = np.sum(X_hat_original[:, :, top5_idx["bid"]], axis=2)
    pred_top5_ask = np.sum(X_hat_original[:, :, top5_idx["ask"]], axis=2)

    true_top5_imb = _imbalance(true_top5_bid, true_top5_ask)
    pred_top5_imb = _imbalance(pred_top5_bid, pred_top5_ask)

    top1_masks = _imbalance_validity_masks(
        true_bid=true_bid_vol1,
        true_ask=true_ask_vol1,
        pred_bid=pred_bid_vol1,
        pred_ask=pred_ask_vol1,
    )
    top5_masks = _imbalance_validity_masks(
        true_bid=true_top5_bid,
        true_ask=true_top5_ask,
        pred_bid=pred_top5_bid,
        pred_ask=pred_top5_ask,
    )

    top1_valid_ratio_per_sample = np.mean(top1_masks["valid"], axis=1)
    top5_valid_ratio_per_sample = np.mean(top5_masks["valid"], axis=1)

    top1_diff = np.abs(pred_top1_imb - true_top1_imb)
    top5_diff = np.abs(pred_top5_imb - true_top5_imb)
    top1_masked_sum = np.sum(np.where(top1_masks["valid"], top1_diff, 0.0), axis=1)
    top5_masked_sum = np.sum(np.where(top5_masks["valid"], top5_diff, 0.0), axis=1)
    top1_mask_count = np.sum(top1_masks["valid"], axis=1)
    top5_mask_count = np.sum(top5_masks["valid"], axis=1)
    top1_imbalance_mae = np.divide(
        top1_masked_sum,
        top1_mask_count,
        out=np.full_like(top1_masked_sum, np.nan, dtype=np.float64),
        where=top1_mask_count > 0,
    )
    top5_imbalance_mae = np.divide(
        top5_masked_sum,
        top5_mask_count,
        out=np.full_like(top5_masked_sum, np.nan, dtype=np.float64),
        where=top5_mask_count > 0,
    )

    top1_valid_gate = top1_valid_ratio_per_sample >= IMBALANCE_VALID_RATIO_THRESHOLD
    top5_valid_gate = top5_valid_ratio_per_sample >= IMBALANCE_VALID_RATIO_THRESHOLD
    top1_imbalance_mae = np.where(top1_valid_gate, top1_imbalance_mae, np.nan)
    top5_imbalance_mae = np.where(top5_valid_gate, top5_imbalance_mae, np.nan)

    top1_true_sum = true_bid_vol1 + true_ask_vol1
    top1_pred_sum = pred_bid_vol1 + pred_ask_vol1
    top1_true_diff = true_bid_vol1 - true_ask_vol1
    top1_pred_diff = pred_bid_vol1 - pred_ask_vol1

    top5_true_sum = true_top5_bid + true_top5_ask
    top5_pred_sum = pred_top5_bid + pred_top5_ask
    top5_true_diff = true_top5_bid - true_top5_ask
    top5_pred_diff = pred_top5_bid - pred_top5_ask

    return {
        "normalized_mse": normalized_mse.astype(np.float64),
        "normalized_mae": normalized_mae.astype(np.float64),
        "top_of_book_mse": top_of_book_mse.astype(np.float64),
        "last_step_mse": last_step_mse.astype(np.float64),
        "midprice_mae": np.mean(np.abs(pred_mid - true_mid), axis=1).astype(np.float64),
        "spread_mae": np.mean(np.abs(pred_spread - true_spread), axis=1).astype(np.float64),
        "top1_imbalance_mae": top1_imbalance_mae.astype(np.float64),
        "top5_imbalance_mae": top5_imbalance_mae.astype(np.float64),
        "top1_imbalance_valid": top1_valid_gate.astype(bool),
        "top5_imbalance_valid": top5_valid_gate.astype(bool),
        "top1_imbalance_valid_ratio": top1_valid_ratio_per_sample.astype(np.float64),
        "top5_imbalance_valid_ratio": top5_valid_ratio_per_sample.astype(np.float64),
        "top1_volume_sum_mae": np.mean(np.abs(top1_pred_sum - top1_true_sum), axis=1).astype(np.float64),
        "top5_volume_sum_mae": np.mean(np.abs(top5_pred_sum - top5_true_sum), axis=1).astype(np.float64),
        "top1_volume_diff_mae": np.mean(np.abs(top1_pred_diff - top1_true_diff), axis=1).astype(np.float64),
        "top5_volume_diff_mae": np.mean(np.abs(top5_pred_diff - top5_true_diff), axis=1).astype(np.float64),
    }


def class_distribution(y: np.ndarray) -> Dict[str, int]:
    """Return label counts in fixed class order {0,1,2}."""
    out = {"0": 0, "1": 0, "2": 0}
    vals, cnt = np.unique(y, return_counts=True)
    for v, c in zip(vals, cnt):
        out[str(int(v))] = int(c)
    return out


def validate_feature_contract(feature_dim: int, window_len: int, expected_window_len: int = WINDOW_LEN) -> None:
    """Validate Step 3 fixed input contract for Step 6 usage."""
    if feature_dim != FEATURE_DIM:
        raise ValueError(f"Expected feature_dim={FEATURE_DIM}, got {feature_dim}")
    if window_len != expected_window_len:
        raise ValueError(f"Expected window_len={expected_window_len}, got {window_len}")
