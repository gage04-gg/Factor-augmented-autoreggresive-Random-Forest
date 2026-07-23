"""
metrics.py — Evaluation metrics for MRF replication.

All functions are NaN-safe (pairs where either value is NaN are excluded).
"""

import numpy as np


def compute_rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error (NaN-pair safe)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def compute_mae(y_true, y_pred) -> float:
    """Mean Absolute Error (NaN-pair safe)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def compute_rmse_ratio(y_true, preds_model, preds_bench) -> float:
    """
    RMSE ratio: model RMSE / benchmark RMSE.
    Values < 1 mean the model beats the benchmark.
    Paper Table 4 reports this ratio vs AR(4).
    """
    return compute_rmse(y_true, preds_model) / compute_rmse(y_true, preds_bench)
