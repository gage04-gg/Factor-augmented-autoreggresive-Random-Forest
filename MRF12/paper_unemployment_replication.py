"""
Standalone unemployment-rate replication script for
"The Macroeconomy as a Random Forest" (Goulet Coulombe, 2024 / WP 2021).

Purpose
-------
This file is intentionally self-contained. It does not import the local
project's earlier forest implementation and instead rebuilds the paper-style
pipeline from scratch:

  1. Read a quarterly FRED-QD-derived workbook.
  2. Reconstruct a paper-style state matrix S_t.
  3. Build direct forecasts for the unemployment-rate change target.
  4. Estimate Macroeconomic Random Forest (MRF) / FA-ARRF from scratch.
  5. Compute RMSE ratios against AR(4) and plot paper-style figures.

Design choices intentionally mirror the paper as closely as possible:
  - Quarterly pseudo-OOS window: 2003Q1 to 2014Q4
  - Expanding window estimation
  - Re-estimation every 8 quarters (2 years)
  - FA-ARRF linear part: [1, y_t, y_{t-1}, F1_t, F2_t]
  - S_t: trend + 8 lags of target + 2 lags of FRED-QD variables
         + 8 lags of 5 factors + 2 MAFs per variable
  - Block subsampling and random-walk regularization in leaf regressions

Notes
-----
1. The paper's exact Table 4 unemployment RMSE ratios are not embedded here as
   hard-coded numeric truths unless you provide a verified transcription.
   The script therefore computes empirical RMSE ratios from the supplied data
   and can optionally display manually-entered paper targets for comparison.
2. The workbook used in this project is model-ready and not the raw St. Louis
   Fed FRED-QD file. Forecasts can only match the paper if the underlying
   vintage and transformations also match the paper's setup.

Usage
-----
python3 paper_unemployment_replication.py
python3 paper_unemployment_replication.py --data fredqd_final_model_ready.xlsx --n-trees 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib").resolve()))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA


META_COLS = {"observation_date", "quarter", "sample_flag", "trend_t"}
FACTOR_COLS = [f"F{i}" for i in range(1, 6)]
DEFAULT_HORIZONS = (1, 2, 4, 6, 8)
PAPER_OOS_START = pd.Timestamp("2003-01-01")
PAPER_OOS_END = pd.Timestamp("2014-10-01")

# Fill these with verified Table 4 numbers if you have a trusted transcription.
PAPER_UR_FA_ARRF_RMSE_TARGETS = {
    1: np.nan,
    2: np.nan,
    4: np.nan,
    6: np.nan,
    8: np.nan,
}


@dataclass
class ReplicationConfig:
    data_path: Path
    output_dir: Path
    target: str = "UNRATE"
    n_factors: int = 5
    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    update_freq: int = 8
    n_trees: int = 100
    random_state: int = 1234
    min_samples_leaf: int = 10
    mtry_frac: float = 1 / 3
    ridge_lambda: float = 0.10
    rw_regul: float = 0.75
    subsampling_rate: float = 0.75
    block_size: int = 8
    trend_push: float = 1.0
    max_depth: int = 100
    min_leaf_frac_of_x: float = 1.0
    fast_rw: bool = True


def parse_args() -> ReplicationConfig:
    parser = argparse.ArgumentParser(description="Standalone FA-ARRF unemployment replication.")
    parser.add_argument("--data", type=Path, default=Path("fredqd_final_model_ready.xlsx"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_unemployment_replication"))
    parser.add_argument("--target", type=str, default="UNRATE")
    parser.add_argument("--n-factors", type=int, default=5)
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--update-freq", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--mtry-frac", type=float, default=1 / 3)
    parser.add_argument("--ridge-lambda", type=float, default=0.10)
    parser.add_argument("--rw-regul", type=float, default=0.75)
    parser.add_argument("--subsampling-rate", type=float, default=0.75)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--trend-push", type=float, default=1.0)
    parser.add_argument("--max-depth", type=int, default=100)
    parser.add_argument("--min-leaf-frac-of-x", type=float, default=1.0)
    parser.add_argument("--slow-rw", action="store_true", help="Use full RW split criterion instead of fast approximation.")
    args = parser.parse_args()
    return ReplicationConfig(
        data_path=args.data,
        output_dir=args.output_dir,
        target=args.target,
        n_factors=args.n_factors,
        n_trees=args.n_trees,
        update_freq=args.update_freq,
        random_state=args.seed,
        min_samples_leaf=args.min_samples_leaf,
        mtry_frac=args.mtry_frac,
        ridge_lambda=args.ridge_lambda,
        rw_regul=args.rw_regul,
        subsampling_rate=args.subsampling_rate,
        block_size=args.block_size,
        trend_push=args.trend_push,
        max_depth=args.max_depth,
        min_leaf_frac_of_x=args.min_leaf_frac_of_x,
        fast_rw=not args.slow_rw,
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def diebold_mariano_test(
    y_true: np.ndarray,
    preds_model: np.ndarray,
    preds_bench: np.ndarray,
    h: int,
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    preds_model = np.asarray(preds_model, dtype=float)
    preds_bench = np.asarray(preds_bench, dtype=float)
    mask = ~(np.isnan(y_true) | np.isnan(preds_model) | np.isnan(preds_bench))
    y = y_true[mask]
    pm = preds_model[mask]
    pb = preds_bench[mask]
    n = len(y)
    if n < 2:
        return {"dm_stat": np.nan, "p_value": np.nan, "mean_d": np.nan, "n": n}

    d = (y - pm) ** 2 - (y - pb) ** 2
    mean_d = float(np.mean(d))
    max_lag = max(0, h - 1)
    gamma0 = np.var(d, ddof=0)
    lrv = gamma0
    for lag in range(1, max_lag + 1):
        gamma_l = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        w = 1.0 - lag / (max_lag + 1)
        lrv += 2.0 * w * gamma_l
    lrv = max(lrv, 1e-16)
    hln_factor = max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-16)
    dm_stat = mean_d / math.sqrt(hln_factor * lrv)
    p_value = float(stats.t.cdf(dm_stat, df=n - 1))
    return {"dm_stat": float(dm_stat), "p_value": p_value, "mean_d": mean_d, "n": n}


def base_series_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in META_COLS:
            continue
        if any(col.endswith(sfx) for sfx in ("_MAF1", "_MAF2")):
            continue
        if any(f"_L{k}" in col for k in range(1, 9)):
            continue
        if any(f"_h{k}" in col for k in (1, 2, 4, 6, 8)):
            continue
        if "_yL" in col:
            continue
        if col in FACTOR_COLS:
            continue
        cols.append(col)
    return cols


def reextract_paper_factors(df: pd.DataFrame, target: str, n_factors: int) -> pd.DataFrame:
    """
    Extract paper-style principal components from the core macro panel.
    """
    df = df.copy()
    pca_cols = [c for c in base_series_columns(df) if c != target]
    X = df[pca_cols].astype(float).to_numpy()
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0, ddof=1)
    sigma[sigma == 0] = 1.0
    X_std = (X - mu) / sigma

    if np.isnan(X_std).any():
        col_means = np.nanmean(X_std, axis=0)
        inds = np.where(np.isnan(X_std))
        X_std[inds] = col_means[inds[1]]

    pca = PCA(n_components=n_factors, random_state=0)
    F = pca.fit_transform(X_std)
    f_sigma = F.std(axis=0, ddof=1)
    f_sigma[f_sigma == 0] = 1.0
    F = F / f_sigma

    anchor = "INDPRO" if "INDPRO" in df.columns else target
    anchor_vec = df[anchor].to_numpy(dtype=float)
    for k in range(n_factors):
        corr = np.corrcoef(anchor_vec, F[:, k])[0, 1]
        if np.isfinite(corr) and corr < 0:
            F[:, k] *= -1

    for i in range(n_factors):
        df[f"F{i + 1}"] = F[:, i]

    return df


def ensure_feature_derivatives(df: pd.DataFrame, target: str, n_factors: int) -> pd.DataFrame:
    """
    Rebuild lags and MAF features needed by the paper-style quarterly S_t.
    """
    df = df.copy()

    core_vars = [target] + [f"F{i}" for i in range(1, n_factors + 1)]
    core_vars += [c for c in base_series_columns(df) if c != target]

    seen = set()
    unique_core_vars = []
    for col in core_vars:
        if col in df.columns and col not in seen:
            seen.add(col)
            unique_core_vars.append(col)

    for col in unique_core_vars:
        for lag in range(1, 9):
            lag_col = f"{col}_L{lag}"
            df[lag_col] = df[col].shift(lag)
        df[f"{col}_MAF1"] = df[col].rolling(4, min_periods=4).mean()
        df[f"{col}_MAF2"] = df[col].rolling(8, min_periods=8).mean()

    return df


def build_state_matrix(df: pd.DataFrame, target: str, n_factors: int) -> tuple[pd.DataFrame, list[str]]:
    cols: list[str] = []

    if "trend_t" in df.columns:
        cols.append("trend_t")
    else:
        df = df.copy()
        df["trend_t"] = np.arange(len(df), dtype=float)
        cols.append("trend_t")

    for lag in range(1, 9):
        cols.append(f"{target}_L{lag}")

    for raw_col in sorted(c for c in base_series_columns(df) if c != target):
        cols.append(f"{raw_col}_L1")
        cols.append(f"{raw_col}_L2")

    for factor in [f"F{i}" for i in range(1, n_factors + 1)]:
        for lag in range(1, 9):
            cols.append(f"{factor}_L{lag}")

    for raw_col in sorted(base_series_columns(df) + [target] + [f"F{i}" for i in range(1, n_factors + 1)]):
        maf1 = f"{raw_col}_MAF1"
        maf2 = f"{raw_col}_MAF2"
        if maf1 in df.columns:
            cols.append(maf1)
        if maf2 in df.columns:
            cols.append(maf2)

    cols = [c for c in dict.fromkeys(cols) if c in df.columns]
    s_df = df[cols].dropna().copy()
    return s_df, cols


def build_direct_problem(
    df: pd.DataFrame,
    s_df: pd.DataFrame,
    target: str,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a direct forecasting problem aligned on the realized target date.

    For FA-ARRF:
      X_t = [1, y_t, y_{t-1}, F1_t, F2_t]
      target = y_{t+h}
      OOS membership defined by target date, not origin date
    """
    aligned = df.loc[s_df.index].reset_index(drop=True)
    s_arr = s_df.reset_index(drop=True).to_numpy(dtype=float)
    dates = pd.to_datetime(aligned["observation_date"]).to_numpy()
    y = aligned[target].to_numpy(dtype=float)
    f1 = aligned["F1"].to_numpy(dtype=float)
    f2 = aligned["F2"].to_numpy(dtype=float)

    X_rows = []
    y_rows = []
    S_rows = []
    origin_dates = []
    target_dates = []

    for t in range(1, len(aligned) - horizon):
        x_t = np.array([1.0, y[t], y[t - 1], f1[t], f2[t]], dtype=float)
        target_idx = t + horizon
        if np.isnan(x_t).any() or np.isnan(y[target_idx]) or np.isnan(s_arr[t]).any():
            continue
        X_rows.append(x_t)
        y_rows.append(y[target_idx])
        S_rows.append(s_arr[t])
        origin_dates.append(dates[t])
        target_dates.append(dates[target_idx])

    return (
        np.asarray(X_rows, dtype=float),
        np.asarray(y_rows, dtype=float),
        np.asarray(S_rows, dtype=float),
        np.asarray(origin_dates),
        np.asarray(target_dates),
    )


def standardize_vector(v: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu = float(np.mean(v))
    sigma = float(np.std(v, ddof=1))
    if sigma == 0:
        sigma = 1.0
    return (v - mu) / sigma, mu, sigma


def standardize_matrix(M: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.mean(M, axis=0)
    sigma = np.std(M, axis=0, ddof=1)
    sigma[sigma == 0] = 1.0
    return (M - mu) / sigma, mu, sigma


class MRFTree:
    def __init__(
        self,
        min_samples_leaf: int,
        mtry_frac: float,
        ridge_lambda: float,
        rw_regul: float,
        subsampling_rate: float,
        block_size: int,
        trend_push: float,
        trend_col_idx: int,
        max_depth: int,
        min_leaf_frac_of_x: float,
        fast_rw: bool,
        rng: np.random.Generator,
    ) -> None:
        self.min_samples_leaf = min_samples_leaf
        self.mtry_frac = mtry_frac
        self.ridge_lambda = ridge_lambda
        self.rw_regul = rw_regul
        self.subsampling_rate = subsampling_rate
        self.block_size = block_size
        self.trend_push = trend_push
        self.trend_col_idx = trend_col_idx
        self.max_depth = max_depth
        self.min_leaf_frac_of_x = min_leaf_frac_of_x
        self.fast_rw = fast_rw
        self.rng = rng
        self.root: dict | None = None

    def _regularization(self, k: int) -> np.ndarray:
        return self.ridge_lambda * np.eye(k)

    def _sample_features(self, p: int) -> np.ndarray:
        n = max(1, round(p * self.mtry_frac))
        probs = np.ones(p, dtype=float)
        if 0 <= self.trend_col_idx < p:
            probs[self.trend_col_idx] = max(probs[self.trend_col_idx], float(self.trend_push))
        probs /= probs.sum()
        return self.rng.choice(np.arange(p), size=n, replace=False, p=probs)

    def _block_subsample(self, T: int) -> np.ndarray:
        n_blocks = max(1, int(np.ceil(T / self.block_size)))
        groups = np.minimum(np.arange(T) // self.block_size, n_blocks - 1)
        weights = self.rng.exponential(1.0, size=n_blocks) + 0.1
        obs_weights = weights[groups]
        threshold = np.quantile(obs_weights, 1.0 - self.subsampling_rate)
        return np.sort(np.where(obs_weights > threshold)[0])

    def _neighbors(self, idx: np.ndarray, steps: int, T: int, inbag: np.ndarray) -> np.ndarray:
        nbrs = np.unique(np.concatenate([idx - steps, idx + steps]))
        nbrs = nbrs[(nbrs >= 0) & (nbrs < T)]
        nbrs = nbrs[~np.isin(nbrs, idx)]
        return np.intersect1d(nbrs, inbag)

    def _rw_augment(
        self,
        y: np.ndarray,
        Z: np.ndarray,
        idx: np.ndarray,
        y_full: np.ndarray,
        Z_full: np.ndarray,
        inbag: np.ndarray,
        T: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        n1 = self._neighbors(idx, 1, T, inbag)
        n2 = np.setdiff1d(self._neighbors(idx, 2, T, inbag), n1)
        pieces_y = [y]
        pieces_Z = [Z]
        if len(n1) > 0:
            pieces_y.append(self.rw_regul * y_full[n1])
            pieces_Z.append(self.rw_regul * Z_full[n1])
        if len(n2) > 0:
            pieces_y.append((self.rw_regul ** 2) * y_full[n2])
            pieces_Z.append((self.rw_regul ** 2) * Z_full[n2])
        return np.concatenate(pieces_y), np.vstack(pieces_Z)

    def _solve(self, Z: np.ndarray, y: np.ndarray) -> np.ndarray:
        R = self._regularization(Z.shape[1])
        return np.linalg.solve(Z.T @ Z + R, Z.T @ y)

    def _split_sse_fast(
        self,
        Xb: np.ndarray,
        yb: np.ndarray,
        Sb: np.ndarray,
        node_idx: np.ndarray,
        feat_idx: np.ndarray,
        min_leaf: int,
    ) -> tuple[int, float] | None:
        best_score = np.inf
        best = None
        for j in feat_idx:
            values = np.unique(Sb[node_idx, j])
            if len(values) < 2:
                continue
            for split in values[:-1]:
                left_mask = Sb[node_idx, j] < split
                right_mask = ~left_mask
                if left_mask.sum() < min_leaf or right_mask.sum() < min_leaf:
                    continue
                left_idx = node_idx[left_mask]
                right_idx = node_idx[right_mask]
                b_l = self._solve(Xb[left_idx], yb[left_idx])
                b_r = self._solve(Xb[right_idx], yb[right_idx])
                err = np.sum((yb[left_idx] - Xb[left_idx] @ b_l) ** 2)
                err += np.sum((yb[right_idx] - Xb[right_idx] @ b_r) ** 2)
                if err < best_score:
                    best_score = err
                    best = (int(j), float(split))
        return best

    def _split_sse_full(
        self,
        Xb: np.ndarray,
        yb: np.ndarray,
        Sb: np.ndarray,
        node_idx: np.ndarray,
        feat_idx: np.ndarray,
        min_leaf: int,
        X_full: np.ndarray,
        y_full: np.ndarray,
        inbag: np.ndarray,
        T: int,
    ) -> tuple[int, float] | None:
        best_score = np.inf
        best = None
        for j in feat_idx:
            values = np.unique(Sb[node_idx, j])
            if len(values) < 2:
                continue
            for split in values[:-1]:
                left_mask = Sb[node_idx, j] < split
                right_mask = ~left_mask
                if left_mask.sum() < min_leaf or right_mask.sum() < min_leaf:
                    continue
                left_idx = node_idx[left_mask]
                right_idx = node_idx[right_mask]

                y_la, Z_la = self._rw_augment(yb[left_idx], Xb[left_idx], inbag[left_idx], y_full, X_full, inbag, T)
                y_ra, Z_ra = self._rw_augment(yb[right_idx], Xb[right_idx], inbag[right_idx], y_full, X_full, inbag, T)
                b_l = self._solve(Z_la, y_la)
                b_r = self._solve(Z_ra, y_ra)
                err = np.sum((yb[left_idx] - Xb[left_idx] @ b_l) ** 2)
                err += np.sum((yb[right_idx] - Xb[right_idx] @ b_r) ** 2)
                if err < best_score:
                    best_score = err
                    best = (int(j), float(split))
        return best

    def fit(self, X: np.ndarray, y: np.ndarray, S: np.ndarray) -> None:
        T, K = X.shape
        std_y, mu_y, sig_y = standardize_vector(y)
        std_X_body, mu_X, sig_X = standardize_matrix(X[:, 1:])
        std_X = np.column_stack([np.ones(T), std_X_body])
        std_S, mu_S, sig_S = standardize_matrix(S)

        self.mu_y, self.sig_y = mu_y, sig_y
        self.mu_X, self.sig_X = mu_X, sig_X
        self.mu_S, self.sig_S = mu_S, sig_S
        self.K = K
        self.T = T
        self.std_X = std_X
        self.std_y = std_y
        self.std_S = std_S

        inbag = self._block_subsample(T)
        Xb = std_X[inbag]
        yb = std_y[inbag] + 1.5e-7 * self.rng.normal(size=len(inbag))
        Sb = std_S[inbag]
        self.inbag = inbag

        min_leaf = max(
            self.min_samples_leaf,
            int(2 * self.min_leaf_frac_of_x * (K + 1) + 2),
        )

        root = {
            "idx": np.arange(len(inbag)),
            "depth": 0,
            "split_var": None,
            "split_val": None,
            "left": None,
            "right": None,
            "is_leaf": False,
            "beta_std": None,
            "beta_raw": None,
        }

        stack = [root]
        while stack:
            node = stack.pop()
            idx = node["idx"]
            ib_idx = inbag[idx]
            yy = yb[idx]
            ZZ = Xb[idx]
            y_aug, Z_aug = self._rw_augment(yy, ZZ, ib_idx, std_y, std_X, inbag, T)
            node["beta_std"] = self._solve(Z_aug, y_aug)

            if node["depth"] >= self.max_depth or len(idx) < 2 * min_leaf:
                node["is_leaf"] = True
                continue

            feat_idx = self._sample_features(std_S.shape[1])
            if self.fast_rw:
                best = self._split_sse_fast(Xb, yb, Sb, idx, feat_idx, min_leaf)
            else:
                best = self._split_sse_full(Xb, yb, Sb, idx, feat_idx, min_leaf, std_X, std_y, inbag, T)

            if best is None:
                node["is_leaf"] = True
                continue

            j_star, c_star = best
            left_mask = Sb[idx, j_star] < c_star
            right_mask = ~left_mask
            node["split_var"] = j_star
            node["split_val"] = c_star
            node["left"] = {
                "idx": idx[left_mask],
                "depth": node["depth"] + 1,
                "split_var": None,
                "split_val": None,
                "left": None,
                "right": None,
                "is_leaf": False,
                "beta_std": None,
                "beta_raw": None,
            }
            node["right"] = {
                "idx": idx[right_mask],
                "depth": node["depth"] + 1,
                "split_var": None,
                "split_val": None,
                "left": None,
                "right": None,
                "is_leaf": False,
                "beta_std": None,
                "beta_raw": None,
            }
            stack.extend([node["left"], node["right"]])

        self.root = root
        self._finalize_leaf_betas()

    def _collect_leaves(self, node: dict | None) -> list[dict]:
        if node is None:
            return []
        if node["is_leaf"] or node["left"] is None:
            node["is_leaf"] = True
            return [node]
        return self._collect_leaves(node["left"]) + self._collect_leaves(node["right"])

    def _route_one(self, s_std: np.ndarray) -> dict:
        node = self.root
        assert node is not None
        while not node["is_leaf"]:
            if s_std[node["split_var"]] < node["split_val"]:
                node = node["left"]
            else:
                node = node["right"]
        return node

    def _route_all(self, std_S: np.ndarray) -> np.ndarray:
        leaves = self._collect_leaves(self.root)
        for i, leaf in enumerate(leaves):
            leaf["_id"] = i
        out = np.zeros(len(std_S), dtype=int)
        for i in range(len(std_S)):
            out[i] = self._route_one(std_S[i])["_id"]
        return out

    def _back_transform_beta(self, b_std: np.ndarray) -> np.ndarray:
        b = b_std.copy()
        out = np.empty_like(b)
        out[0] = b[0] * self.sig_y + self.mu_y
        for k in range(1, len(b)):
            out[k] = b[k] * self.sig_y / self.sig_X[k - 1]
        for k in range(1, len(b)):
            out[0] -= out[k] * self.mu_X[k - 1]
        return out

    def _finalize_leaf_betas(self) -> None:
        leaves = self._collect_leaves(self.root)
        leaf_ids = self._route_all(self.std_S)
        for leaf in leaves:
            idx_all = np.where(leaf_ids == leaf["_id"])[0]
            if len(idx_all) < self.min_samples_leaf:
                leaf["beta_raw"] = self._back_transform_beta(leaf["beta_std"])
                continue
            y_aug, Z_aug = self._rw_augment(
                self.std_y[idx_all],
                self.std_X[idx_all],
                idx_all,
                self.std_y,
                self.std_X,
                np.arange(self.T),
                self.T,
            )
            b_std = self._solve(Z_aug, y_aug)
            leaf["beta_raw"] = self._back_transform_beta(b_std)

    def predict(self, X_new: np.ndarray, S_new: np.ndarray) -> np.ndarray:
        S_std = (S_new - self.mu_S) / self.sig_S
        out = np.empty(len(X_new))
        for i in range(len(X_new)):
            leaf = self._route_one(S_std[i])
            out[i] = float(X_new[i] @ leaf["beta_raw"])
        return out

    def get_betas(self, S_new: np.ndarray) -> np.ndarray:
        S_std = (S_new - self.mu_S) / self.sig_S
        out = np.empty((len(S_new), self.K))
        for i in range(len(S_new)):
            out[i] = self._route_one(S_std[i])["beta_raw"]
        return out


class MacroeconomicRandomForest:
    def __init__(self, cfg: ReplicationConfig, trend_col_idx: int) -> None:
        self.cfg = cfg
        self.trend_col_idx = trend_col_idx
        self.trees: list[MRFTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray, S: np.ndarray) -> None:
        seed_seq = np.random.SeedSequence(self.cfg.random_state)
        child_seeds = seed_seq.spawn(self.cfg.n_trees)
        self.trees = []
        for seed in child_seeds:
            rng = np.random.default_rng(seed)
            tree = MRFTree(
                min_samples_leaf=self.cfg.min_samples_leaf,
                mtry_frac=self.cfg.mtry_frac,
                ridge_lambda=self.cfg.ridge_lambda,
                rw_regul=self.cfg.rw_regul,
                subsampling_rate=self.cfg.subsampling_rate,
                block_size=self.cfg.block_size,
                trend_push=self.cfg.trend_push,
                trend_col_idx=self.trend_col_idx,
                max_depth=self.cfg.max_depth,
                min_leaf_frac_of_x=self.cfg.min_leaf_frac_of_x,
                fast_rw=self.cfg.fast_rw,
                rng=rng,
            )
            tree.fit(X, y, S)
            self.trees.append(tree)

    def predict(self, X_new: np.ndarray, S_new: np.ndarray) -> np.ndarray:
        preds = np.array([tree.predict(X_new, S_new) for tree in self.trees])
        return np.nanmean(preds, axis=0)

    def get_betas(self, S_new: np.ndarray) -> np.ndarray:
        betas = np.array([tree.get_betas(S_new) for tree in self.trees])
        return np.nanmean(betas, axis=0)


def oos_mask_from_target_dates(target_dates: np.ndarray) -> np.ndarray:
    target_dates = pd.to_datetime(target_dates)
    return (target_dates >= PAPER_OOS_START) & (target_dates <= PAPER_OOS_END)


def expanding_window_mrf_forecast(
    X: np.ndarray,
    y: np.ndarray,
    S: np.ndarray,
    origin_dates: np.ndarray,
    target_dates: np.ndarray,
    model_factory,
    horizon: int,
    update_freq: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    oos_mask = oos_mask_from_target_dates(target_dates)
    oos_idx = np.where(oos_mask)[0]
    preds = []
    actuals = []
    betas = []

    current_model = None
    for step, row_idx in enumerate(oos_idx):
        if step == 0 or step % update_freq == 0:
            train_end = max(0, row_idx - horizon + 1)
            current_model = model_factory()
            current_model.fit(X[:train_end], y[:train_end], S[:train_end])

        assert current_model is not None
        preds.append(current_model.predict(X[row_idx : row_idx + 1], S[row_idx : row_idx + 1])[0])
        actuals.append(y[row_idx])
        betas.append(current_model.get_betas(S[row_idx : row_idx + 1])[0])

    return np.asarray(preds), np.asarray(actuals), np.asarray(betas)


def ar4_direct_forecast(
    y_full: np.ndarray,
    origin_dates: np.ndarray,
    target_dates: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    oos_idx = np.where(oos_mask_from_target_dates(target_dates))[0]
    preds = []
    actuals = []
    for row_idx in oos_idx:
        train_end = max(4, row_idx - horizon + 1)
        X_train = []
        y_train = []
        for s in range(3, train_end):
            if s + horizon >= len(y_full):
                break
            X_train.append([1.0, y_full[s], y_full[s - 1], y_full[s - 2], y_full[s - 3]])
            y_train.append(y_full[s + horizon])
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
        x_now = np.asarray([1.0, y_full[row_idx], y_full[row_idx - 1], y_full[row_idx - 2], y_full[row_idx - 3]])
        preds.append(float(x_now @ beta))
        actuals.append(y_full[row_idx + horizon])
    return np.asarray(preds), np.asarray(actuals)


def fa_ar_direct_forecast(
    y_full: np.ndarray,
    f1_full: np.ndarray,
    f2_full: np.ndarray,
    target_dates: np.ndarray,
    horizon: int,
) -> np.ndarray:
    oos_idx = np.where(oos_mask_from_target_dates(target_dates))[0]
    preds = []
    for row_idx in oos_idx:
        train_end = max(4, row_idx - horizon + 1)
        X_train = []
        y_train = []
        for s in range(1, train_end):
            if s + horizon >= len(y_full):
                break
            X_train.append([1.0, y_full[s], y_full[s - 1], f1_full[s], f2_full[s]])
            y_train.append(y_full[s + horizon])
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        beta, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
        x_now = np.asarray([1.0, y_full[row_idx], y_full[row_idx - 1], f1_full[row_idx], f2_full[row_idx]])
        preds.append(float(x_now @ beta))
    return np.asarray(preds)


def plot_forecasts(
    frame: pd.DataFrame,
    horizon: int,
    output_path: Path,
    target_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    scale = 1.0

    ax.plot(frame["target_date"], frame["actual"] * scale, color="black", lw=2.0, label="Realized")
    ax.plot(frame["target_date"], frame["fa_arrf"] * scale, color="#d18f2a", lw=2.2, label="FA-ARRF")
    ax.plot(frame["target_date"], frame["ar4"] * scale, color="#4c72b0", lw=1.5, alpha=0.9, label="AR(4)")

    ax.axvspan(pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-01"), color="#c44e52", alpha=0.18, zorder=0)
    ax.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.7)
    ax.grid(True, linestyle="-", linewidth=0.6, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%y"))
    ax.set_title(f"{target_label}: h={horizon}", fontsize=12, weight="bold")
    ax.set_ylabel(target_label)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rmse_ratios(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.bar(summary_df["horizon"].astype(str), summary_df["fa_arrf_ratio"], color="#1a8a5a", width=0.55, label="FA-ARRF")
    ax.bar(summary_df["horizon"].astype(str), summary_df["fa_ar_ratio"], bottom=0, color="#d9d9d9", alpha=0.0)
    ax.axhline(1.0, color="black", lw=1.2)
    ax.set_ylim(0.0, max(1.4, float(summary_df["fa_arrf_ratio"].max()) + 0.1))
    ax.set_xlabel("Horizon (quarters)")
    ax.set_ylabel("RMSE / RMSE_AR(4)")
    ax.set_title("UR (change): paper-style RMSE ratios", fontsize=12, weight="bold")
    ax.grid(True, axis="y", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gtvp_paths(frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8))
    titles = [
        ("Intercept", "beta_0"),
        ("Persistence", None),
        ("Real Activity Factor", "beta_3"),
        ("Forward-Looking Factor", "beta_4"),
    ]
    for ax, (title, key) in zip(axes.flatten(), titles):
        if key is None:
            series = frame["beta_1"] + frame["beta_2"]
        else:
            series = frame[key]
        ax.plot(frame["target_date"], series, color="#1a7a4a", lw=1.4)
        ax.axvspan(pd.Timestamp("2007-12-01"), pd.Timestamp("2009-06-01"), color="#ffb0b0", alpha=0.45, linewidth=0)
        ax.axhline(0, color="lightgray", lw=0.7)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(mdates.YearLocator(3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.random_state)

    df = pd.read_excel(cfg.data_path).reset_index(drop=True)
    if cfg.target not in df.columns:
        raise ValueError(f"Target '{cfg.target}' is not in {cfg.data_path.name}.")

    # Build a paper-style working panel from the workbook.
    df = reextract_paper_factors(df, target=cfg.target, n_factors=cfg.n_factors)
    df = ensure_feature_derivatives(df, target=cfg.target, n_factors=cfg.n_factors)
    if "trend_t" not in df.columns:
        df["trend_t"] = np.arange(len(df), dtype=float)

    s_df, s_cols = build_state_matrix(df, target=cfg.target, n_factors=cfg.n_factors)
    trend_col_idx = s_cols.index("trend_t")

    summary_rows = []
    beta_plot_frame = None

    for horizon in cfg.horizons:
        X, y_direct, S, origin_dates, target_dates = build_direct_problem(df, s_df, target=cfg.target, horizon=horizon)
        oos_mask = oos_mask_from_target_dates(target_dates)
        oos_idx = np.where(oos_mask)[0]

        y_full = y_direct
        f1_full = X[:, 3]
        f2_full = X[:, 4]

        def build_model() -> MacroeconomicRandomForest:
            return MacroeconomicRandomForest(cfg=cfg, trend_col_idx=trend_col_idx)

        fa_arrf_preds, actuals, betas = expanding_window_mrf_forecast(
            X=X,
            y=y_direct,
            S=S,
            origin_dates=origin_dates,
            target_dates=target_dates,
            model_factory=build_model,
            horizon=horizon,
            update_freq=cfg.update_freq,
        )

        ar4_preds, ar4_actuals = ar4_direct_forecast(
            y_full=y_full,
            origin_dates=origin_dates,
            target_dates=target_dates,
            horizon=horizon,
        )
        fa_ar_preds = fa_ar_direct_forecast(
            y_full=y_full,
            f1_full=f1_full,
            f2_full=f2_full,
            target_dates=target_dates,
            horizon=horizon,
        )

        oos_target_dates = pd.to_datetime(target_dates[oos_idx])
        oos_origin_dates = pd.to_datetime(origin_dates[oos_idx])
        frame = pd.DataFrame(
            {
                "origin_date": oos_origin_dates,
                "target_date": oos_target_dates,
                "actual": actuals,
                "fa_arrf": fa_arrf_preds,
                "fa_ar": fa_ar_preds,
                "ar4": ar4_preds,
            }
        )
        for k in range(betas.shape[1]):
            frame[f"beta_{k}"] = betas[:, k]

        frame.to_csv(cfg.output_dir / f"ur_h{horizon}_forecast.csv", index=False)
        plot_forecasts(frame, horizon, cfg.output_dir / f"ur_h{horizon}_forecast.png", "UR change")

        ar_rmse = rmse(ar4_actuals, ar4_preds)
        fa_arrf_rmse = rmse(actuals, fa_arrf_preds)
        fa_ar_rmse = rmse(actuals, fa_ar_preds)
        dm = diebold_mariano_test(actuals, fa_arrf_preds, ar4_preds, h=horizon)

        summary_rows.append(
            {
                "horizon": horizon,
                "ar_rmse": ar_rmse,
                "fa_arrf_rmse": fa_arrf_rmse,
                "fa_ar_rmse": fa_ar_rmse,
                "fa_arrf_ratio": fa_arrf_rmse / ar_rmse,
                "fa_ar_ratio": fa_ar_rmse / ar_rmse,
                "paper_fa_arrf_ratio": PAPER_UR_FA_ARRF_RMSE_TARGETS.get(horizon, np.nan),
                "dm_stat": dm["dm_stat"],
                "dm_p_value": dm["p_value"],
            }
        )

        if horizon == 1:
            beta_plot_frame = frame.copy()

    summary_df = pd.DataFrame(summary_rows).sort_values("horizon").reset_index(drop=True)
    summary_df.to_csv(cfg.output_dir / "ur_summary.csv", index=False)
    plot_rmse_ratios(summary_df, cfg.output_dir / "ur_rmse_ratios.png")

    if beta_plot_frame is not None:
        plot_gtvp_paths(beta_plot_frame, cfg.output_dir / "ur_h1_gtvp.png")

    meta = {
        "target": cfg.target,
        "oos_window": [str(PAPER_OOS_START.date()), str(PAPER_OOS_END.date())],
        "horizons": list(cfg.horizons),
        "n_trees": cfg.n_trees,
        "update_freq": cfg.update_freq,
        "paper_note": (
            "Table 4 unemployment FA-ARRF targets are left as NaN unless you provide "
            "verified values from the paper or appendix."
        ),
    }
    (cfg.output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
