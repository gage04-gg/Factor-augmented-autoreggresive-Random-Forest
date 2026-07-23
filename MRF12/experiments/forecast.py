"""
forecast.py — Data construction and forecasting pipeline for MRF replication.

Implements:
  build_St()            — construct the S_t state-space matrix from FRED-QD data
  create_X_y_S()        — align X_t, y_{t+h}, S_t, sample flags, and dates
  expanding_window_oos()— expanding-window OOS loop (Algorithm 1 steps 3-4)
  ar4_benchmark()       — expanding-window AR(4) baseline (same index alignment)
  compute_rmse()        — NaN-safe RMSE

Paper: Goulet Coulombe (2024), Sections 3 & App A.6
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ─────────────────────────────────────────────────────────────────────────────
# Series excluded from PCA factor extraction and from S_t construction.
#
# These fall into three groups:
#   1. _yL*  — pre-computed lags of target variables (GDP, UR, CPI, IR, SPREAD).
#              Redundant: build_St already constructs target lags explicitly.
#   2. _h*   — direct forecast targets (GDP_h1, UR_h2 …).
#              Must be excluded — they are look-ahead values that would leak
#              future information into both PCA factors and tree splits.
#   3. Non-FRED-QD additions — NIKKEI225, NASDAQCOM were not in the paper's
#              original FRED-QD vintage (~248 series).  Excluding them brings
#              the PCA input set to ~243 series, matching the paper's scope.
#
# After exclusion the PCA input has 243 series vs the paper's 248; the small
# gap reflects minor vintage differences in FRED-QD since 2024.
# ─────────────────────────────────────────────────────────────────────────────
_S_EXCLUDE_PATTERNS = ("_yL", "_h1", "_h2", "_h4", "_h6", "_h8")
_S_EXCLUDE_EXACT    = frozenset({"NIKKEI225", "NASDAQCOM", "quarter"})


def _is_extra_series(col: str) -> bool:
    """Return True if col should be excluded from PCA and S_t."""
    if col in _S_EXCLUDE_EXACT:
        return True
    for pat in _S_EXCLUDE_PATTERNS:
        if pat in col:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Factor re-extraction with unit-variance normalisation
# ─────────────────────────────────────────────────────────────────────────────

def reextract_factors(
    data:       pd.DataFrame,
    n_factors:  int = 5,
    target:     str = "GDPC1",
    inplace:    bool = True,
) -> pd.DataFrame:
    """
    Re-extract PCA factors from the cleaned FRED-QD series and replace the
    F1…F5 columns (and their lags/MAF derivatives) in *data* with the new
    unit-variance normalised factors.

    Why this is needed
    ------------------
    The original F1/F2 columns in the Excel file have std ≈ 7.6 / 4.3 because
    they are raw sklearn PCA scores computed over ALL 310 base series including:
      • 40 _yL* pre-computed target lags
      • 25 _h* forecast-target columns
      • NIKKEI225, NASDAQCOM (not in the paper's FRED-QD vintage)
    The paper (Goulet Coulombe 2024, App A.1) uses factors extracted from ~248
    FRED-QD series and normalises them to UNIT VARIANCE before entering X_t
    (standard FA-VAR convention, Stock & Watson 2002).

    After re-extraction the γ coefficients in get_betas() will be on the same
    scale as paper Figure 22 (+0.010–0.020 range instead of +0.0003).

    Parameters
    ----------
    data      : full model-ready DataFrame (read from Excel).
    n_factors : number of PC factors to extract (default 5, matches paper).
    target    : GDP column name; excluded from PCA input.
    inplace   : if True, modify data in-place and return it; else return a copy.

    Returns
    -------
    DataFrame with F1…F5 columns (and their _L* and _MAF* derivatives)
    replaced by the re-extracted, unit-variance-normalised factors.
    """
    if not inplace:
        data = data.copy()

    # ── Identify PCA input columns ─────────────────────────────────────────
    suffixes = ("_L1","_L2","_L3","_L4","_L5","_L6","_L7","_L8","_MAF1","_MAF2")
    meta     = {"observation_date","quarter","sample_flag","trend_t",
                "F1","F2","F3","F4","F5", target}
    derived  = {c for c in data.columns if c.endswith(suffixes)}

    pca_cols = [
        c for c in data.columns
        if c not in meta
        and c not in derived
        and not _is_extra_series(c)
    ]

    X = data[pca_cols].values.astype(float)
    # Standardise each series to mean=0, std=1 before PCA (Stock & Watson 2002)
    col_mean = X.mean(axis=0)
    col_std  = X.std(axis=0, ddof=1)
    col_std[col_std == 0] = 1.0
    X_std = (X - col_mean) / col_std

    # ── Extract factors ────────────────────────────────────────────────────
    pca      = PCA(n_components=n_factors, random_state=0)
    factors  = pca.fit_transform(X_std)          # shape (T, n_factors)

    # Normalise to unit variance (match paper convention)
    f_std = factors.std(axis=0, ddof=1)
    f_std[f_std == 0] = 1.0
    factors = factors / f_std

    # Sign convention: each factor is flipped so it correlates non-negatively
    # with INDPRO (real-activity anchor used in the paper).
    if "INDPRO" in data.columns:
        indpro = data["INDPRO"].values
        for k in range(n_factors):
            if np.corrcoef(indpro, factors[:, k])[0, 1] < 0:
                factors[:, k] *= -1

    # ── Replace F1…F5 and all their derivatives in data ───────────────────
    T = len(data)
    for k in range(n_factors):
        fi = f"F{k + 1}"
        data[fi] = factors[:, k]

        # Recompute lags stored in the DataFrame (_L1 … _L8)
        for lag in range(1, 9):
            col = f"{fi}_L{lag}"
            if col in data.columns:
                data[col] = pd.Series(factors[:, k]).shift(lag).values

        # Recompute moving-average filters (_MAF1 = 4-quarter MA, _MAF2 = 8-quarter MA)
        for mw, tag in ((4, "_MAF1"), (8, "_MAF2")):
            col = f"{fi}{tag}"
            if col in data.columns:
                data[col] = (pd.Series(factors[:, k])
                             .rolling(mw, min_periods=mw).mean().values)

    n_pca_series = len(pca_cols)
    ev = pca.explained_variance_ratio_
    print(f"  [reextract_factors] PCA on {n_pca_series} series  "
          f"(excluded: _yL, _h, NIKKEI225, NASDAQCOM)")
    print(f"  Explained variance: " +
          "  ".join(f"F{k+1}={ev[k]:.3f}" for k in range(n_factors)))
    print(f"  Factor std after normalisation: " +
          "  ".join(f"F{k+1}={data[f'F{k+1}'].std():.4f}" for k in range(n_factors)))

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Priority leading indicators for mtry upweighting (FA-ARRF specification)
# These are financial/real-activity predictors known to lead GDP.
# Passed as priority_col_idxs to MRF to increase their mtry selection probability.
# ─────────────────────────────────────────────────────────────────────────────
PRIORITY_VARS = [
    # Yield spreads (strong GDP leading indicators)
    "GS10TB3Mx_L1", "GS10TB3Mx_L2", "GS10TB3Mx_MAF1", "GS10TB3Mx_MAF2",
    "CPF3MTB3Mx_L1", "CPF3MTB3Mx_L2", "CPF3MTB3Mx_MAF1",
    "TB3SMFFM_L1",   "TB3SMFFM_L2",   "T5YFFM_MAF1",
    # Housing starts (leading real activity)
    "PERMIT_L1",     "PERMIT_L2",     "PERMIT_MAF1",   "PERMIT_MAF2",
    # Factors and factor lags
    "F1", "F2", "F1_L1", "F2_L1", "F1_L2", "F2_L2",
    # Consumer sentiment
    "UMCSENTx_L1",   "UMCSENTx_MAF1",
    # Manufacturing hours (real activity)
    "AWHMAN_L1",     "AWHMAN_MAF1",
    # GDP own lags
    "GDPC1_L1", "GDPC1_L2", "GDPC1_L3", "GDPC1_L4",
    # Time trend (Trend Push target)
    "trend_t",
]


# ─────────────────────────────────────────────────────────────────────────────
# S_t construction
# ─────────────────────────────────────────────────────────────────────────────

def build_St(
    data:      pd.DataFrame,
    target:    str = "GDPC1",
    n_y_lags:  int = 2,
) -> tuple:
    """
    Construct the S_t state-space matrix used for tree splitting.

    Column order (matches paper App A.3 / Section 3):
      [trend_t]  +  [raw_L1, raw_L2 for all non-target, non-factor series]
      + [F1_L1…F5_Lk]  + [*_MAF1, *_MAF2]  + [F1…F5]  + [target_L1…target_Ln]

    trend_t is always column 0 — required by MRF.trend_col_idx = 0.

    Series excluded from S_t (and from PCA — see reextract_factors):
      • _yL* cols  — redundant pre-computed target lags
      • _h*  cols  — look-ahead forecast targets (leakage)
      • NIKKEI225, NASDAQCOM — not in paper's FRED-QD vintage

    Returns
    -------
    S_df            : pd.DataFrame aligned to rows where all S_t cols are non-NaN
    S_col_names     : list of column names (used for diagnostics)
    priority_col_idxs : list of int — column indices of PRIORITY_VARS in S_df
    """
    cols = []

    # 1. Time trend (must be first for trend_col_idx=0 in MRF)
    if "trend_t" in data.columns:
        cols.append("trend_t")

    # 2. L1 and L2 of all non-target, non-factor raw series
    #    Extra series (_yL, _h, NIKKEI225, NASDAQCOM) are excluded here.
    raw_l1l2 = sorted([
        c for c in data.columns
        if (c.endswith("_L1") or c.endswith("_L2"))
        and not any(c.startswith(f"F{i}_") for i in range(1, 6))
        and target not in c
        and not _is_extra_series(c)
    ])
    cols += raw_l1l2

    # 3. Factor lags (F1_L1, F1_L2, ..., F5_Lk)
    for fi in ["F1", "F2", "F3", "F4", "F5"]:
        cols += sorted([c for c in data.columns if c.startswith(fi + "_L")])

    # 4. Moving-average filters (MAF1, MAF2) — extra series excluded
    cols += sorted([c for c in data.columns
                    if (c.endswith("_MAF1") or c.endswith("_MAF2"))
                    and not _is_extra_series(c)])

    # 5. Current-period factors (F1…F5) — without lags
    for fi in ["F1", "F2", "F3", "F4", "F5"]:
        if fi in data.columns and fi not in cols:
            cols.append(fi)

    # 6. Target lags (y_{t-1} … y_{t-n_y_lags})
    extra_lag_dfs = []
    for lag in range(1, n_y_lags + 1):
        col_name = f"{target}_L{lag}"
        if col_name in data.columns:
            if col_name not in cols:
                cols.append(col_name)
        else:
            # Compute the lag on-the-fly if not already in data
            extra_lag_dfs.append(
                data[target].shift(lag).rename(col_name).to_frame()
            )

    # De-duplicate while preserving order; drop cols not in data
    cols = list(dict.fromkeys([c for c in cols if c in data.columns]))

    # Concatenate and dropna
    parts = [data[cols]]
    if extra_lag_dfs:
        parts += extra_lag_dfs
    S_df = pd.concat(parts, axis=1).dropna()

    S_col_names = list(S_df.columns)

    # Priority column indices (for mtry upweighting in MRF)
    priority_col_idxs = [
        S_col_names.index(v)
        for v in PRIORITY_VARS
        if v in S_col_names
    ]

    return S_df, S_col_names, priority_col_idxs


# ─────────────────────────────────────────────────────────────────────────────
# X_t / y_{t+h} / S_t alignment
# ─────────────────────────────────────────────────────────────────────────────

def create_X_y_S(
    data_aligned: pd.DataFrame,
    S_df:         pd.DataFrame,
    target:       str = "GDPC1",
    h:            int = 1,
) -> tuple:
    """
    Build aligned arrays for the direct h-step-ahead forecasting problem.

    Linear part X_t = [1, y_{t-1}, y_{t-2}, F1_{t-1}, F2_{t-1}]
    (FA-ARRF specification — factors extracted from FRED-QD, paper Section 3)

    NOTE: call reextract_factors(data) before this function to ensure F1/F2
    are unit-variance normalised PCA scores (std=1), matching the paper's
    FA-VAR convention.  After that, back-transformed gamma coefficients will
    be on the same scale as paper Figure 22.

    Target: y_{t+h}  (direct multi-step, not iterated)

    Parameters
    ----------
    data_aligned : DataFrame with columns including target, F1, F2,
                   sample_flag, observation_date
    S_df         : DataFrame of S_t (output of build_St, already dropna'd)
    target       : name of GDP growth column
    h            : forecast horizon in quarters

    Returns
    -------
    X      : (N, 5)   float64
    y      : (N,)     float64
    S      : (N, p)   float64
    flags  : (N,)     object — 'train', 'oos', 'post_oos'
    dates  : (N,)     datetime64 — date of the conditioning observation (t, not t+h)
    """
    y_raw  = data_aligned[target].values
    f1     = data_aligned["F1"].values
    f2     = data_aligned["F2"].values
    flags_ = data_aligned["sample_flag"].values
    dates_ = pd.to_datetime(data_aligned["observation_date"].values)
    S_arr  = S_df.values

    n_lags = 2   # need y_{t-1} and y_{t-2} in X_t
    T      = len(data_aligned)

    X_rows, y_rows, S_rows, flag_rows, date_rows = [], [], [], [], []

    for t in range(n_lags, T - h):
        # X_t: linear part (paper FA-ARRF)
        X_rows.append([1.0, y_raw[t - 1], y_raw[t - 2], f1[t - 1], f2[t - 1]])
        # Target: h steps ahead
        y_rows.append(y_raw[t + h])
        # State space at time t (used to route the tree)
        S_rows.append(S_arr[t])
        # Sample flag at time t (not t+h — avoids look-ahead)
        flag_rows.append(flags_[t])
        date_rows.append(dates_[t])

    return (
        np.array(X_rows,    dtype=float),
        np.array(y_rows,    dtype=float),
        np.array(S_rows,    dtype=float),
        np.array(flag_rows),
        np.array(date_rows),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Expanding-window OOS loop
# ─────────────────────────────────────────────────────────────────────────────

def expanding_window_oos(
    X:           np.ndarray,
    y:           np.ndarray,
    S:           np.ndarray,
    flags:       np.ndarray,
    model,                       # MRF instance
    update_freq: int  = 8,       # refit every update_freq OOS steps (8 = 2 years)
    horizon:    int  = 1,        # direct-forecast horizon used to build y
    S_col_names: list = None,
    print_diag:  bool = True,
) -> tuple:
    """
    Expanding-window out-of-sample forecasting loop.

    For each OOS observation t:
      - Every update_freq steps (or at the first step), refit the MRF on all
        observations whose direct target y_{s+h} is already observed at time t.
        In aligned-array indexing this means rows s <= t - h, or
        X[: t - h + 1] in Python slicing.
      - Predict: y_hat_t = model.predict(X[t], S[t])
      - Extract GTVPs: beta_t = model.get_betas(S[t])

    Parameters
    ----------
    X, y, S   : arrays from create_X_y_S
    flags     : sample_flag array from create_X_y_S
    model     : unfitted MRF instance (will be mutated in-place)
    update_freq: refit interval in OOS quarters (default 8 = 2 years)
    horizon   : direct-forecast horizon used to create y (default 1)
    S_col_names: if provided, print split diagnostics after first fit
    print_diag : whether to call model.print_diagnostics after first fit

    Returns
    -------
    preds   : (n_oos,) float64 — model predictions
    actuals : (n_oos,) float64 — realised y_{t+h}
    betas   : (n_oos, K) float64 — GTVP paths
    """
    oos_idx   = np.where(flags == "oos")[0]
    oos_start = oos_idx[0]
    oos_end   = oos_idx[-1]
    n_oos     = len(oos_idx)

    preds, actuals, betas = [], [], []

    for step, t in enumerate(range(oos_start, oos_end + 1)):
        # Refit condition: first OOS step OR every update_freq steps
        if step == 0 or step % update_freq == 0:
            # Only rows whose direct targets are observed by forecast-origin t
            train_end = max(0, t - horizon + 1)
            train_mask = (np.arange(train_end) < train_end) & (flags[:train_end] != "post_oos")
            n_train    = train_mask.sum()
            print(f"  [OOS step {step + 1:3d}/{n_oos}] "
                  f"Fitting on t={t} obs ({n_train} effective train) ...")

            model.fit(X[:train_end], y[:train_end], S[:train_end])

            # Diagnostics on first fit only
            if print_diag and step == 0 and S_col_names is not None:
                model.print_diagnostics(S_col_names)

        # Predict for observation t (Algorithm 1 step 3)
        y_pred = model.predict(X[t:t + 1], S[t:t + 1])[0]
        # Extract beta_t (Algorithm 1 step 4)
        beta_t = model.get_betas(S[t:t + 1])[0]

        preds.append(y_pred)
        actuals.append(y[t])
        betas.append(beta_t)

    return np.array(preds), np.array(actuals), np.array(betas)


# ─────────────────────────────────────────────────────────────────────────────
# AR(4) benchmark
# ─────────────────────────────────────────────────────────────────────────────

def ar4_benchmark(
    data_aligned: pd.DataFrame,
    flags:        np.ndarray,
    target:       str = "GDPC1",
    h:            int = 1,
    n_lags_X:     int = 4,
) -> tuple:
    """
    Expanding-window AR(4) OLS benchmark, aligned with create_X_y_S indexing.

    Uses the same OOS period as create_X_y_S (flags == 'oos').
    At each OOS step t, fits AR(4) on all available IS data and predicts y_{t+h}.

    Parameters
    ----------
    data_aligned : same DataFrame passed to create_X_y_S
    flags        : the FLAG ARRAY FROM data_aligned['sample_flag'] (full sample,
                   not pre-sliced) — this function re-creates the alignment internally
    target       : column name for GDP growth
    h            : forecast horizon
    n_lags_X     : number of AR lags (default 4)

    Returns
    -------
    preds   : (n_oos,) float64 — AR(4) predictions
    actuals : (n_oos,) float64 — realised y_{t+h}
    """
    from numpy.linalg import lstsq

    y_raw  = data_aligned[target].values
    flags_ = data_aligned["sample_flag"].values
    T      = len(data_aligned)
    n_lags = 2   # must match create_X_y_S (to align OOS indices)

    # Re-create row alignment identical to create_X_y_S
    flag_aligned = np.array([flags_[t] for t in range(n_lags, T - h)])
    oos_idx      = np.where(flag_aligned == "oos")[0]

    preds, actuals = [], []

    for i in oos_idx:
        t = i + n_lags   # absolute index in data_aligned

        # Build expanding training set using only rows whose h-step target is
        # already observed by forecast-origin t.
        X_ar, y_ar = [], []
        for s in range(n_lags_X, t - h + 1):
            X_ar.append([1.0] + [y_raw[s - l] for l in range(1, n_lags_X + 1)])
            y_ar.append(y_raw[s + h])

        X_ar = np.array(X_ar, dtype=float)
        y_ar = np.array(y_ar, dtype=float)

        if len(y_ar) < n_lags_X + 2:   # need at least one more obs than params
            preds.append(np.nan)
            actuals.append(y_raw[t + h])
            continue

        beta_ar, _, _, _ = lstsq(X_ar, y_ar, rcond=None)
        x_pred = [1.0] + [y_raw[t - l] for l in range(1, n_lags_X + 1)]
        preds.append(float(np.dot(x_pred, beta_ar)))
        actuals.append(y_raw[t + h])

    return np.array(preds, dtype=float), np.array(actuals, dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# RMSE
# ─────────────────────────────────────────────────────────────────────────────

def compute_rmse(y_true, y_pred) -> float:
    """Root mean squared error, NaN-pair safe."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))
