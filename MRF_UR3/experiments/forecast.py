import numpy as np
import pandas as pd


PRIORITY_VARS = [
    # ── Labor market (MOST IMPORTANT) ─────────────────────────────
    "UNRATE_L1", "UNRATE_L2", "UNRATE_L3", "UNRATE_L4",
    "UNRATE_MAF1", "UNRATE_MAF2",

    "UEMP5TO14_L1", "UEMP5TO14_L2",
    "UEMP15OV_L1",  "UEMP15OV_L2",
    "UEMP27OV_L1",  "UEMP27OV_L2",

    "CLAIMSx_L1", "CLAIMSx_L2", "CLAIMSx_MAF1",

    "PAYEMS_L1", "PAYEMS_L2", "PAYEMS_MAF1",

    "AWHMAN_L1", "AWHMAN_MAF1",   # hours worked (leading labor signal)

    # ── Macroeconomic leading indicators (still important) ────────
    "GS10TB3Mx_L1", "GS10TB3Mx_L2", "GS10TB3Mx_MAF1",
    "TB3SMFFM_L1",  "TB3SMFFM_L2",

    "PERMIT_L1", "PERMIT_L2", "PERMIT_MAF1",

    # ── Factors ───────────────────────────────────────────────────
    "F1", "F2", "F1_L1", "F2_L1",

    # ── Sentiment ─────────────────────────────────────────────────
    "UMCSENTx_L1", "UMCSENTx_MAF1",

    # ── Target lags (VERY IMPORTANT for persistence) ──────────────
    "UNRATE_L1", "UNRATE_L2", "UNRATE_L3", "UNRATE_L4",

    # ── Trend (structural breaks) ─────────────────────────────────
    "trend_t",
]


# ─────────────────────────────────────────────────────────────────────────────
# S_t construction
# ─────────────────────────────────────────────────────────────────────────────

def build_St(
    data:      pd.DataFrame,
    target:    str = "UNRATE",
    n_y_lags:  int = 8,
) -> tuple:
    """
    Construct the S_t state-space matrix used for tree splitting.

    Column order (matches paper App A.3 / Section 3):
      [trend_t]  +  [raw_L1, raw_L2 for all non-target, non-factor series]
      + [F1_L1…F5_Lk]  + [*_MAF1, *_MAF2]  + [F1…F5]  + [target_L1…target_Ln]

    trend_t is always column 0 — required by MRF.trend_col_idx = 0.

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
    raw_l1l2 = sorted([
        c for c in data.columns
        if (c.endswith("_L1") or c.endswith("_L2"))
        and not any(c.startswith(f"F{i}_") for i in range(1, 6))
        and target not in c
    ])
    cols += raw_l1l2

    # 3. Factor lags (F1_L1, F1_L2, ..., F5_Lk)
    for fi in ["F1", "F2", "F3", "F4", "F5"]:
        cols += sorted([c for c in data.columns if c.startswith(fi + "_L")])

    # 4. Moving-average filters (MAF1, MAF2)
    cols += sorted([c for c in data.columns
                    if c.endswith("_MAF1") or c.endswith("_MAF2")])

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
    target:       str = "UNRATE",
    h:            int = 1,
) -> tuple:
    """
    Build aligned arrays for the direct h-step-ahead forecasting problem.

    Linear part X_t = [1, y_{t-1}, y_{t-2}, F1_{t-1}, F2_{t-1}]
    (FA-ARRF specification — factors extracted from FRED-QD, paper Section 3)

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
    S_col_names: list = None,
    print_diag:  bool = True,
) -> tuple:
    """
    Expanding-window out-of-sample forecasting loop.

    For each OOS observation t:
      - Every update_freq steps (or at the first step), refit the MRF on all
        observations strictly before t (i.e., y[0:t], X[0:t], S[0:t]).
        Only 'train' and 'oos' rows contribute to training (post_oos excluded).
      - Predict: y_hat_t = model.predict(X[t], S[t])
      - Extract GTVPs: beta_t = model.get_betas(S[t])

    Parameters
    ----------
    X, y, S   : arrays from create_X_y_S
    flags     : sample_flag array from create_X_y_S
    model     : unfitted MRF instance (will be mutated in-place)
    update_freq: refit interval in OOS quarters (default 8 = 2 years)
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
            # Training mask: all obs before t, excluding post_oos rows
            train_mask = (np.arange(t) < t) & (flags[:t] != "post_oos")
            n_train    = train_mask.sum()
            print(f"  [OOS step {step + 1:3d}/{n_oos}] "
                  f"Fitting on t={t} obs ({n_train} effective train) ...")

            model.fit(X[:t], y[:t], S[:t])

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
    target:       str = "UNRATE",
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

        # Build expanding training set (strictly before t, all data available)
        X_ar, y_ar = [], []
        for s in range(n_lags_X, t):
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
