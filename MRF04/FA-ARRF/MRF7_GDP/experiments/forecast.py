import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
PRIORITY_VARS = [
    "TB3SMFFM_L1", "TB3SMFFM_L2", "TB3SMFFM_MAF1", "TB3SMFFM_MAF2",
    "CPF3MTB3Mx_L1", "CPF3MTB3Mx_L2", "CPF3MTB3Mx_MAF1", "CPF3MTB3Mx_MAF2",
    "GS10TB3Mx_L1", "GS10TB3Mx_L2", "GS10TB3Mx_MAF1", "GS10TB3Mx_MAF2",
    "PERMIT_L1", "PERMIT_L2", "PERMIT_MAF1", "PERMIT_MAF2",
    "F_lead", "F_lead_L1", "F_lead_L2",
    "F1", "F2", "F1_L1", "F2_L1",
    "UMCSENTx_L1", "UMCSENTx_MAF1",
    "GDPC1_L1", "GDPC1_L2", "GDPC1_L3", "GDPC1_L4",
    "trend_t",
]

def build_leading_factor(data: pd.DataFrame) -> pd.Series:
    lead_vars = ["TB3SMFFM", "CPF3MTB3Mx", "GS10TB3Mx", "PERMIT", "UMCSENTx"]
    available = [v for v in lead_vars if v in data.columns]
    if len(available) < 2:
        return data["F2"].copy().rename("F_lead")
    df_sub = data[available].copy()
    mu     = df_sub.mean()
    sig    = df_sub.std().replace(0, 1)
    df_std = (df_sub - mu) / sig
    df_std = df_std.fillna(0.0)
    pca    = PCA(n_components=1)
    fl     = pca.fit_transform(df_std)[:, 0]
    if "TB3SMFFM" in available:
        tb_corr = np.corrcoef(fl, df_sub["TB3SMFFM"].fillna(0))[0, 1]
        if tb_corr > 0:
            fl = -fl      
    return pd.Series(fl, index=data.index, name="F_lead")


def build_St(data: pd.DataFrame, target: str = "GDPC1",
             n_y_lags: int = 8) -> tuple:
    cols = []
    if "trend_t" in data.columns:
        cols.append("trend_t")
    raw_l1l2 = sorted([
        c for c in data.columns
        if (c.endswith("_L1") or c.endswith("_L2"))
        and not any(c.startswith(f"F{i}_") for i in range(1, 6))
        and target not in c
        and "F_lead" not in c
    ])
    cols += raw_l1l2
    for fi in ["F1", "F2", "F3", "F4", "F5"]:
        cols += sorted([c for c in data.columns if c.startswith(fi + "_L")])
    cols += sorted([c for c in data.columns
                    if (c.endswith("_MAF1") or c.endswith("_MAF2"))
                    and "F_lead" not in c])
    for fi in ["F1", "F2", "F3", "F4", "F5"]:
        if fi in data.columns and fi not in cols:
            cols.append(fi)
    for fl_col in ["F_lead", "F_lead_L1", "F_lead_L2"]:
        if fl_col in data.columns and fl_col not in cols:
            cols.append(fl_col)
    extra_lag_dfs = []
    for lag in range(1, n_y_lags + 1):
        col_name = f"{target}_L{lag}"
        if col_name in data.columns:
            if col_name not in cols:
                cols.append(col_name)
        else:
            extra_lag_dfs.append(
                data[target].shift(lag).rename(col_name).to_frame())
    cols   = list(dict.fromkeys([c for c in cols if c in data.columns]))
    parts  = [data[cols]]
    if extra_lag_dfs:
        parts += extra_lag_dfs

    S_df        = pd.concat(parts, axis=1).dropna()
    S_col_names = list(S_df.columns)

    priority_col_idxs = [
        S_col_names.index(v) for v in PRIORITY_VARS if v in S_col_names
    ]
    return S_df, S_col_names, priority_col_idxs


def create_X_y_S(data_aligned: pd.DataFrame,
                 S_df:         pd.DataFrame,
                 target:       str = "GDPC1",
                 h:            int = 1) -> tuple:
    y_raw  = data_aligned[target].values
    f1     = data_aligned["F1"].values
    f_lead = data_aligned["F_lead"].values
    flags_ = data_aligned["sample_flag"].values
    dates_ = pd.to_datetime(data_aligned["observation_date"].values)
    S_arr  = S_df.values
    n_lags = 4   
    T      = len(data_aligned)
    X_rows, y_rows, S_rows, flag_rows, date_rows = [], [], [], [], []
    for t in range(n_lags, T - h):
        row = [
            1.0,
            y_raw[t-1], y_raw[t-2], y_raw[t-3], y_raw[t-4],
            f1[t-1],
            f_lead[t-1],
        ]
        X_rows.append(row)
        y_rows.append(y_raw[t + h])
        S_rows.append(S_arr[t])
        flag_rows.append(flags_[t])
        date_rows.append(dates_[t])
    return (np.array(X_rows), np.array(y_rows),
            np.array(S_rows),  np.array(flag_rows),
            np.array(date_rows))

def expanding_window_oos(X, y, S, flags, model,
                         update_freq: int = 8,
                         S_col_names: list = None,
                         print_diag:  bool = False) -> tuple:
    oos_idx   = np.where(flags == "oos")[0]
    oos_start = oos_idx[0]
    oos_end   = oos_idx[-1]
    n_oos     = len(oos_idx)
    preds, actuals, betas = [], [], []
    for step, t in enumerate(range(oos_start, oos_end + 1)):
        if step == 0 or step % update_freq == 0:
            n_eff = (flags[:t] != "post_oos").sum()
            print(f"  [Step {step+1:2d}/{n_oos}] fit on {t} obs ({n_eff} effective) ...")
            model.fit(X[:t], y[:t], S[:t])
            if print_diag and S_col_names:
                model.print_diagnostics(S_col_names)
        y_pred = model.predict(X[t:t+1], S[t:t+1])[0]
        beta_t = model.get_betas(S[t:t+1])[0]
        preds.append(y_pred);  actuals.append(y[t]);  betas.append(beta_t)
    return np.array(preds), np.array(actuals), np.array(betas)


def ar4_benchmark(data_aligned: pd.DataFrame, flags_X: np.ndarray,
                  target: str = "GDPC1", h: int = 1,
                  n_lags_ar: int = 4) -> tuple:
    from numpy.linalg import lstsq
    y_raw    = data_aligned[target].values
    flags_   = data_aligned["sample_flag"].values
    n_lags   = 4  # matches create_X_y_S offset
    T        = len(data_aligned)
    flag_aligned = np.array([flags_[t] for t in range(n_lags, T - h)])
    oos_idx      = np.where(flag_aligned == "oos")[0]
    preds, actuals = [], []
    for i in oos_idx:
        t = i + n_lags
        X_ar, y_ar = [], []
        for s in range(n_lags_ar, t):
            X_ar.append([1.0] + [y_raw[s-l] for l in range(1, n_lags_ar+1)])
            y_ar.append(y_raw[s + h])
        if len(y_ar) < n_lags_ar + 2:
            preds.append(np.nan); actuals.append(y_raw[t + h]); continue
        X_ar, y_ar = np.array(X_ar), np.array(y_ar)
        b, _, _, _ = lstsq(X_ar, y_ar, rcond=None)
        x_pred = [1.0] + [y_raw[t-l] for l in range(1, n_lags_ar+1)]
        preds.append(float(np.dot(x_pred, b)))
        actuals.append(y_raw[t + h])
    return np.array(preds), np.array(actuals)


def compute_rmse(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2)))
