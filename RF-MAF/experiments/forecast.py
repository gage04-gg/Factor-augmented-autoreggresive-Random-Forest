import numpy as np
import pandas as pd

PRIORITY_VARS = [
    "GS10TB3Mx_L1","GS10TB3Mx_L2","GS10TB3Mx_MAF1","GS10TB3Mx_MAF2",
    "CPF3MTB3Mx_L1","CPF3MTB3Mx_L2","CPF3MTB3Mx_MAF1",
    "TB3SMFFM_L1","TB3SMFFM_L2","T5YFFM_MAF1",
    "PERMIT_L1","PERMIT_L2","PERMIT_MAF1","PERMIT_MAF2",
    "F1","F2","F1_L1","F2_L1","F1_L2","F2_L2",
    "UMCSENTx_L1","UMCSENTx_MAF1",
    "AWHMAN_L1","AWHMAN_MAF1",
    "GDPC1_L1","GDPC1_L2","GDPC1_L3","GDPC1_L4",
    "trend_t"
]

def build_St(data, target="GDPC1", n_y_lags=8):
    cols = ["trend_t"] if "trend_t" in data.columns else []

    raw_l1l2 = sorted([
        c for c in data.columns
        if (c.endswith("_L1") or c.endswith("_L2"))
        and target not in c
    ])
    cols += raw_l1l2

    for fi in ["F1","F2","F3","F4","F5"]:
        cols += sorted([c for c in data.columns if c.startswith(fi+"_L")])

    cols += sorted([c for c in data.columns if c.endswith("_MAF1") or c.endswith("_MAF2")])

    for fi in ["F1","F2","F3","F4","F5"]:
        if fi in data.columns and fi not in cols:
            cols.append(fi)

    S_df = data[cols].dropna()
    S_col_names = list(S_df.columns)

    priority_col_idxs = [
        S_col_names.index(v)
        for v in PRIORITY_VARS if v in S_col_names
    ]

    return S_df, S_col_names, priority_col_idxs
def expanding_window_oos_ar_rf(y, S, flags, model, update_freq=8):

    oos_idx = np.where(flags == "oos")[0]
    preds, actuals = [], []

    for step, t in enumerate(oos_idx):

        if step == 0 or step % update_freq == 0:

            # 🔹 Step 1: Fit AR(2)
            y_train = y[:t]
            phi1, phi2 = 0.0, 0.0

            if len(y_train) > 2:
                Y = y_train[2:]
                X = np.column_stack([y_train[1:-1], y_train[:-2]])
                beta = np.linalg.lstsq(X, Y, rcond=None)[0]
                phi1, phi2 = beta

            # 🔹 Step 2: residuals
            u = y_train[2:] - (phi1*y_train[1:-1] + phi2*y_train[:-2])

            model.fit(u, S[2:t])

        # 🔹 Step 3: predict residual
        u_hat = model.predict(S[t:t+1])[0]

        # 🔹 Step 4: reconstruct y
        y_hat = phi1*y[t-1] + phi2*y[t-2] + u_hat

        preds.append(y_hat)
        actuals.append(y[t])

    return np.array(preds), np.array(actuals)
def create_y_S(data_aligned, S_df, target="GDPC1", h=1):
    import numpy as np
    import pandas as pd

    y_raw  = data_aligned[target].values
    flags_ = data_aligned["sample_flag"].values
    dates_ = pd.to_datetime(data_aligned["observation_date"].values)
    S_arr  = S_df.values

    n_lags = 2
    T      = len(data_aligned)

    y_rows, S_rows, flag_rows, date_rows = [], [], [], []

    for t in range(n_lags, T - h):
        y_rows.append(y_raw[t + h])
        S_rows.append(S_arr[t])
        flag_rows.append(flags_[t])
        date_rows.append(dates_[t])

    return (
        np.array(y_rows),
        np.array(S_rows),
        np.array(flag_rows),
        np.array(date_rows),
    )
def ar4_benchmark(data_aligned, flags, target="GDPC1", h=1, n_lags_X=4):
    import numpy as np

    y_raw  = data_aligned[target].values
    flags_ = data_aligned["sample_flag"].values
    T      = len(data_aligned)

    n_lags = 2  # alignment offset

    # Align flags with y,S construction
    flag_aligned = np.array([flags_[t] for t in range(n_lags, T - h)])
    oos_idx      = np.where(flag_aligned == "oos")[0]

    preds, actuals = [], []

    for i in oos_idx:
        t = i + n_lags

        X_ar, y_ar = [], []

        # Build AR(4) training data
        for s in range(n_lags_X, t):
            X_ar.append([1.0] + [y_raw[s-l] for l in range(1, n_lags_X+1)])
            y_ar.append(y_raw[s + h])

        X_ar = np.array(X_ar)
        y_ar = np.array(y_ar)

        if len(y_ar) < n_lags_X + 1:
            preds.append(np.nan)
            actuals.append(y_raw[t + h])
            continue

        # OLS estimation
        beta_ar, _, _, _ = np.linalg.lstsq(X_ar, y_ar, rcond=None)

        # Prediction
        x_pred = [1.0] + [y_raw[t-l] for l in range(1, n_lags_X+1)]
        preds.append(float(np.dot(x_pred, beta_ar)))
        actuals.append(y_raw[t + h])

    return np.array(preds), np.array(actuals)
def compute_rmse(y_true, y_pred):
    import numpy as np

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = ~(np.isnan(y_true) | np.isnan(y_pred))

    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2)))

