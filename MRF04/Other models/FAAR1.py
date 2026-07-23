

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from scipy import stats


DATA_PATH = "fredqd_final_model_ready.xlsx"
output_dir = "FAAR_FINAL_RESULTS"
os.makedirs(output_dir, exist_ok=True)


TARGET_MAP = {
    "GDP": "GDPC1",
    "UR": "UNRATE",
    "CPI": "CPIAUCSL",
    "IR": "GS10"
}

HORIZONS = [1, 2, 4]
p = 4


def dm_test(e_model, e_ar, h=1):

    d = e_model**2 - e_ar**2
    d = d[~np.isnan(d)]

    T = len(d)
    if T < 10:
        return np.nan, np.nan, ""

    d_bar = np.mean(d)
    gamma0 = np.var(d, ddof=1)

    if h > 1:
        for k in range(1, h):
            gamma_k = np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
            gamma0 += 2 * (1 - k/h) * gamma_k

    se = np.sqrt(gamma0 / T)
    if se == 0:
        return np.nan, np.nan, ""

    DM = d_bar / se
    pval = stats.norm.cdf(DM)

    if pval < 0.01:
        sig = "***"
    elif pval < 0.05:
        sig = "**"
    elif pval < 0.10:
        sig = "*"
    else:
        sig = ""

    return round(DM, 4), round(pval, 4), sig



df = pd.read_excel(DATA_PATH)

df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

# Create SPREAD
if 'GS10' in df.columns and 'TB3MS' in df.columns:
    df['SPREAD'] = df['GS10'] - df['TB3MS']

print("Dataset loaded")


exclude_cols = ['sample_flag', 'trend_t', 'quarter']

factor_data = df.drop(columns=[c for c in exclude_cols if c in df.columns],
                     errors='ignore')

factor_data = factor_data.drop(columns=list(TARGET_MAP.values()) + ["SPREAD"], errors='ignore')
factor_data = factor_data.select_dtypes(include=[np.number])

factor_data = factor_data.diff().dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(factor_data)

pca = PCA(n_components=2)
F = pca.fit_transform(X_scaled)

F_df = pd.DataFrame(F, index=factor_data.index, columns=["F1","F2"])


results = []

for target_name, col in TARGET_MAP.items():

    if col not in df.columns:
        print(f"Skipping {target_name}")
        continue

    print(f"\nRunning FA-AR for {target_name}")

    y = df[col].diff().dropna()

    # Align
    common_index = y.index.intersection(F_df.index)
    y = y.loc[common_index]
    F_used = F_df.loc[common_index]

    # OOS split
    train_mask = y.index < "2003-01-01"
    test_mask  = (y.index >= "2003-01-01") & (y.index <= "2014-12-31")

    y_train, y_test = y[train_mask], y[test_mask]
    F_train, F_test = F_used[train_mask], F_used[test_mask]

    print(f"Train: {len(y_train)}, Test: {len(y_test)}")

    for h in HORIZONS:

        preds_FAAR, preds_AR, actuals, dates = [], [], [], []

        for t in range(len(y_test) - h):

            y_exp = pd.concat([y_train, y_test.iloc[:t]])
            F_exp = pd.concat([F_train, F_test.iloc[:t]])

            df_reg = pd.DataFrame({'y': y_exp})

            # AR lags
            for lag in range(1, 5):
                df_reg[f'y_lag{lag}'] = df_reg['y'].shift(lag)

            # factor lags
            for lag in range(1, 3):
                df_reg[f'F1_lag{lag}'] = F_exp['F1'].shift(lag)
                df_reg[f'F2_lag{lag}'] = F_exp['F2'].shift(lag)

            df_reg = df_reg.dropna()

            if len(df_reg) < 30:
                continue

            Y = df_reg['y']

            # FAAR
            X = sm.add_constant(df_reg.drop(columns=['y']))
            model_FAAR = sm.OLS(Y, X).fit()

            # AR
            X_ar = sm.add_constant(df_reg[[f'y_lag{i}' for i in range(1, 5)]])
            model_AR = sm.OLS(Y, X_ar).fit()

            last = df_reg.iloc[-1:].copy()

           
            temp = last.copy()
            for _ in range(h):
                X_last = sm.add_constant(temp.drop(columns=['y']), has_constant='add')
                fcast = model_FAAR.predict(X_last).iloc[0]

                new = temp.copy()
                new['y'] = fcast
                for lag in range(4,1,-1):
                    new[f'y_lag{lag}'] = new[f'y_lag{lag-1}']
                new['y_lag1'] = fcast
                temp = new

            preds_FAAR.append(fcast)

           
            temp = last.copy()
            for _ in range(h):
                X_last = sm.add_constant(temp[[f'y_lag{i}' for i in range(1,5)]],
                                         has_constant='add')
                fcast_ar = model_AR.predict(X_last).iloc[0]

                new = temp.copy()
                new['y'] = fcast_ar
                for lag in range(4,1,-1):
                    new[f'y_lag{lag}'] = new[f'y_lag{lag-1}']
                new['y_lag1'] = fcast_ar
                temp = new

            preds_AR.append(fcast_ar)

            actuals.append(y_test.iloc[t + h - 1])
            dates.append(y_test.index[t + h - 1])

        if len(actuals) == 0:
            print(f"No results for {target_name}, h={h}")
            continue

        preds_FAAR = np.array(preds_FAAR)
        preds_AR = np.array(preds_AR)
        actuals = np.array(actuals)

        rmse = np.sqrt(mean_squared_error(actuals, preds_FAAR))

        # DM TEST
        e_model = actuals - preds_FAAR
        e_ar = actuals - preds_AR
        dm_stat, pval, sig = dm_test(e_model, e_ar, h=h)

        print(f"{target_name} | h={h} → RMSE: {rmse:.4f} | DM: {dm_stat} {sig}")

        results.append([target_name, h, rmse, dm_stat, pval, sig])

      
        pd.DataFrame({
            "Date": dates,
            "Actual": actuals,
            "FAAR": preds_FAAR,
            "AR": preds_AR
        }).to_csv(f"{output_dir}/{target_name}_h{h}.csv", index=False)

        # Plot
        plt.figure(figsize=(10,5))
        plt.plot(dates, actuals, label="Actual")
        plt.plot(dates, preds_FAAR, linestyle="--", label="FA-AR")

        plt.axvspan(pd.Timestamp("2007-12-01"),
                    pd.Timestamp("2009-06-01"),
                    color="gray", alpha=0.2)

        plt.title(f"{target_name} | h={h} | RMSE={rmse:.4f}")
        plt.legend()
        plt.grid()

        plt.savefig(f"{output_dir}/{target_name}_h{h}.png")
        plt.close()


summary = pd.DataFrame(
    results,
    columns=["Target","Horizon","RMSE","DM_stat","p_value","Significance"]
)

summary.to_csv(f"{output_dir}/FAAR_summary.csv", index=False)

print("\n DONE  ")