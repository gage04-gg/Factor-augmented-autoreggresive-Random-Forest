
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from scipy import stats


DATA_PATH = "fredqd_final_model_ready.xlsx"
output_dir = "SETAR_RESULTS"
os.makedirs(output_dir, exist_ok=True)

TARGET_MAP = {
    "GDP": "GDPC1",
    "UR": "UNRATE",
    "CPI": "CPIAUCSL",
    "IR": "GS10"
}

HORIZONS = [1, 2, 4]
p = 2  # SETAR uses AR(2)

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

# SPREAD
if 'GS10' in df.columns and 'TB3MS' in df.columns:
    df['SPREAD'] = df['GS10'] - df['TB3MS']
    TARGET_MAP["SPREAD"] = "SPREAD"

print("Dataset loaded")


results = []


for target_name, col in TARGET_MAP.items():

    if col not in df.columns:
        continue

    print(f"\nRunning SETAR for {target_name}")

    y = df[col].diff().dropna()

  
    train_mask = y.index < "2003-01-01"
    test_mask  = (y.index >= "2003-01-01") & (y.index <= "2014-12-31")

    y_train, y_test = y[train_mask], y[test_mask]

    for h in HORIZONS:

        preds_SETAR, preds_AR, actuals, dates = [], [], [], []

        for t in range(len(y_test) - h):

            y_exp = pd.concat([y_train, y_test.iloc[:t]])

            df_reg = pd.DataFrame({'y': y_exp})

           
            for lag in range(1, 3):
                df_reg[f'y_lag{lag}'] = df_reg['y'].shift(lag)

            df_reg = df_reg.dropna()

            if len(df_reg) < 30:
                continue

            
            threshold_var = df_reg['y_lag1']
            threshold = threshold_var.median()

            
            low_regime = df_reg[threshold_var <= threshold]
            high_regime = df_reg[threshold_var > threshold]

           
            if len(low_regime) < 10 or len(high_regime) < 10:
                continue

        
            Y_low = low_regime['y']
            X_low = sm.add_constant(low_regime[['y_lag1','y_lag2']])
            model_low = sm.OLS(Y_low, X_low).fit()

            
            Y_high = high_regime['y']
            X_high = sm.add_constant(high_regime[['y_lag1','y_lag2']])
            model_high = sm.OLS(Y_high, X_high).fit()

            
            Y = df_reg['y']
            X_ar = sm.add_constant(df_reg[['y_lag1','y_lag2']])
            model_AR = sm.OLS(Y, X_ar).fit()

            last = df_reg.iloc[-1:].copy()

            #
            temp = last.copy()

            for _ in range(h):

                if temp['y_lag1'].values[0] <= threshold:
                    X_last = sm.add_constant(temp[['y_lag1','y_lag2']], has_constant='add')
                    fcast = model_low.predict(X_last).iloc[0]
                else:
                    X_last = sm.add_constant(temp[['y_lag1','y_lag2']], has_constant='add')
                    fcast = model_high.predict(X_last).iloc[0]

                new = temp.copy()
                new['y'] = fcast
                new['y_lag2'] = new['y_lag1']
                new['y_lag1'] = fcast
                temp = new

            preds_SETAR.append(fcast)

          
            temp = last.copy()
            for _ in range(h):
                X_last = sm.add_constant(temp[['y_lag1','y_lag2']], has_constant='add')
                fcast_ar = model_AR.predict(X_last).iloc[0]

                new = temp.copy()
                new['y'] = fcast_ar
                new['y_lag2'] = new['y_lag1']
                new['y_lag1'] = fcast_ar
                temp = new

            preds_AR.append(fcast_ar)

            actuals.append(y_test.iloc[t + h - 1])
            dates.append(y_test.index[t + h - 1])

        if len(actuals) == 0:
            continue

        preds_SETAR = np.array(preds_SETAR)
        preds_AR = np.array(preds_AR)
        actuals = np.array(actuals)

        rmse = np.sqrt(mean_squared_error(actuals, preds_SETAR))

        
        e_model = actuals - preds_SETAR
        e_ar = actuals - preds_AR
        dm_stat, pval, sig = dm_test(e_model, e_ar, h=h)

        print(f"{target_name} | h={h} → RMSE: {rmse:.4f} | DM: {dm_stat} {sig}")

        results.append([target_name, h, rmse, dm_stat, pval, sig])

       
        pd.DataFrame({
            "Date": dates,
            "Actual": actuals,
            "SETAR": preds_SETAR,
            "AR": preds_AR
        }).to_csv(f"{output_dir}/{target_name}_h{h}.csv", index=False)

        plt.figure(figsize=(10,5))
        plt.plot(dates, actuals, label="Actual")
        plt.plot(dates, preds_SETAR, linestyle="--", label="SETAR")

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

summary.to_csv(f"{output_dir}/SETAR_summary.csv", index=False)

print("\n SETAR MODEL COMPLETE")