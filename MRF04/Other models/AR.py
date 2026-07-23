
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error


output_dir = "AR4_results"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_excel("fredqd_final_model_ready.xlsx")

df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)

print("Dataset loaded")


targets = []

for col in ['GDPC1', 'UNRATE', 'CPIAUCSL', 'GS10']:
    if col in df.columns:
        targets.append(col)

# SPREAD
if 'GS10' in df.columns and 'TB3MS' in df.columns:
    df['SPREAD'] = df['GS10'] - df['TB3MS']
    targets.append('SPREAD')

print("Variables used:", targets)


p = 4
horizons = [1, 2, 4, 6]

rmse_results = []


for target in targets:

    print(f"\nRunning AR(4) for {target}")

   
    y = df[target].dropna()

  
    y = y.diff().dropna()

  
    train = y[y.index < "2003-01-01"]
    test  = y[(y.index >= "2003-01-01") & (y.index <= "2014-12-31")]

    if len(test) < 20:
        print(f"{target} → Not enough data, skipping")
        continue

 
    for h in horizons:

        preds = []
        actuals = []
        dates = []


        for t in range(len(test) - h):

            y_expanded = pd.concat([train, test.iloc[:t]])

            if len(y_expanded) <= p:
                continue

            try:
                model = AutoReg(y_expanded, lags=p, old_names=False)
                model_fit = model.fit()

                forecast = model_fit.predict(
                    start=len(y_expanded),
                    end=len(y_expanded) + h - 1
                )

                preds.append(forecast.iloc[-1])
                actuals.append(test.iloc[t + h - 1])
                dates.append(test.index[t + h - 1])

            except:
                continue


        if len(actuals) == 0:
            print(f"{target} | h={h} → No forecasts")
            continue

        rmse = np.sqrt(mean_squared_error(actuals, preds))
        print(f"{target} | h={h} RMSE: {rmse:.4f}")

        rmse_results.append([target, h, rmse])


        results_df = pd.DataFrame({
            'Date': dates,
            'Actual': actuals,
            'Forecast': preds
        })

        results_df.to_csv(
            f"{output_dir}/{target}_AR4_h{h}.csv",
            index=False
        )


        plt.figure(figsize=(12,6))

        plt.plot(dates, actuals, label="Actual", linewidth=2)
        plt.plot(dates, preds, label="Forecast (AR(4))", linestyle='--')

        # RECESSION SHADING (2007–09)
        plt.axvspan(pd.Timestamp("2007-12-01"),
                    pd.Timestamp("2009-06-01"),
                    color='gray', alpha=0.3)

        
        plt.xlim(pd.Timestamp("2003-01-01"),
                 pd.Timestamp("2014-12-31"))

        plt.title(f"{target} AR(4) Forecast (2003–2014, h={h})")
        plt.legend()
        plt.grid()

        # SAVE PLOT
        plt.savefig(f"{output_dir}/{target}_AR4_h{h}.png")
        plt.close()

rmse_df = pd.DataFrame(rmse_results, columns=['Target', 'Horizon', 'RMSE'])
rmse_df.to_csv(f"{output_dir}/AR4_RMSE_summary.csv", index=False)

print("\nAll results saved successfully!")