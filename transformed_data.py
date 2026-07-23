import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
data = pd.read_excel("Clean_MRF.xlsx")
data['observation_date'] = pd.PeriodIndex(data['observation_date'], freq='Q')
data = data.set_index('observation_date')
log_diff_vars = [
    'Real_GDP',
    'CPI',
    'Core_CPI',
    'PPI',
    'Industrial_production',
    'Personal_Consumption',
    'Disposable_Income',
    'Housing_price_index',
    'Building_permits',
    'HOUST',
    'M1REAL',
    'M2REAL',
    'S&P 500 Index' 
]

rate_vars = [
    '3M_treasury_bill',
    '10_yr_yield',
    'FedFunds',
    'Unemployment_Rate',
    'Capacity_Utilization',
    'Credit_Spread_AAA',
    'Credit_Spread_BAA'
]
for col in log_diff_vars:
    if col in data.columns:
        data[col + "_trans"] = np.log(data[col]).diff()

for col in rate_vars:
    if col in data.columns:
        data[col + "_trans"] = data[col]

transformed_cols = [col for col in data.columns if col.endswith("_trans")]
data = data[transformed_cols]
data = data.dropna()
scaler = StandardScaler()
data_scaled = pd.DataFrame(
    scaler.fit_transform(data),
    columns=data.columns,
    index=data.index
)
data_scaled = data_scaled.reset_index()
data_scaled.to_excel("transformed_MRF.xlsx", index=False)
print(data_scaled.head())
print("Shape:", data_scaled.shape)