import pandas as pd
import numpy as np
data = pd.read_csv("Clean_MRF.csv")
data['observation_date'] = pd.to_datetime(data['observation_date'])
data = data.sort_values('observation_date')
data = data.groupby('observation_date').mean()
missing = data.isna().sum()
print("Missing values per column:\n", missing)
threshold = 0.8 * len(data)
data = data.dropna(axis=1, thresh=threshold)
data = data.fillna(method='ffill')
data = data.fillna(method='bfill')
constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
data = data.drop(columns=constant_cols)
for col in data.columns:
    lower = data[col].quantile(0.01)
    upper = data[col].quantile(0.99)
    data[col] = np.clip(data[col], lower, upper)
data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()
print("\nFinal shape:", data.shape)
print("Remaining NA:", data.isna().sum().sum())
data.index = pd.PeriodIndex(data.index, freq='Q')
data.to_excel("cleaned_MRF3.xlsx")