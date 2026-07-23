import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Tree.MRF import MRF
from experiments.forecast import create_X, expanding_window_forecast, extract_betas_over_time
from utils.metrics import compute_rmse
data = pd.read_excel("fredqd_final_model_ready.xlsx")
target = 'GDPC1'
X = create_X(data, target, lags=2)
y = data[target].values[2:]
S = data.drop(columns=[target]).values[2:]
model = MRF(n_trees=10, max_depth=4)
preds, actuals = expanding_window_forecast(X, y, S, model)
rmse = compute_rmse(actuals, preds)
print("RMSE:", rmse)
with open("outputs/results.txt", "w") as f:
    f.write(f"RMSE: {rmse}")
plt.plot(actuals, label='Actual')
plt.plot(preds, label='MRF Prediction')
plt.legend()
plt.title("Forecast vs Actual")
plt.savefig("outputs/predictions.png")
plt.show()
betas = extract_betas_over_time(X, y, S, model)
for i in range(betas.shape[1]):
    plt.plot(betas[:, i], label=f'Beta {i}')
plt.legend()
plt.title("Time-Varying Coefficients")
plt.savefig("outputs/betas.png")
plt.show()