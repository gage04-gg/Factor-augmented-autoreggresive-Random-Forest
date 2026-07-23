import numpy as np
def compute_rmse(y_true, y_pred):
    mask = np.isfinite(np.asarray(y_true)) & np.isfinite(np.asarray(y_pred))
    return float(np.sqrt(np.mean((np.asarray(y_true)[mask] - np.asarray(y_pred)[mask])**2)))
