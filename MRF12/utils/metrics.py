"""
metrics.py — Evaluation metrics for MRF replication.

All functions are NaN-safe (pairs where either value is NaN are excluded).
"""

import numpy as np
from scipy import stats


def compute_rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error (NaN-pair safe)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def compute_mae(y_true, y_pred) -> float:
    """Mean Absolute Error (NaN-pair safe)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask   = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def compute_rmse_ratio(y_true, preds_model, preds_bench) -> float:
    """
    RMSE ratio: model RMSE / benchmark RMSE.
    Values < 1 mean the model beats the benchmark.
    Paper Table 4 reports this ratio vs AR(4).
    """
    return compute_rmse(y_true, preds_model) / compute_rmse(y_true, preds_bench)


def diebold_mariano_test(
    y_true:      np.ndarray,
    preds_model: np.ndarray,
    preds_bench: np.ndarray,
    h:           int   = 1,
    loss:        str   = "squared",
    alternative: str   = "less",
) -> dict:
    """
    Diebold-Mariano (2002) test for equal predictive accuracy.

    Tests H0: E[d_t] = 0  where d_t = loss(e1_t) - loss(e2_t),
    e1_t = y_true - preds_model,  e2_t = y_true - preds_bench.

    A negative mean d_t means the model has SMALLER loss than the benchmark.

    Parameters
    ----------
    y_true       : realised values
    preds_model  : model predictions (e.g. FA-ARRF)
    preds_bench  : benchmark predictions (e.g. AR(4))
    h            : forecast horizon (quarters).  Used to set the HAC lag
                   truncation to h-1 (Harvey et al. 1997 correction).
    loss         : "squared" (MSE-based, default) or "absolute" (MAE-based).
                   Paper uses squared loss → RMSPE comparisons (Section 3).
    alternative  : "less"    → H1: model loss < benchmark loss  (one-sided, recommended)
                   "two-sided"→ H1: losses differ
                   "greater" → H1: model loss > benchmark loss

    Returns
    -------
    dict with keys:
        dm_stat   : DM test statistic (Harvey et al. 1997 small-sample corrected)
        p_value   : p-value under chosen alternative
        mean_d    : mean loss differential (negative = model wins)
        n         : number of OOS observations used
        loss      : loss function used
        alternative: alternative hypothesis used

    Reference: Diebold & Mariano (2002), J. Business & Economic Statistics 20(1).
               Harvey, Leybourne & Newbold (1997) small-sample correction also applied.
    """
    y_true       = np.asarray(y_true,       dtype=float)
    preds_model  = np.asarray(preds_model,  dtype=float)
    preds_bench  = np.asarray(preds_bench,  dtype=float)

    # NaN-safe mask
    mask = ~(np.isnan(y_true) | np.isnan(preds_model) | np.isnan(preds_bench))
    y   = y_true[mask]
    pm  = preds_model[mask]
    pb  = preds_bench[mask]
    n   = mask.sum()

    if n < 2:
        return dict(dm_stat=np.nan, p_value=np.nan, mean_d=np.nan,
                    n=n, loss=loss, alternative=alternative)

    e1 = y - pm    # model errors
    e2 = y - pb    # benchmark errors

    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"loss must be 'squared' or 'absolute', got '{loss}'")

    mean_d = d.mean()

    # HAC long-run variance with Newey-West truncation at lag h-1
    # (matches the autocorrelation structure induced by h-step-ahead forecasts)
    max_lag = max(0, h - 1)
    gamma0  = np.var(d, ddof=0)
    lrv     = gamma0
    for lag in range(1, max_lag + 1):
        gamma_l = np.mean((d[lag:] - mean_d) * (d[:-lag] - mean_d))
        # Bartlett kernel weight
        w      = 1.0 - lag / (max_lag + 1)
        lrv   += 2.0 * w * gamma_l
    lrv = max(lrv, 1e-16)   # numerical guard

    # Standard DM statistic
    dm_raw = mean_d / np.sqrt(lrv / n)

    # Harvey, Leybourne & Newbold (1997) small-sample correction
    # Scales variance estimate: V_corr = (n + 1 - 2h + h(h-1)/n) / n * lrv
    hln_factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
    hln_factor = max(hln_factor, 1e-16)
    dm_stat    = mean_d / np.sqrt(hln_factor * lrv)

    # p-value: use t(n-1) distribution (Harvey et al. 1997)
    df = n - 1
    if alternative == "less":
        p_value = stats.t.cdf(dm_stat, df=df)
    elif alternative == "greater":
        p_value = 1.0 - stats.t.cdf(dm_stat, df=df)
    elif alternative == "two-sided":
        p_value = 2.0 * stats.t.cdf(-abs(dm_stat), df=df)
    else:
        raise ValueError(
            f"alternative must be 'less', 'greater', or 'two-sided', got '{alternative}'"
        )

    return dict(
        dm_stat     = float(dm_stat),
        p_value     = float(p_value),
        mean_d      = float(mean_d),
        n           = int(n),
        loss        = loss,
        alternative = alternative,
    )
