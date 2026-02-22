from __future__ import annotations
import numpy as np

def adf_test(series: np.ndarray):
    """
    If statsmodels is installed, run Augmented Dickey-Fuller test.
    Otherwise return a simple heuristic diagnostic.
    """
    series = np.asarray(series, dtype=float)
    try:
        from statsmodels.tsa.stattools import adfuller
        stat, pvalue, *_ = adfuller(series, autolag="AIC")
        return {"method": "adf", "stat": float(stat), "pvalue": float(pvalue)}
    except Exception:
        # heuristic: compare variance of differences vs variance of series
        diff = np.diff(series)
        ratio = np.var(diff) / (np.var(series) + 1e-12)
        return {"method": "heuristic", "diff_var_ratio": float(ratio)}

