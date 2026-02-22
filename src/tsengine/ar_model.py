from __future__ import annotations
import numpy as np

class ARModel:
    def __init__(self, p: int = 1):
        if p <= 0:
            raise ValueError("p must be positive")
        self.p = p
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit_ols(self, series: np.ndarray) -> "ARModel":
        x = np.asarray(series, dtype=float)
        p = self.p
        if x.size <= p + 1:
            raise ValueError("series is too short for chosen p")

        Y = x[p:]
        X = np.column_stack([x[p - i - 1: -(i + 1)] for i in range(p)])
        X = np.c_[np.ones(len(Y)), X]  # intercept

        # OLS with small regularization for stability
        reg = 1e-8 * np.eye(X.shape[1])
        beta = np.linalg.solve(X.T @ X + reg, X.T @ Y)

        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]
        return self

    def forecast(self, history: np.ndarray, steps: int = 10) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted yet.")
        hist = list(np.asarray(history, dtype=float).tolist())
        p = self.p
        phi = self.coef_

        preds = []
        for _ in range(steps):
            if len(hist) < p:
                raise ValueError("Not enough history points to forecast.")
            x_next = self.intercept_ + sum(phi[i] * hist[-i-1] for i in range(p))
            preds.append(float(x_next))
            hist.append(float(x_next))
        return np.array(preds, dtype=float)

