
from __future__ import annotations
import numpy as np

def generate_ar_series(phi: list[float], n: int = 500, sigma: float = 1.0, seed: int = 42) -> np.ndarray:
    """
    Generate AR(p): x_t = sum(phi_i * x_{t-i}) + e_t
    """
    rng = np.random.default_rng(seed)
    p = len(phi)
    x = np.zeros(n, dtype=float)
    eps = rng.normal(0, sigma, size=n)
    for t in range(p, n):
        x[t] = sum(phi[i] * x[t - i - 1] for i in range(p)) + eps[t]
    return x

