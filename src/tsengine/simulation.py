from __future__ import annotations
import numpy as np

def random_walk(n: int = 252, sigma: float = 1.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, size=n)
    return np.cumsum(eps)

def geometric_brownian_motion(
    n: int = 252,
    s0: float = 100.0,
    mu: float = 0.05,
    sigma: float = 0.2,
    dt: float = 1.0/252,
    seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, size=n)
    increments = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_path = np.log(s0) + np.cumsum(increments)
    return np.exp(log_path)

