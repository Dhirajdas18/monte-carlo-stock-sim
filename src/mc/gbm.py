"""Geometric Brownian motion (GBM): the baseline price-path model.

Each day the log price moves by a normal draw: (mu - sigma^2/2) * dt + sigma * sqrt(dt) * Z.
"""

import numpy as np
import pandas as pd

from mc.returns import TRADING_DAYS

DT = 1 / TRADING_DAYS


def gbm_paths(s0: float, mu: float, sigma: float, days: int, n_paths: int,
              seed: int | None = None, antithetic: bool = False, dt: float = DT) -> np.ndarray:
    """Simulated prices, shape (n_paths, days + 1); column 0 is s0.

    antithetic=True pairs every random draw Z with -Z, cancelling some sampling noise.
    """
    rng = np.random.default_rng(seed)
    if antithetic:
        if n_paths % 2:
            raise ValueError("antithetic sampling needs an even number of paths")
        half = rng.standard_normal((n_paths // 2, days))
        z = np.vstack([half, -half])
    else:
        z = rng.standard_normal((n_paths, days))
    steps = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(steps, axis=1)
    return s0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))


def gbm_closed_form(s0: float, mu: float, sigma: float, t_years: float) -> dict:
    """Exact answers GBM gives without simulating: mean and median final price."""
    return {
        "mean": s0 * np.exp(mu * t_years),
        "median": s0 * np.exp((mu - 0.5 * sigma**2) * t_years),
    }


def drift_stderr(daily_log_returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Standard error of the annual mean return: annual vol / sqrt(years of data)."""
    years = len(daily_log_returns) / periods
    return float(daily_log_returns.std(ddof=1) * np.sqrt(periods) / np.sqrt(years))


def mc_error_curve(s0: float, mu: float, sigma: float, days: int, ns: list[int], seed: int = 0,
                   antithetic: bool = False, repeats: int = 20) -> pd.DataFrame:
    """Average |simulated mean - exact mean| for each path count, over `repeats` seeds."""
    exact = gbm_closed_form(s0, mu, sigma, days * DT)["mean"]
    rows = []
    for n in ns:
        errs = [abs(gbm_paths(s0, mu, sigma, days, n, seed=seed + k,
                              antithetic=antithetic)[:, -1].mean() - exact)
                for k in range(repeats)]
        rows.append((n, float(np.mean(errs))))
    return pd.DataFrame(rows, columns=["n", "abs_error"])
