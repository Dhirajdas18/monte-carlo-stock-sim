"""Checking a model against reality before trusting it.

A Model turns past daily log returns into simulated future prices:
    model(hist_log_returns, s0, days, n_paths, rng) -> array (n_paths, days + 1)
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy import stats

Model = Callable[[np.ndarray, float, int, int, np.random.Generator], np.ndarray]


def _to_prices(s0: float, log_steps: np.ndarray) -> np.ndarray:
    n = log_steps.shape[0]
    return s0 * np.exp(np.hstack([np.zeros((n, 1)), np.cumsum(log_steps, axis=1)]))


def gbm_model(hist: np.ndarray, s0: float, days: int, n_paths: int,
              rng: np.random.Generator) -> np.ndarray:
    """Normal daily moves with the history's volatility and zero drift."""
    sd = float(np.std(hist, ddof=1))
    steps = -0.5 * sd**2 + sd * rng.standard_normal((n_paths, days))
    return _to_prices(s0, steps)


def coverage(prices: pd.Series, model: Model, horizon: int = 21, window: int = 756,
             step: int = 21, level: float = 0.90, n_paths: int = 5000,
             seed: int = 0) -> pd.DataFrame:
    """Walk forward through history: on each forecast date, fit the model on the past `window`
    days only, predict a `level` band for the price `horizon` days later, and record whether
    the real price landed inside it."""
    rng = np.random.default_rng(seed)
    logp = np.log(prices.to_numpy(dtype=float))
    tail = (1 - level) / 2
    rows, dates = [], []
    for i in range(window, len(prices) - horizon, step):
        hist = np.diff(logp[i - window : i + 1])
        final = model(hist, float(prices.iloc[i]), horizon, n_paths, rng)[:, -1]
        lo, hi = np.quantile(final, [tail, 1 - tail])
        actual = float(prices.iloc[i + horizon])
        rows.append((lo, hi, actual, lo <= actual <= hi))
        dates.append(prices.index[i])
    return pd.DataFrame(rows, columns=["lo", "hi", "actual", "hit"], index=pd.Index(dates))


def qq_points(returns: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Quantiles of standardised returns vs a standard normal. On the diagonal = normal."""
    z = np.sort(((returns - returns.mean()) / returns.std(ddof=1)).to_numpy())
    theo = stats.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
    return theo, z
