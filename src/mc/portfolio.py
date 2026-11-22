"""Several stocks at once: correlated simulation, buy-and-hold portfolios, crash correlation."""

import numpy as np
import pandas as pd

from mc.returns import TRADING_DAYS

DT = 1 / TRADING_DAYS


def correlated_gbm_paths(s0: np.ndarray, sigma: np.ndarray, corr: np.ndarray, days: int,
                         n_paths: int, seed: int | None = None, dt: float = DT) -> np.ndarray:
    """Zero-drift GBM for several assets whose daily moves have correlation `corr`.

    Cholesky: corr = L L^T; multiplying independent draws by L gives correlated ones.
    Returns shape (n_paths, days + 1, n_assets). Raises LinAlgError if `corr` is impossible.
    """
    s0, sigma = np.asarray(s0, float), np.asarray(sigma, float)
    chol = np.linalg.cholesky(np.asarray(corr, float))
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, days, len(s0))) @ chol.T
    steps = -0.5 * sigma**2 * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((n_paths, 1, len(s0))), np.cumsum(steps, axis=1)], 1)
    return s0 * np.exp(log_paths)


def bootstrap_days_paths(hist_log_returns: pd.DataFrame, s0: np.ndarray, days: int,
                         n_paths: int, seed: int | None = None, block: int = 1) -> np.ndarray:
    """Replay whole historical days (all stocks together), in runs of `block` days.

    Picking the same date for every stock keeps their real co-movement, crash days included,
    with no correlation matrix needed. Each column is centred to zero drift.
    """
    h = hist_log_returns.to_numpy(float)
    h = h - h.mean(axis=0) - 0.5 * h.var(axis=0, ddof=1)
    rng = np.random.default_rng(seed)
    n_blocks = -(-days // block)
    starts = rng.integers(0, len(h), (n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :days] % len(h)
    steps = h[idx]
    log_paths = np.concatenate([np.zeros((n_paths, 1, h.shape[1])), np.cumsum(steps, axis=1)], 1)
    return np.asarray(s0, float) * np.exp(log_paths)


def portfolio_value(paths: np.ndarray, weights: np.ndarray, capital: float) -> np.ndarray:
    """Buy-and-hold value: split `capital` by `weights` on day 0, never rebalance."""
    shares = capital * np.asarray(weights, float) / paths[:, 0, :]
    return np.einsum("ntk,nk->nt", paths, shares)


def calm_vs_crash_corr(returns: pd.DataFrame, market: str = "NIFTY",
                       q: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correlation matrix on normal days vs on the market's worst `q` share of days."""
    crash = returns[market] <= returns[market].quantile(q)
    return returns[~crash].corr(), returns[crash].corr()
