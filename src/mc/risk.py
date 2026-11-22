"""Turning simulated paths into decisions: bands, VaR/CVaR, touch vs end probabilities."""

import numpy as np
import pandas as pd


def percentile_bands(paths: np.ndarray, q=(5, 50, 95)) -> pd.DataFrame:
    """Per-day percentiles of the simulated price; rows are days, columns are q."""
    return pd.DataFrame(np.percentile(paths, q, axis=0).T, columns=list(q))


def var_cvar(pnl: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Value at Risk and Conditional VaR, as positive loss amounts.

    VaR: the loss exceeded only (1 - level) of the time.
    CVaR: the average loss in those worst (1 - level) cases.
    """
    loss = -np.asarray(pnl, dtype=float)
    var = float(np.quantile(loss, level))
    return var, float(loss[loss >= var].mean())


def _beyond(values: np.ndarray, barrier: float, s0: np.ndarray) -> np.ndarray:
    return values >= barrier if barrier > s0[0] else values <= barrier


def prob_touch(paths: np.ndarray, barrier: float) -> float:
    """Share of paths that reach `barrier` at any point (up if above the start, else down)."""
    return float(_beyond(paths, barrier, paths[:, 0]).any(axis=1).mean())


def prob_end_beyond(paths: np.ndarray, level: float) -> float:
    """Share of paths that finish beyond `level` (same direction rule as prob_touch)."""
    return float(_beyond(paths[:, -1], level, paths[:, 0]).mean())
