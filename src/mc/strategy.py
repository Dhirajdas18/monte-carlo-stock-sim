"""Strategy risk: resample a strategy's daily returns to see the drawdowns, ruin odds and luck
that a single backtest equity curve hides."""

import numpy as np
import pandas as pd


def momentum_returns(prices: pd.DataFrame, lookback: int = 252, top_n: int = 2,
                     rebalance: int = 21) -> pd.Series:
    """Daily simple returns of an equal-weight top-`top_n` momentum portfolio, no costs.

    Every `rebalance` days, rank stocks by their return over the past `lookback` days using
    closes up to and including that day, then hold the leaders from the next day on.
    """
    rets = prices.pct_change()
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    current = pd.Series(0.0, index=prices.columns)
    for i in range(len(prices)):
        if i >= lookback and (i - lookback) % rebalance == 0:
            past = prices.iloc[i] / prices.iloc[i - lookback] - 1
            current = pd.Series(0.0, index=prices.columns)
            current[past.nlargest(top_n).index] = 1.0 / top_n
        weights.iloc[i] = current
    # Weights set at day i's close earn day i+1's return.
    out = (weights.shift(1) * rets).sum(axis=1)
    return out.iloc[lookback + 1 :]


def max_drawdown(equity: np.ndarray) -> np.ndarray | float:
    """Largest fall from a running peak, as a positive fraction; per row for 2-D input."""
    eq = np.asarray(equity, dtype=float)
    dd = 1 - eq / np.maximum.accumulate(eq, axis=-1)
    out = dd.max(axis=-1)
    return float(out) if np.ndim(out) == 0 else out


def _resample(r: np.ndarray, days: int, n_paths: int, block: int | None,
              rng: np.random.Generator) -> np.ndarray:
    if not block:
        return r[rng.integers(0, len(r), (n_paths, days))]
    n_blocks = -(-days // block)
    starts = rng.integers(0, len(r), (n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)[:, :days] % len(r)
    return r[idx]


def simulate_equity(daily_returns: pd.Series, days: int, n_paths: int, block: int | None = None,
                    fraction: float = 1.0, seed: int | None = None) -> np.ndarray:
    """Equity curves starting at 1, built by resampling the strategy's own daily returns.

    `fraction` scales exposure: 0.5 = half the capital in the strategy, 2.0 = 2x leverage.
    """
    rng = np.random.default_rng(seed)
    r = _resample(np.asarray(daily_returns, float), days, n_paths, block, rng)
    # A leveraged loss beyond -100% wipes the account; equity can't go negative and recover.
    growth = np.cumprod(np.maximum(1 + fraction * r, 0.0), axis=1)
    return np.hstack([np.ones((n_paths, 1)), growth])


def kelly_fraction(daily_returns: pd.Series) -> float:
    """Growth-maximising exposure for small, frequent bets: mean / variance."""
    return float(daily_returns.mean() / daily_returns.var())


def ruin_probability(daily_returns: pd.Series, fraction: float, days: int, n_paths: int,
                     ruin_level: float = 0.5, block: int | None = None,
                     seed: int | None = None) -> float:
    """Chance that equity ever falls to `ruin_level` of the starting capital."""
    eq = simulate_equity(daily_returns, days, n_paths, block, fraction, seed)
    return float((eq.min(axis=1) <= ruin_level).mean())


def luck_probability(strategy: pd.Series, benchmark: pd.Series, n_boot: int = 10_000,
                     block: int = 20, seed: int | None = None) -> float:
    """Share of block-bootstrap resamples in which the strategy's average daily excess return
    over the benchmark is zero or worse: how easily the measured edge could be noise."""
    excess = (strategy - benchmark).dropna().to_numpy(float)
    rng = np.random.default_rng(seed)
    means = _resample(excess, len(excess), n_boot, block, rng).mean(axis=1)
    return float((means <= 0).mean())
