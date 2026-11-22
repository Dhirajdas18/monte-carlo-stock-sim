"""Daily returns, annualised mean/volatility and tail diagnostics."""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simple_returns(prices):
    """P_t / P_{t-1} - 1. What your account shows; adds across stocks in a portfolio."""
    return prices.pct_change().iloc[1:]


def log_returns(prices):
    """ln(P_t / P_{t-1}). Adds across time; the input for price-path simulation."""
    if (prices <= 0).any(axis=None):
        raise ValueError("log returns need strictly positive prices")
    return np.log(prices / prices.shift(1)).iloc[1:]


def annualise(returns: pd.DataFrame, periods: int = TRADING_DAYS) -> pd.DataFrame:
    """Mean scales with time (x periods); volatility with the square root of time."""
    return pd.DataFrame(
        {"mean": returns.mean() * periods, "vol": returns.std(ddof=1) * np.sqrt(periods)}
    ).T


def tail_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """How far the returns are from a normal bell curve. Normal: skew 0, excess kurtosis 0."""
    z = (returns - returns.mean()) / returns.std(ddof=1)
    return pd.DataFrame(
        {
            "skew": returns.skew(),
            "excess_kurtosis": returns.kurt(),
            "worst_day": returns.min(),
            "n_beyond_4sd": (z.abs() > 4).sum(),
        }
    ).T
