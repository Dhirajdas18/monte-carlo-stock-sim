import numpy as np
import pandas as pd
import pytest

from mc.strategy import (
    kelly_fraction,
    luck_probability,
    max_drawdown,
    momentum_returns,
    ruin_probability,
    simulate_equity,
)


def _prices(n=600, k=4, seed=0):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.015, (n, k))
    idx = pd.bdate_range("2015-01-01", periods=n)
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx,
                        columns=list("ABCD")[:k])


def test_max_drawdown_known_values():
    assert max_drawdown(np.array([1.0, 2.0, 1.0, 3.0])) == pytest.approx(0.5)
    assert max_drawdown(np.array([1.0, 1.1, 1.2])) == pytest.approx(0.0)
    rows = max_drawdown(np.array([[1.0, 2.0, 1.0], [1.0, 0.8, 1.2]]))
    assert rows == pytest.approx([0.5, 0.2])


def test_momentum_has_no_look_ahead():
    p = _prices()
    base = momentum_returns(p, lookback=126, top_n=2, rebalance=21)
    shocked = p.copy()
    shocked.iloc[-1] *= 3.0
    after = momentum_returns(shocked, lookback=126, top_n=2, rebalance=21)
    assert np.allclose(base.iloc[:-1], after.iloc[:-1])


def test_momentum_holds_top_names_equal_weight():
    idx = pd.bdate_range("2020-01-01", periods=6)
    p = pd.DataFrame({"UP": [1, 2, 4, 8, 16, 32], "FLAT": [1.0] * 6, "DOWN": [32, 16, 8, 4, 2, 1]},
                     index=idx, dtype=float)
    r = momentum_returns(p, lookback=2, top_n=1, rebalance=1)
    assert r.iloc[-1] == pytest.approx(1.0)  # holds UP, which doubles each day


def test_simulate_equity_flat_when_fraction_zero():
    r = pd.Series(np.random.default_rng(1).normal(0.001, 0.02, 500))
    eq = simulate_equity(r, 50, 100, fraction=0.0, seed=0)
    assert eq.shape == (100, 51)
    assert np.allclose(eq, 1.0)


def test_simulate_equity_block_and_seed():
    r = pd.Series(np.random.default_rng(2).normal(0.0, 0.01, 300))
    a = simulate_equity(r, 40, 20, block=10, seed=5)
    assert np.array_equal(a, simulate_equity(r, 40, 20, block=10, seed=5))
    assert (a[:, 0] == 1.0).all()


def test_ruin_rises_with_bet_size():
    r = pd.Series(np.random.default_rng(3).normal(0.0005, 0.015, 2500))
    k = kelly_fraction(r)
    assert k == pytest.approx(r.mean() / r.var())
    probs = [ruin_probability(r, f * k, 1000, 2000, seed=0) for f in (0.5, 1.0, 2.0)]
    assert probs[0] < probs[1] < probs[2]


def test_luck_probability():
    rng = np.random.default_rng(4)
    bench = pd.Series(rng.normal(0.0004, 0.01, 2000))
    noise = rng.normal(0, 0.005, 2000)
    same_edge = bench + (noise - noise.mean())  # zero true edge
    assert 0.35 < luck_probability(same_edge, bench, n_boot=2000, seed=0) < 0.65
    strong = bench + 0.002 + rng.normal(0, 0.002, 2000)
    assert luck_probability(strong, bench, n_boot=2000, seed=0) < 0.01


def test_leveraged_equity_cannot_go_negative_and_recover():
    r = pd.Series([0.0, -0.6, 0.5, 0.5])  # a -60% day at 2x leverage wipes the account
    eq = simulate_equity(r, 200, 50, fraction=2.0, seed=0)
    assert (eq >= 0).all()
    wiped = (eq == 0).any(axis=1)
    assert wiped.any()
    assert (eq[wiped, -1] == 0).all()  # once at zero, it stays at zero


def test_momentum_mid_series_shock_leaves_earlier_returns_unchanged():
    p = _prices()
    k = 400
    base = momentum_returns(p, lookback=126, top_n=2, rebalance=21)
    shocked = p.copy()
    shocked.iloc[k:] *= np.linspace(1.0, 4.0, len(p) - k)[:, None]
    after = momentum_returns(shocked, lookback=126, top_n=2, rebalance=21)
    cut = p.index[k - 1]
    assert np.allclose(base.loc[:cut], after.loc[:cut])
