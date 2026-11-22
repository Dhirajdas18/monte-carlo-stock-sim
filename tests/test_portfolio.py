import numpy as np
import pandas as pd
import pytest

from mc.portfolio import (
    bootstrap_days_paths,
    calm_vs_crash_corr,
    correlated_gbm_paths,
    portfolio_value,
)

S0 = np.array([100.0, 50.0, 20.0])
SIG = np.array([0.2, 0.3, 0.25])
CORR = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.5], [0.3, 0.5, 1.0]])


def _sim_corr(paths):
    r = np.diff(np.log(paths), axis=1).reshape(-1, paths.shape[2])
    return np.corrcoef(r, rowvar=False)


def test_correlated_paths_shape_start_and_target_correlation():
    paths = correlated_gbm_paths(S0, SIG, CORR, days=21, n_paths=20_000, seed=0)
    assert paths.shape == (20_000, 22, 3)
    assert np.allclose(paths[:, 0, :], S0)
    assert np.allclose(_sim_corr(paths), CORR, atol=0.02)


def test_identity_correlation_gives_independent_assets():
    paths = correlated_gbm_paths(S0, SIG, np.eye(3), 21, 20_000, seed=1)
    off = _sim_corr(paths)[~np.eye(3, dtype=bool)]
    assert np.abs(off).max() < 0.02


def test_impossible_correlation_matrix_raises():
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    with pytest.raises(np.linalg.LinAlgError):
        correlated_gbm_paths(S0, SIG, bad, 5, 10, seed=0)


def test_correlated_paths_seeded():
    a = correlated_gbm_paths(S0, SIG, CORR, 5, 10, seed=3)
    assert np.array_equal(a, correlated_gbm_paths(S0, SIG, CORR, 5, 10, seed=3))
    assert not np.array_equal(a, correlated_gbm_paths(S0, SIG, CORR, 5, 10, seed=4))


def test_bootstrap_days_keeps_same_day_moves_together():
    hist = pd.DataFrame({"A": [0.01, -0.02, 0.03], "B": [0.10, -0.20, 0.30]})
    paths = bootstrap_days_paths(hist, np.array([1.0, 1.0]), 30, 50, seed=0)
    steps = np.diff(np.log(paths), axis=1)
    centred = hist - hist.mean() - 0.5 * hist.var(ddof=1)
    pairs = {tuple(np.round(row, 10)) for row in centred.to_numpy()}
    assert {tuple(np.round(row, 10)) for row in steps.reshape(-1, 2)} <= pairs


def test_portfolio_value_starts_at_capital_and_is_buy_and_hold():
    paths = np.array([[[100.0, 10.0], [200.0, 10.0]]])  # A doubles, B flat
    value = portfolio_value(paths, np.array([0.5, 0.5]), 1000.0)
    assert value.shape == (1, 2)
    assert value[0, 0] == pytest.approx(1000.0)
    assert value[0, 1] == pytest.approx(1500.0)


def test_calm_vs_crash_correlation_splits_on_market_days():
    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.01, 5000)
    crash = m <= np.quantile(m, 0.1)
    x = np.where(crash, m + rng.normal(0, 0.002, 5000), rng.normal(0, 0.01, 5000))
    df = pd.DataFrame({"X": x, "NIFTY": m})
    calm, bad = calm_vs_crash_corr(df, market="NIFTY", q=0.1)
    assert bad.loc["X", "NIFTY"] > 0.8 and abs(calm.loc["X", "NIFTY"]) < 0.1
