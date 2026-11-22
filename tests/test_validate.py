import numpy as np
import pandas as pd

from mc.gbm import gbm_paths
from mc.validate import coverage, gbm_model, qq_points


def _gbm_prices(n_days, seed=3, sigma=0.25):
    path = gbm_paths(100.0, 0.0, sigma, n_days, 1, seed=seed)[0]
    return pd.Series(path, index=pd.bdate_range("2000-01-03", periods=n_days + 1))


def test_gbm_model_shape_and_vol():
    hist = np.random.default_rng(0).normal(0, 0.01, 1000)
    paths = gbm_model(hist, 100.0, 21, 20_000, np.random.default_rng(1))
    assert paths.shape == (20_000, 22)
    assert (paths[:, 0] == 100.0).all()
    sim_vol = np.diff(np.log(paths), axis=1).std()
    assert abs(sim_vol / hist.std() - 1) < 0.02


def test_coverage_hits_target_when_model_is_true():
    # One history can be unlucky (a perfect model scored 86% on one seed), so average several.
    rates = []
    for seed in range(10, 16):
        cov = coverage(_gbm_prices(10_000, seed=seed), gbm_model, horizon=21, window=756,
                       step=21, n_paths=1000, seed=0)
        rates.append(cov["hit"].mean())
    assert list(cov.columns) == ["lo", "hi", "actual", "hit"]
    assert abs(np.mean(rates) - 0.90) < 0.025


def test_coverage_bands_use_no_future_data():
    prices = _gbm_prices(2000)
    cut = prices.index[1200]
    shocked = prices.copy()
    shocked[shocked.index > cut] *= 0.5
    a = coverage(prices, gbm_model, window=756, step=21, n_paths=500, seed=0)
    b = coverage(shocked, gbm_model, window=756, step=21, n_paths=500, seed=0)
    early = a.index <= cut
    assert np.allclose(a.loc[early, ["lo", "hi"]], b.loc[early, ["lo", "hi"]])


def test_qq_points_normal_data_on_diagonal():
    r = pd.Series(np.random.default_rng(5).normal(0.001, 0.02, 50_000))
    theo, sample = qq_points(r)
    slope = np.polyfit(theo, sample, 1)[0]
    assert abs(slope - 1) < 0.05
