import numpy as np
import pytest

from mc.gbm import gbm_paths
from mc.risk import percentile_bands, prob_end_beyond, prob_touch, var_cvar


def test_var_cvar_known_losses():
    pnl = -np.arange(1, 101, dtype=float)  # losses of 1..100
    var, cvar = var_cvar(pnl, 0.95)
    assert var == pytest.approx(np.quantile(np.arange(1, 101), 0.95))
    assert cvar == pytest.approx(np.mean([96, 97, 98, 99, 100]))


def test_cvar_never_below_var():
    pnl = np.random.default_rng(0).normal(0, 1000, 50_000)
    var, cvar = var_cvar(pnl, 0.99)
    assert cvar >= var > 0


def test_var_negative_when_no_losses():
    var, _ = var_cvar(np.array([5.0, 10.0, 20.0]), 0.95)
    assert var < 0  # the "worst case" is still a gain


def test_touch_and_end_on_hand_built_paths():
    paths = np.array([[100, 80, 100], [100, 95, 85], [100, 105, 110]], dtype=float)
    assert prob_touch(paths, 90) == pytest.approx(2 / 3)
    assert prob_end_beyond(paths, 90) == pytest.approx(1 / 3)
    assert prob_touch(paths, 108) == pytest.approx(1 / 3)


def test_touch_at_least_end_on_gbm():
    paths = gbm_paths(100.0, 0.0, 0.3, 63, 20_000, seed=1)
    assert prob_touch(paths, 90) >= prob_end_beyond(paths, 90)


def test_percentile_bands_shape_and_order():
    paths = gbm_paths(100.0, 0.0, 0.3, 21, 5_000, seed=2)
    bands = percentile_bands(paths)
    assert bands.shape == (22, 3)
    assert (bands[5] <= bands[50]).all() and (bands[50] <= bands[95]).all()
