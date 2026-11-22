import math

import numpy as np
import pandas as pd
import pytest

from mc.gbm import drift_stderr, gbm_closed_form, gbm_paths, mc_error_curve


def test_shape_and_start_price():
    paths = gbm_paths(100.0, 0.1, 0.2, days=21, n_paths=50, seed=1)
    assert paths.shape == (50, 22)
    assert (paths[:, 0] == 100.0).all()
    assert (paths > 0).all()


def test_zero_vol_grows_at_drift_exactly():
    paths = gbm_paths(100.0, 0.1, 0.0, days=252, n_paths=3, seed=1)
    assert paths[:, -1] == pytest.approx(100.0 * math.exp(0.1))


def test_seed_reproducible_and_seeds_differ():
    a = gbm_paths(100.0, 0.1, 0.2, 10, 20, seed=7)
    assert np.array_equal(a, gbm_paths(100.0, 0.1, 0.2, 10, 20, seed=7))
    assert not np.array_equal(a, gbm_paths(100.0, 0.1, 0.2, 10, 20, seed=8))


def test_mean_and_median_match_closed_form():
    paths = gbm_paths(100.0, 0.1, 0.3, days=252, n_paths=200_000, seed=0)
    exact = gbm_closed_form(100.0, 0.1, 0.3, 1.0)
    assert paths[:, -1].mean() == pytest.approx(exact["mean"], rel=0.01)
    assert np.median(paths[:, -1]) == pytest.approx(exact["median"], rel=0.01)
    assert exact["median"] < exact["mean"]


def test_antithetic_needs_even_paths():
    with pytest.raises(ValueError):
        gbm_paths(100.0, 0.1, 0.2, 10, 21, seed=0, antithetic=True)


def test_antithetic_reduces_error_of_the_mean():
    exact = gbm_closed_form(100.0, 0.1, 0.3, 1.0)["mean"]

    def avg_err(anti):
        errs = [abs(gbm_paths(100.0, 0.1, 0.3, 252, 1000, seed=s, antithetic=anti)[:, -1].mean()
                    - exact) for s in range(20)]
        return np.mean(errs)

    assert avg_err(True) < avg_err(False)


def test_drift_stderr_is_vol_over_sqrt_years():
    r = pd.Series([0.01, -0.01] * 504)  # 1008 days = 4 years
    vol = r.std(ddof=1) * math.sqrt(252)
    assert drift_stderr(r) == pytest.approx(vol / math.sqrt(4))


def test_error_curve_shrinks_with_more_paths():
    curve = mc_error_curve(100.0, 0.1, 0.3, 252, [100, 10_000], seed=0, repeats=20)
    assert list(curve.columns) == ["n", "abs_error"]
    assert curve["abs_error"].iloc[1] < curve["abs_error"].iloc[0]
