import numpy as np
import pytest
from scipy import stats

from mc.models import (
    MODELS,
    block_bootstrap_model,
    bootstrap_model,
    student_t_model,
)


def _fat_history(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    return 0.01 * rng.standard_t(3, n) / np.sqrt(3)


def _clustered_history(n=4000, seed=1):
    """Calm and stormy spells: volatility clusters, like real markets."""
    rng = np.random.default_rng(seed)
    vol = np.repeat(rng.choice([0.005, 0.03], size=n // 50), 50)
    return vol * rng.standard_normal(n)


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_shape_start_and_seed(name):
    model = MODELS[name]
    hist = _fat_history()
    a = model(hist, 100.0, 21, 500, np.random.default_rng(3))
    assert a.shape == (500, 22)
    assert (a[:, 0] == 100.0).all() and (a > 0).all()
    assert np.array_equal(a, model(hist, 100.0, 21, 500, np.random.default_rng(3)))
    assert not np.array_equal(a, model(hist, 100.0, 21, 500, np.random.default_rng(4)))


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_matches_history_volatility(name):
    hist = _fat_history()
    paths = MODELS[name](hist, 100.0, 21, 20_000, np.random.default_rng(5))
    sim_vol = np.diff(np.log(paths), axis=1).std()
    assert abs(sim_vol / hist.std() - 1) < 0.05


def test_bootstrap_only_replays_history():
    hist = np.array([0.01, -0.02, 0.005, 0.03])
    paths = bootstrap_model(hist, 100.0, 10, 50, np.random.default_rng(0))
    steps = np.round(np.diff(np.log(paths), axis=1), 12)
    centred = np.round(hist - hist.mean() - 0.5 * hist.var(ddof=1), 12)
    assert set(np.unique(steps)) <= set(centred)


def test_block_bootstrap_keeps_volatility_clusters():
    hist = _clustered_history()
    paths = block_bootstrap_model(hist, 100.0, 200, 2000, np.random.default_rng(2), block=20)
    a = np.abs(np.diff(np.log(paths), axis=1))
    lag1 = np.mean([np.corrcoef(row[:-1], row[1:])[0, 1] for row in a[:300]])
    iid = np.abs(np.diff(np.log(bootstrap_model(hist, 100.0, 200, 300,
                                                np.random.default_rng(2))), axis=1))
    lag1_iid = np.mean([np.corrcoef(row[:-1], row[1:])[0, 1] for row in iid])
    assert lag1 > 0.05 and lag1 > lag1_iid + 0.05


def test_student_t_model_has_fat_tails():
    paths = student_t_model(_fat_history(), 100.0, 1, 200_000, np.random.default_rng(6))
    assert stats.kurtosis(np.diff(np.log(paths), axis=1).ravel()) > 1
