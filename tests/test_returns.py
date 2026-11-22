import math

import numpy as np
import pandas as pd
import pytest

from mc.returns import annualise, log_returns, simple_returns, tail_stats


def _prices(values):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"A": values}, index=idx, dtype=float)


def test_simple_and_log_returns_known_values():
    p = _prices([100, 110, 99])
    assert simple_returns(p)["A"].tolist() == pytest.approx([0.10, -0.10])
    assert log_returns(p)["A"].tolist() == pytest.approx([math.log(1.1), math.log(0.9)])


def test_log_returns_add_up_to_total_move():
    p = _prices([100, 120, 90, 130])
    assert log_returns(p)["A"].sum() == pytest.approx(math.log(130 / 100))


def test_log_returns_reject_non_positive_prices():
    with pytest.raises(ValueError):
        log_returns(_prices([100, 0]))


def test_annualise_constant_return():
    r = pd.DataFrame({"A": [0.001] * 10})
    out = annualise(r)
    assert out.loc["mean", "A"] == pytest.approx(0.252)
    assert out.loc["vol", "A"] == pytest.approx(0.0, abs=1e-12)


def test_annualise_vol_scales_with_sqrt_time():
    r = pd.DataFrame({"A": [0.01, -0.01]})
    assert annualise(r).loc["vol", "A"] == pytest.approx(0.0141421356 * math.sqrt(252), rel=1e-6)


def test_tail_stats_normal_data_has_no_excess_kurtosis():
    rng = np.random.default_rng(0)
    r = pd.DataFrame({"A": rng.normal(0, 0.01, 200_000)})
    out = tail_stats(r)
    assert abs(out.loc["excess_kurtosis", "A"]) < 0.05
    assert list(out.index) == ["skew", "excess_kurtosis", "worst_day", "n_beyond_4sd"]
