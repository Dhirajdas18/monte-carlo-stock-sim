import json

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mc.site import build_html, model_params, page_data


def _prices():
    idx = pd.bdate_range("2013-01-01", "2026-11-20")
    rng = np.random.default_rng(0)
    cols = ["RELIANCE", "HDFCBANK", "INFY", "TCS", "ITC", "NIFTY"]
    r = rng.standard_t(4, (len(idx), len(cols))) * 0.012
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=cols)


def test_model_params_match_the_python_models():
    hist = np.random.default_rng(1).standard_t(4, 1500) * 0.015
    p = model_params(hist)
    sd = float(np.std(hist, ddof=1))
    assert p["sd"] == pytest.approx(sd)
    assert p["df"] == pytest.approx(max(float(stats.t.fit(hist)[0]), 3.0))
    dev = hist - hist.mean()
    jump = np.abs(dev) > 3 * sd
    assert p["jumpOk"] == bool(jump.sum() >= 2)
    assert p["lam"] == pytest.approx(jump.mean())
    assert p["mJ"] == pytest.approx(dev[jump].mean())
    assert p["sJ"] == pytest.approx(dev[jump].std(ddof=1))
    assert p["sdD"] == pytest.approx(dev[~jump].std(ddof=1))


def test_page_data_has_each_window_and_compact_returns():
    d = page_data(_prices())
    assert set(d["windows"]) == {"3y", "5y", "all"}
    r = d["r"]["INFY"]
    assert all(isinstance(x, int) for x in r) and len(r) == len(_prices()) - 1
    assert d["params"]["INFY"]["3y"]["n"] == 756
    assert d["last"]["INFY"] == pytest.approx(_prices()["INFY"].iloc[-1])
    assert json.loads(json.dumps(d)) == d


def test_page_builds():
    html = build_html(_prices())
    assert "function simulate" in html and 'id="simulator"' in html
    assert "$" + "js" not in html and "$" + "data" not in html
