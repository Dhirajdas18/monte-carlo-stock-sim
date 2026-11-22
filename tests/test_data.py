import numpy as np
import pandas as pd
import pytest

from mc.data import load_prices


def _write(tmp_path, df):
    path = tmp_path / "prices.parquet"
    df.to_parquet(path)
    return path


def _frame():
    idx = pd.to_datetime(["2020-01-03", "2020-01-01", "2020-01-02", "2020-01-06"])
    return pd.DataFrame(
        {"RELIANCE": [3.0, 1.0, 2.0, 4.0], "NIFTY": [30.0, 10.0, np.nan, 40.0]},
        index=pd.Index(idx, name="date"),
    )


def test_load_prices_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_data.py"):
        load_prices(path=tmp_path / "nope.parquet")


def test_load_prices_drops_nan_rows_and_sorts(tmp_path):
    out = load_prices(path=_write(tmp_path, _frame()))
    assert out.index.is_monotonic_increasing
    assert not out.isna().any().any()
    assert len(out) == 3


def test_load_prices_column_subset_keeps_rows_nan_elsewhere(tmp_path):
    out = load_prices(["RELIANCE"], path=_write(tmp_path, _frame()))
    assert list(out.columns) == ["RELIANCE"]
    assert len(out) == 4


def test_load_prices_unknown_column(tmp_path):
    with pytest.raises(KeyError):
        load_prices(["NOPE"], path=_write(tmp_path, _frame()))


def _series(values):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def test_find_splits_detects_halving():
    from mc.data import find_splits

    raw = _series([100, 100, 50, 50])
    adj = _series([50, 50, 50, 50])
    assert list(find_splits(raw, adj)) == [raw.index[2]]


def test_find_splits_ignores_real_crash():
    from mc.data import find_splits

    raw = _series([100, 100, 50, 50])
    assert len(find_splits(raw, raw.copy())) == 0


def test_growth_of_rebases_first_row_and_keeps_ratios():
    from mc.data import growth_of

    prices = pd.DataFrame({"A": [10.0, 20.0, 15.0], "B": [2.0, 1.0, 4.0]})
    out = growth_of(prices, amount=100_000)
    assert (out.iloc[0] == 100_000).all()
    assert out["A"].tolist() == [100_000, 200_000, 150_000]
    assert out["B"].tolist() == [100_000, 50_000, 200_000]


def test_unadjust_splits_rebuilds_raw_prices():
    from mc.data import unadjust_splits

    close = _series([50, 50, 50, 25])  # split-adjusted
    splits = _series([0, 0, 2, 0])  # 2-for-1 on day 3
    raw = unadjust_splits(close, splits)
    assert raw.tolist() == [100, 100, 50, 25]


def test_unadjust_splits_compounds_multiple_splits():
    from mc.data import unadjust_splits

    close = _series([10, 10, 10])
    splits = _series([0, 2, 1.5])
    assert unadjust_splits(close, splits).tolist() == [30, 15, 10]


def test_flag_adjusted_drops_catches_missed_bonus():
    from mc.data import flag_adjusted_drops

    adj = pd.DataFrame({"ITC": [300.0, 301.0, 201.0, 202.0], "TCS": [10.0, 10.1, 10.2, 10.0]},
                       index=pd.date_range("2016-06-29", periods=4, freq="D"))
    flags = flag_adjusted_drops(adj, threshold=0.2)
    assert list(flags) == [("ITC", pd.Timestamp("2016-07-01"))]
