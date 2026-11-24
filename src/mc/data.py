"""Price data: fetching (Yahoo Finance or a local NSE bhavcopy lake) and loading."""

import time
from pathlib import Path

import pandas as pd

SYMBOLS = ["RELIANCE", "HDFCBANK", "INFY", "TCS", "ITC"]
INDEX = "NIFTY"
START = "2013-01-01"
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "prices.parquet"


def fetch_yfinance(start: str = START, end: str | None = None) -> pd.DataFrame:
    """Daily adjusted + raw closes from Yahoo Finance (free, anyone can run it)."""
    import yfinance as yf

    tickers = {f"{s}.NS": s for s in SYMBOLS} | {"^NSEI": INDEX}
    # threads=False: parallel downloads race on yfinance's SQLite cache ("database is locked")
    # and silently drop a ticker, so download one at a time and retry any that come back empty.
    for attempt in range(3):
        raw = yf.download(
            list(tickers), start=start, end=end, auto_adjust=False, actions=True,
            progress=False, threads=False,
        )
        adj_close = raw["Adj Close"]
        missing = [t for t in tickers if t not in adj_close or adj_close[t].isna().all()]
        if not missing:
            break
        time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"Yahoo returned no data for {missing} after 3 attempts")
    adj = raw["Adj Close"].rename(columns=tickers)
    close = raw["Close"].rename(columns=tickers)  # Yahoo's Close is already split-adjusted
    splits = raw["Stock Splits"].rename(columns=tickers).fillna(0)
    unadjusted = pd.concat(
        {f"{s}_raw": unadjust_splits(close[s], splits[s]) for s in SYMBOLS}, axis=1
    )
    out = pd.concat([adj[SYMBOLS + [INDEX]], unadjusted], axis=1)
    out.index = pd.DatetimeIndex(out.index.tz_localize(None), name="date")
    return out


def unadjust_splits(close: pd.Series, splits: pd.Series) -> pd.Series:
    """Undo split adjustment: before each split, the traded price was `ratio` times higher.

    `splits` holds the ratio on the split day (2.0 for 2-for-1, 1.5 for a 1:2 bonus), 0 elsewhere.
    """
    ratio = splits.where(splits > 0, 1.0)
    # factor_t = product of all split ratios strictly after day t
    after = ratio[::-1].cumprod()[::-1].shift(-1, fill_value=1.0)
    return close * after


def fetch_lake(lake_dir: Path) -> pd.DataFrame:
    """Same table built from a local NSE bhavcopy lake (panel.parquet + market.parquet).

    Check its adjustment first: a lake that misses a bonus or demerger shows fake crashes.
    """
    lake_dir = Path(lake_dir)
    panel = pd.read_parquet(
        lake_dir / "panel.parquet", columns=["date", "symbol", "close", "adj_close"]
    )
    panel = panel[panel["symbol"].isin(SYMBOLS)]
    adj = panel.pivot_table(index="date", columns="symbol", values="adj_close")[SYMBOLS]
    close = panel.pivot_table(index="date", columns="symbol", values="close")[SYMBOLS]
    market = pd.read_parquet(lake_dir / "market.parquet", columns=["date", "nifty"])
    nifty = market.set_index("date")["nifty"].rename(INDEX)
    out = pd.concat([adj, nifty, close.add_suffix("_raw")], axis=1)
    out = out[out.index >= START]
    out.index = pd.DatetimeIndex(out.index, name="date")
    return out


def load_prices(columns: list[str] | None = None, path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the price table, keeping only rows where every requested column has a price."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python scripts/fetch_data.py` first."
        )
    df = pd.read_parquet(path)
    if columns is not None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise KeyError(f"columns not in price table: {missing}")
        df = df[columns]
    return df.sort_index().dropna(how="any")


def find_splits(raw: pd.Series, adj: pd.Series, threshold: float = 0.2) -> pd.DatetimeIndex:
    """Days where the raw close drops by more than `threshold` but the adjusted close barely moves.

    That pattern is a split or bonus issue (share count changed), not a real loss.
    """
    raw_move = raw.pct_change()
    adj_move = adj.pct_change()
    fake = (raw_move < -threshold) & (adj_move.abs() < threshold / 4)
    return pd.DatetimeIndex(raw.index[fake.to_numpy()])


def growth_of(prices: pd.DataFrame, amount: float = 100_000) -> pd.DataFrame:
    """Value of `amount` invested in each column on the first row, held to the end."""
    return prices / prices.iloc[0] * amount


def flag_adjusted_drops(
    adj: pd.DataFrame, threshold: float = 0.2
) -> list[tuple[str, pd.Timestamp]]:
    """(symbol, date) pairs where an *adjusted* price falls more than `threshold` in a day.

    Large-cap stocks rarely do that for real, so each hit is worth checking for a corporate
    action the data source failed to adjust (e.g. a missed 1:2 bonus shows as about -33%).
    """
    moves = adj.pct_change().stack()
    hits = moves[moves < -threshold]
    return [(sym, date) for date, sym in hits.index]
