"""Build data/prices.parquet.

Default: Yahoo Finance (no setup needed).
Optional: set MC_LAKE_DIR to a local NSE bhavcopy lake for cleaner corporate-action adjustment.
"""

import os
import sys
from pathlib import Path

from mc.data import DATA_PATH, INDEX, SYMBOLS, fetch_lake, fetch_yfinance, flag_adjusted_drops


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_PATH
    lake = os.environ.get("MC_LAKE_DIR")
    df = fetch_lake(Path(lake)) if lake else fetch_yfinance()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    source = "local lake" if lake else "Yahoo Finance"
    span = f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}"
    print(f"{source}: {len(df)} rows, {span} -> {out}")
    for sym, date in flag_adjusted_drops(df[SYMBOLS + [INDEX]]):
        print(f"  check {sym} {date:%Y-%m-%d}: adjusted price fell >20% in a day "
              "(possible unadjusted split/bonus)")


if __name__ == "__main__":
    main()
