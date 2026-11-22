"""Build data/prices.parquet.

Default: Yahoo Finance (no setup needed).
Optional: set MC_LAKE_DIR to a local NSE bhavcopy lake for cleaner corporate-action adjustment.
"""

import os
import sys
from pathlib import Path

from mc.data import DATA_PATH, fetch_lake, fetch_yfinance


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_PATH
    lake = os.environ.get("MC_LAKE_DIR")
    df = fetch_lake(Path(lake)) if lake else fetch_yfinance()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    source = "local lake" if lake else "Yahoo Finance"
    span = f"{df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}"
    print(f"{source}: {len(df)} rows, {span} -> {out}")


if __name__ == "__main__":
    main()
