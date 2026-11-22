"""`mc-report <topic>`: build a topic's charts into out/ and print its key numbers."""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mc import plots
from mc.data import INDEX, SYMBOLS, find_splits, growth_of, load_prices

OUT = Path("out")


def _years(index) -> float:
    return (index[-1] - index[0]).days / 365.25


def report_data(out_dir: Path) -> dict:
    """Adjusted vs raw close; growth of ₹1 lakh."""
    raw_cols = [f"{s}_raw" for s in SYMBOLS]
    df = load_prices(SYMBOLS + raw_cols)
    result: dict = {}

    # 1. Biggest fake crash: raw close falls, adjusted close does not.
    splits = {s: find_splits(df[f"{s}_raw"], df[s]) for s in SYMBOLS}
    result["split_days"] = "; ".join(
        f"{s}: {', '.join(d.strftime('%Y-%m-%d') for d in days)}"
        for s, days in splits.items()
        if len(days)
    ) or "none"
    drops = {
        (s, d): df[f"{s}_raw"].pct_change().loc[d] for s, days in splits.items() for d in days
    }
    if drops:
        (sym, day), drop = min(drops.items(), key=lambda kv: kv[1])
        result["biggest_fake_drop"] = f"{sym} {day:%Y-%m-%d} raw {drop:.1%}"
        win = df.loc[day - np.timedelta64(60, "D") : day + np.timedelta64(60, "D")]
        fig, ax = plt.subplots(figsize=(8, 4))
        raw_r = win[f"{sym}_raw"] / win[f"{sym}_raw"].iloc[0] * 100
        adj_r = win[sym] / win[sym].iloc[0] * 100
        ax.plot(raw_r.index, raw_r, color=plots.MUTED, label="Raw close")
        ax.plot(adj_r.index, adj_r, color=plots.COLORS[sym], label="Adjusted close")
        ax.axvline(day, color=plots.TEXT_2, lw=1, ls="--")
        ax.annotate(
            f"Split/bonus {day:%d %b %Y}\nraw {drop:.0%}, adjusted ~0%",
            (day, raw_r.min()),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9,
            color=plots.TEXT,
        )
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.set_title(f"{sym}: the raw price 'crashes', the investment didn't")
        ax.set_ylabel("Price, rebased to 100")
        ax.legend(loc="upper left")
        plots.save(fig, out_dir, "data-raw-vs-adjusted")

    # 2. Growth of ₹1 lakh (adjusted prices), log scale.
    prices = load_prices(SYMBOLS + [INDEX])
    growth = growth_of(prices)
    yrs = _years(prices.index)
    fig, ax = plt.subplots(figsize=(9, 5))
    ends = growth.iloc[-1].sort_values()
    for col in growth.columns:
        ax.plot(growth.index, growth[col], color=plots.COLORS[col], lw=2 if col != INDEX else 2.5)
    # Direct labels, nudged apart so they don't collide.
    last_y: float | None = None
    for col in ends.index:
        y = ends[col]
        if last_y is not None and y / last_y < 1.17:
            y = last_y * 1.17
        last_y = y
        cagr = (ends[col] / 100_000) ** (1 / yrs) - 1
        plots.label_line_end(
            ax, growth.index[-1], y, f"{col} ₹{ends[col] / 1e5:.1f}L ({cagr:.0%}/yr)",
            plots.COLORS[col],
        )
        result[f"{col}_final_lakh"] = round(ends[col] / 1e5, 2)
        result[f"{col}_cagr"] = round(cagr, 4)
    ax.set_yscale("log")
    ax.set_yticks([1e5, 2e5, 5e5, 1e6, 2e6])
    ax.set_yticklabels(["₹1L", "₹2L", "₹5L", "₹10L", "₹20L"])
    ax.set_title(f"₹1 lakh invested on {prices.index[0]:%d %b %Y} (log scale)")
    ax.set_xlim(right=growth.index[-1] + (growth.index[-1] - growth.index[0]) * 0.28)
    ax.set_xticks(pd.date_range("2014-01-01", growth.index[-1], freq="2YS"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plots.save(fig, out_dir, "data-growth-of-1-lakh")
    result["years"] = round(yrs, 1)
    result["trading_days"] = len(prices)
    return result


REPORTS = {"data": report_data}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1 or args[0] not in REPORTS:
        print(f"usage: mc-report <topic>   (topics: {', '.join(REPORTS)})")
        return 2
    plots.style()
    result = REPORTS[args[0]](OUT)
    for key, value in result.items():
        print(f"{key}: {value}")
    print(f"charts -> {OUT.resolve()}")
    return 0
