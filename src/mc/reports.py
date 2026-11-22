"""`mc-report <topic>`: build a topic's charts into out/ and print its key numbers."""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mc import plots
from mc.data import INDEX, SYMBOLS, find_splits, growth_of, load_prices
from mc.returns import TRADING_DAYS, annualise, log_returns, simple_returns, tail_stats

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


def report_returns(out_dir: Path) -> dict:
    """Return distribution vs normal, volatility per stock, square-root-of-time check."""
    from scipy import stats

    cols = SYMBOLS + [INDEX]
    prices = load_prices(cols)
    lr = log_returns(prices)
    sr = simple_returns(prices)
    ann = annualise(lr)
    tails = tail_stats(lr)
    result: dict = {}

    # 1. RELIANCE daily returns vs the bell curve, log y-axis so the tails are visible.
    sym = "RELIANCE"
    r = lr[sym]
    mu, sd = r.mean(), r.std()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(-0.15, 0.15, 121)
    ax.hist(r, bins=bins, density=True, color=plots.COLORS[sym], alpha=0.85,
            label="Real daily returns")
    x = np.linspace(-0.15, 0.15, 600)
    ax.plot(x, stats.norm.pdf(x, mu, sd), color=plots.TEXT, lw=1.5,
            label="Normal (bell curve), same mean/vol")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.01)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Density (log scale)")
    worst = r.idxmin()
    ax.annotate(f"{worst:%d %b %Y}: {np.expm1(r.min()):+.1%}", (r.min(), 0.13), xytext=(12, 30),
                textcoords="offset points", fontsize=9, color=plots.TEXT,
                arrowprops={"arrowstyle": "-", "color": plots.TEXT_2})
    ax.set_title(f"{sym}: real days vs the bell curve. Big moves happen far more often")
    ax.legend(loc="upper right")
    plots.save(fig, out_dir, "returns-vs-normal")

    n = len(r)
    expected_4sd = 2 * stats.norm.sf(4) * n
    result["days"] = n
    result["expected_4sd_days_if_normal"] = round(expected_4sd, 2)
    for c in cols:
        result[f"{c}_beyond_4sd_days"] = int(tails.loc["n_beyond_4sd", c])
    for c in cols:
        result[f"{c}_excess_kurtosis"] = round(float(tails.loc["excess_kurtosis", c]), 1)
        result[f"{c}_worst_day"] = f"{lr[c].idxmin():%Y-%m-%d} {float(np.expm1(lr[c].min())):+.1%}"

    # 2. Annualised volatility per stock.
    vol = ann.loc["vol"].sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(vol.index, vol.values, color=[plots.COLORS[c] for c in vol.index], height=0.6)
    for i, (c, v) in enumerate(vol.items()):
        daily = v / np.sqrt(TRADING_DAYS)
        ax.text(v + 0.003, i, f"{v:.0%} a year  (≈ ±{daily:.1%} on a typical day)", va="center",
                fontsize=9, color=plots.TEXT)
        result[f"{c}_annual_vol"] = round(float(v), 3)
        result[f"{c}_daily_vol"] = round(float(daily), 4)
    ax.set_xlim(0, vol.max() * 1.75)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.set_title("Volatility: how much each one jumps around (annualised)")
    plots.save(fig, out_dir, "returns-volatility")

    # 3. Does volatility grow with the square root of time? Real vs the rule.
    horizons = [1, 5, 21, 63, 126, 252]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for c in [sym, INDEX]:
        logp = np.log(prices[c])
        real = [logp.diff(h).dropna().std() for h in horizons]
        rule = [lr[c].std() * np.sqrt(h) for h in horizons]
        ax.plot(horizons, rule, color=plots.COLORS[c], lw=1.5, ls="--")
        ax.plot(horizons, real, color=plots.COLORS[c], marker="o", ms=8, lw=2)
        plots.label_line_end(ax, horizons[-1], real[-1], f"{c} real", plots.COLORS[c])
        result[f"{c}_1y_vol_real"] = round(float(real[-1]), 3)
        result[f"{c}_1y_vol_sqrt_rule"] = round(float(rule[-1]), 3)
    ax.plot([], [], color=plots.TEXT_2, ls="--", lw=1.5, label="√time rule: daily vol × √days")
    ax.plot([], [], color=plots.TEXT_2, marker="o", lw=2,
            label="Real, measured over that many days")
    ax.set_xscale("log")
    ax.set_xticks(horizons, ["1 day", "1 wk", "1 mo", "3 mo", "6 mo", "1 yr"])
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylabel("Typical move (1 standard deviation)")
    ax.set_xlim(0.8, 600)
    ax.set_title("Risk grows with the square root of time, not with time")
    ax.legend(loc="upper left")
    plots.save(fig, out_dir, "returns-sqrt-time")

    # Simple vs log, on real data: average daily simple return overstates growth.
    yrs = _years(prices.index)
    cagr = (prices[sym].iloc[-1] / prices[sym].iloc[0]) ** (1 / yrs) - 1
    result[f"{sym}_avg_simple_x252"] = round(float(sr[sym].mean() * TRADING_DAYS), 4)
    result[f"{sym}_avg_log_x252"] = round(float(ann.loc["mean", sym]), 4)
    result[f"{sym}_actual_cagr"] = round(float(cagr), 4)
    result[f"{sym}_cagr_from_log"] = round(float(np.expm1(lr[sym].sum() / yrs)), 4)
    return result


REPORTS = {"data": report_data, "returns": report_returns}


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
