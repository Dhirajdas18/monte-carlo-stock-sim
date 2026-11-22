"""`mc-report <topic>`: build a topic's charts into out/ and print its key numbers."""

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mc import plots
from mc.data import INDEX, SYMBOLS, find_splits, growth_of, load_prices
from mc.gbm import drift_stderr, gbm_closed_form, gbm_paths, mc_error_curve
from mc.models import MODELS
from mc.returns import TRADING_DAYS, annualise, log_returns, simple_returns, tail_stats
from mc.risk import prob_end_beyond, prob_touch, var_cvar
from mc.validate import coverage, gbm_model, qq_points

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


def _rupees(v, _=None) -> str:
    return f"₹{v:,.0f}"


def report_gbm(out_dir: Path) -> dict:
    """GBM baseline: path fan, mean vs median, path count, drift uncertainty."""
    sym = "RELIANCE"
    prices = load_prices([sym])[sym]
    lr = log_returns(prices)
    s0 = float(prices.iloc[-1])
    sigma = float(lr.std() * np.sqrt(TRADING_DAYS))
    mu_hist = float(lr.mean() * TRADING_DAYS + 0.5 * sigma**2)  # GBM drift = mean log return + σ²/2
    se = drift_stderr(lr)
    days, n = TRADING_DAYS, 100_000
    result: dict = {"s0": round(s0, 2), "as_of": f"{prices.index[-1]:%Y-%m-%d}",
                    "sigma": round(sigma, 4), "mu_hist": round(mu_hist, 4),
                    "drift_stderr": round(se, 4), "years_of_data": round(len(lr) / TRADING_DAYS, 1)}

    # 1. Fan of paths, drift 0 (see chart 4 for why).
    paths = gbm_paths(s0, 0.0, sigma, days, n, seed=42)
    final = paths[:, -1]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(days + 1)
    ax.plot(x, paths[:200].T, color=plots.COLORS[sym], lw=0.6, alpha=0.25)
    for q, ls in [(5, ":"), (50, "-"), (95, ":")]:
        band = np.percentile(paths, q, axis=0)
        ax.plot(x, band, color=plots.TEXT, lw=1.5, ls=ls)
        plots.label_line_end(ax, days, band[-1], f"{q}th pct ₹{band[-1]:,.0f}", plots.TEXT_2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_rupees))
    ax.set_xticks([0, 63, 126, 189, 252], ["today", "3 mo", "6 mo", "9 mo", "1 yr"])
    ax.set_xlim(0, days * 1.3)
    ax.set_title(f"{sym}: 200 of {n:,} simulated years (GBM, vol {sigma:.0%}, drift 0)")
    plots.save(fig, out_dir, "gbm-fan")
    for q in (5, 50, 95):
        result[f"final_p{q}"] = round(float(np.percentile(final, q)), 0)
    result["pct_paths_below_start"] = round(float((final < s0).mean()), 3)

    # 2. Mean vs median of the final price.
    exact = gbm_closed_form(s0, 0.0, sigma, 1.0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(final, bins=150, range=(0, np.percentile(final, 99.5)), color=plots.COLORS[sym],
            alpha=0.85)
    for val, name, ls in [(final.mean(), "Mean", "-"), (np.median(final), "Median", "--")]:
        ax.axvline(val, color=plots.TEXT, lw=1.5, ls=ls)
        ax.annotate(f"{name} ₹{val:,.0f}", (val, ax.get_ylim()[1] * 0.92), xytext=(5 if name ==
                    "Mean" else -5, 0), textcoords="offset points", fontsize=9, color=plots.TEXT,
                    ha="left" if name == "Mean" else "right")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_rupees))
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("Price in 1 year: a few huge winners pull the mean above the typical outcome")
    plots.save(fig, out_dir, "gbm-mean-vs-median")
    result["sim_mean"] = round(float(final.mean()), 0)
    result["sim_median"] = round(float(np.median(final)), 0)
    result["exact_mean"] = round(float(exact["mean"]), 0)
    result["exact_median"] = round(float(exact["median"]), 0)

    # 3. How many paths is enough? Error vs N, plain vs antithetic.
    ns = [100, 300, 1_000, 3_000, 10_000, 30_000]
    plain = mc_error_curve(s0, 0.0, sigma, days, ns, seed=0, repeats=30)
    anti = mc_error_curve(s0, 0.0, sigma, days, ns, seed=0, antithetic=True, repeats=30)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(plain.n, plain.abs_error, color=plots.SERIES[0], marker="o", ms=8, label="Plain")
    ax.plot(anti.n, anti.abs_error, color=plots.SERIES[1], marker="o", ms=8,
            label="Antithetic (each draw paired with its mirror)")
    guide = plain.abs_error.iloc[0] * np.sqrt(ns[0] / np.array(ns))
    ax.plot(ns, guide, color=plots.MUTED, ls="--", lw=1.5,
            label="1/√N: 100× the paths, 1/10 the error")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_rupees))
    ax.set_xlabel("Number of simulated paths")
    ax.set_ylabel("Average error in the simulated mean")
    ax.set_title("More paths, less error, but slowly")
    ax.legend(loc="lower left")
    plots.save(fig, out_dir, "gbm-paths-needed")
    for i, nn in enumerate(ns):
        result[f"err_plain_{nn}"] = round(float(plain.abs_error.iloc[i]), 1)
        result[f"err_anti_{nn}"] = round(float(anti.abs_error.iloc[i]), 1)

    # 4. Drift uncertainty: same vol, plausible drifts, very different "forecasts".
    cases = [("Drift 0", 0.0), (f"Hist. −2 SE ({mu_hist - 2 * se:+.0%})", mu_hist - 2 * se),
             (f"Historical ({mu_hist:+.0%})", mu_hist), (f"Hist. +2 SE ({mu_hist + 2 * se:+.0%})",
                                                       mu_hist + 2 * se)]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for i, (_name, mu) in enumerate(cases):
        f = gbm_paths(s0, mu, sigma, days, 50_000, seed=7)[:, -1]
        lo, med, hi = np.percentile(f, [5, 50, 95])
        ax.plot([lo, hi], [i, i], color=plots.SERIES[0], lw=6, solid_capstyle="round", alpha=0.35)
        ax.plot(med, i, "o", color=plots.SERIES[0], ms=9)
        ax.text(hi + s0 * 0.03, i, f"median ₹{med:,.0f}", va="center", fontsize=9,
                color=plots.TEXT)
        key = ["zero", "minus2se", "hist", "plus2se"][i]
        result[f"drift_{key}_p5_p50_p95"] = f"{lo:,.0f} / {med:,.0f} / {hi:,.0f}"
    ax.axvline(s0, color=plots.TEXT_2, lw=1, ls="--")
    ax.set_ylim(-0.7, len(cases) - 0.5)
    ax.annotate(f"today ₹{s0:,.0f}", (s0, -0.55), xytext=(4, 0),
                textcoords="offset points", fontsize=9, color=plots.TEXT_2)
    ax.set_yticks(range(len(cases)), [c[0] for c in cases])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_rupees))
    ax.set_xlim(right=ax.get_xlim()[1] * 1.12)
    ax.grid(axis="y", visible=False)
    ax.set_title("Price in 1 year (5th–95th pct): the drift guess moves the whole answer")
    plots.save(fig, out_dir, "gbm-drift-uncertainty")
    return result


def report_risk(out_dir: Path) -> dict:
    """VaR vs CVaR on ₹1 lakh (GBM vs real history), touch vs end probabilities."""
    sym, capital, days = "RELIANCE", 100_000, 21
    prices = load_prices([sym])[sym]
    lr = log_returns(prices)
    s0 = float(prices.iloc[-1])
    sigma = float(lr.std() * np.sqrt(TRADING_DAYS))
    result: dict = {"capital": capital, "horizon_days": days, "sigma": round(sigma, 4)}

    # 1-month P&L: GBM (drift 0) vs every real 21-day window since 2013.
    sim = gbm_paths(s0, 0.0, sigma, days, 200_000, seed=11, antithetic=True)
    pnl_sim = capital * (sim[:, -1] / s0 - 1)
    real_growth = (prices.shift(-days) / prices).dropna()
    pnl_real = capital * (real_growth.to_numpy() / np.exp(lr.mean() * days) - 1)  # de-trended
    for name, pnl in [("gbm", pnl_sim), ("real", pnl_real)]:
        for lvl in (0.95, 0.99):
            v, c = var_cvar(pnl, lvl)
            result[f"{name}_var{int(lvl * 100)}"] = round(v)
            result[f"{name}_cvar{int(lvl * 100)}"] = round(c)
    result["real_windows"] = len(pnl_real)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (name, pnl, color) in zip(
        axes, [("GBM model (bell curve)", pnl_sim, plots.SERIES[0]),
               ("Real 1-month windows, 2013–2026", pnl_real, plots.SERIES[1])], strict=True):
        ax.hist(pnl, bins=np.linspace(-45_000, 45_000, 91), density=True, color=color, alpha=0.85)
        v, c = var_cvar(pnl, 0.95)
        ax.axvline(-v, color=plots.TEXT, lw=1.5, ls="--")
        ax.axvline(-c, color=plots.TEXT, lw=1.5)
        top = ax.get_ylim()[1]
        ax.annotate(f"VaR 95%\n−₹{v:,.0f}", (-v, top * 0.8), xytext=(4, 0),
                    textcoords="offset points", fontsize=9, color=plots.TEXT)
        ax.annotate(f"CVaR 95%\n−₹{c:,.0f}", (-c, top * 0.55), xytext=(-4, 0), ha="right",
                    textcoords="offset points", fontsize=9, color=plots.TEXT)
        c99 = var_cvar(pnl, 0.99)[1]
        ax.text(0.98, 0.40, f"Worst 1% of months\naverage loss −₹{c99:,.0f}",
                transform=ax.transAxes, va="top", ha="right", fontsize=9, color=plots.TEXT,
                bbox={"boxstyle": "round,pad=0.3", "fc": plots.SURFACE, "ec": plots.GRID})
        ax.set_title(name, fontsize=11)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:+.0f}k"))
        ax.set_yticks([])
        ax.set_xlabel("Profit / loss in 1 month on ₹1 lakh (₹)")
    fig.suptitle(f"{sym}: the worst 5% of months, model vs reality", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    plots.save(fig, out_dir, "risk-var-cvar")

    # Touch vs end below, 3 months, for drops of 5–30%.
    horizon = 63
    paths = gbm_paths(s0, 0.0, sigma, horizon, 100_000, seed=12)
    drops = np.arange(0.05, 0.31, 0.025)
    touch = [prob_touch(paths, s0 * (1 - d)) for d in drops]
    end = [prob_end_beyond(paths, s0 * (1 - d)) for d in drops]
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.plot(drops, touch, color=plots.SERIES[1], marker="o", ms=8,
            label="Touches that level at any point")
    ax.plot(drops, end, color=plots.SERIES[0], marker="o", ms=8,
            label="Is still below it after 3 months")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"−{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Drop from today's price")
    ax.set_ylabel("Chance within 3 months")
    ax.set_title(f"{sym}: a stop-loss gets hit about twice as often as you'd guess")
    ax.legend(loc="upper right")
    plots.save(fig, out_dir, "risk-touch-vs-end")
    for d, t, e in zip(drops, touch, end, strict=True):
        if round(d, 3) in (0.05, 0.1, 0.15, 0.2):
            result[f"touch_minus{round(d * 100)}pct"] = round(t, 3)
            result[f"end_minus{round(d * 100)}pct"] = round(e, 3)

    # Does antithetic sampling help for tail risk (VaR 99%) like it does for the mean?
    exact = capital * (np.exp(-0.5 * sigma**2 * days / TRADING_DAYS
                              + sigma * np.sqrt(days / TRADING_DAYS) * -2.3263478740) - 1)
    for anti in (False, True):
        errs = []
        for k in range(40):
            f = gbm_paths(s0, 0.0, sigma, days, 2_000, seed=100 + k, antithetic=anti)[:, -1]
            errs.append(abs(var_cvar(capital * (f / s0 - 1), 0.99)[0] + exact))
        result[f"var99_err_2000paths_{'anti' if anti else 'plain'}"] = round(float(np.mean(errs)))
    result["var99_exact_gbm"] = round(-exact)
    return result


def _odds(months: float) -> str:
    if months >= 1e6:
        return f"10^{np.log10(months):.0f} months"
    return f"{months:,.0f} months"


def report_validate(out_dir: Path) -> dict:
    """Walk-forward coverage of the GBM 90% band, QQ plot, odds of the worst real months."""
    from scipy import stats

    cols = SYMBOLS + [INDEX]
    prices = load_prices(cols)
    lr = log_returns(prices)
    horizon, window = 21, 756
    result: dict = {"horizon_days": horizon, "window_days": window, "band": "90%"}

    # 1. QQ plot.
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    lim = 0
    for c in ["RELIANCE", INDEX]:
        theo, z = qq_points(lr[c])
        ax.plot(theo, z, "o", ms=3, color=plots.COLORS[c], alpha=0.6, label=c)
        lim = max(lim, np.abs(z).max())
    ax.plot([-4.5, 4.5], [-4.5, 4.5], color=plots.TEXT_2, lw=1.2, ls="--",
            label="Perfect bell curve")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-lim * 1.05, lim * 1.05)
    ax.set_xlabel("Where the day would sit on a bell curve (σ)")
    ax.set_ylabel("Where the real day sits (σ)")
    ax.set_title("Bell curve vs reality: the ends bend away")
    ax.legend(loc="upper left")
    plots.save(fig, out_dir, "validate-qq")

    # 2. Walk-forward coverage: every day, fit on the past 3 years, predict a 90% band 1 month out.
    covs = {}
    for c in cols:
        cov = coverage(prices[c], gbm_model, horizon=horizon, window=window, step=1,
                       n_paths=2000, seed=0)
        covs[c] = cov
        below = (cov["actual"] < cov["lo"]).mean()
        above = (cov["actual"] > cov["hi"]).mean()
        result[f"{c}_coverage"] = round(float(cov["hit"].mean()), 3)
        result[f"{c}_miss_below"] = round(float(below), 3)
        result[f"{c}_miss_above"] = round(float(above), 3)
        c99 = coverage(prices[c], gbm_model, horizon=horizon, window=window, step=1,
                       level=0.99, n_paths=4000, seed=1)
        result[f"{c}_miss_99band"] = round(float(1 - c99["hit"].mean()), 4)
        result[f"{c}_miss_99band_below"] = round(float((c99["actual"] < c99["lo"]).mean()), 4)
    result["forecasts_per_stock"] = len(covs[INDEX])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ends = {}
    for c in ["RELIANCE", INDEX]:
        roll = covs[c]["hit"].astype(float).rolling(252).mean()
        ax.plot(roll.index, roll, color=plots.COLORS[c], lw=2)
        ends[c] = (roll.index[-1], roll.iloc[-1])
    (x1, y1), (x2, y2) = ends["RELIANCE"], ends[INDEX]
    if abs(y1 - y2) < 0.03:
        y1, y2 = max(y1, y2) + 0.015, min(y1, y2) - 0.015
    plots.label_line_end(ax, x1, y1, "RELIANCE", plots.COLORS["RELIANCE"])
    plots.label_line_end(ax, x2, y2, INDEX, plots.COLORS[INDEX])
    ax.axhline(0.90, color=plots.TEXT, lw=1.2, ls="--")
    ax.annotate("Promised: 90%", (covs[INDEX].index[700], 0.90), xytext=(0, -14),
                textcoords="offset points", fontsize=9, color=plots.TEXT,
                bbox={"boxstyle": "round,pad=0.15", "fc": plots.SURFACE, "ec": "none"})
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(right=covs[INDEX].index[-1] + pd.Timedelta(days=420))
    ax.set_ylabel("Share of forecasts that landed in the band")
    ax.set_title("How often reality landed inside the model's 90% band (rolling 1 year)")
    plots.save(fig, out_dir, "validate-coverage")

    # 3. Worst real months: how rare did the model, using the vol it knew then, think they were?
    logp = np.log(prices[INDEX])
    move = logp.shift(-horizon) - logp
    picked: list = []
    for d in move.sort_values().index:
        if all(abs((d - q).days) > 60 for q in picked) and logp.index.get_loc(d) >= window:
            picked.append(d)
        if len(picked) == 5:
            break
    rows = []
    for d in picked:
        i = logp.index.get_loc(d)
        sd = lr[INDEX].iloc[i - window : i].std()
        m = move.loc[d]
        z = (m + 0.5 * sd**2 * horizon) / (sd * np.sqrt(horizon))
        prob = stats.norm.cdf(z)
        rows.append((d, np.expm1(m), prob))
        result[f"worst_{d:%Y-%m-%d}"] = f"{np.expm1(m):+.1%}, model odds 1 in {_odds(1 / prob)}"
    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = [f"{d:%b %Y}  ({r:+.0%})" for d, r, _ in rows]
    odds = [1 / pr for _, _, pr in rows]
    ax.barh(labels[::-1], odds[::-1], color=plots.COLORS[INDEX], height=0.55)
    ax.set_xscale("log")
    for i, o in enumerate(odds[::-1]):
        yrs = o / 12
        note = ("longer than the age of the universe" if yrs > 1.4e10
                else f"≈ once in {yrs:,.0f} years" if yrs >= 2 else "a normal bad month")
        ax.text(o * 1.6, i, f"1 in {_odds(o)}  ({note})", va="center", fontsize=9, color=plots.TEXT)
    ax.set_xlim(1, max(odds) ** 1.35)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("How rare the bell-curve model thought this month was (log scale)")
    ax.set_title("NIFTY's 5 worst months, as the model saw them beforehand")
    plots.save(fig, out_dir, "validate-worst-months")
    return result


def report_models(out_dir: Path) -> dict:
    """Five models through the same walk-forward test, plus tail size and speed."""
    import time

    syms = ["RELIANCE", INDEX]
    prices = load_prices(syms)
    lr = log_returns(prices)
    horizon, window, capital = 21, 756, 100_000
    names = list(MODELS)
    result: dict = {}
    miss99 = {s: [] for s in syms}
    miss99_dn = {s: [] for s in syms}
    cvar = {s: [] for s in syms}
    real_cvar = {}
    for sym in syms:
        growth = (prices[sym].shift(-horizon) / prices[sym]).dropna().to_numpy()
        real_pnl = capital * (growth / np.exp(lr[sym].mean() * horizon) - 1)
        real_cvar[sym] = var_cvar(real_pnl, 0.99)[1]
        result[f"{sym}_real_cvar99"] = round(real_cvar[sym])
        for name in names:
            t0 = time.perf_counter()
            c99 = coverage(prices[sym], MODELS[name], horizon=horizon, window=window, step=5,
                           level=0.99, n_paths=4000, seed=3)
            c90 = coverage(prices[sym], MODELS[name], horizon=horizon, window=window, step=5,
                           level=0.90, n_paths=4000, seed=4)
            secs = time.perf_counter() - t0
            m = 1 - c99["hit"].mean()
            dn = (c99["actual"] < c99["lo"]).mean()
            final = MODELS[name](lr[sym].to_numpy(), 1.0, horizon, 200_000,
                                 np.random.default_rng(9))[:, -1]
            cv = var_cvar(capital * (final - 1), 0.99)[1]
            miss99[sym].append(m)
            miss99_dn[sym].append(dn)
            cvar[sym].append(cv)
            key = name.split(" (")[0].replace(" ", "_").replace("-", "_").lower()
            result[f"{sym}_{key}"] = (f"cov90 {c90['hit'].mean():.1%} | miss99 {m:.2%} "
                                      f"(down {dn:.2%}) | cvar99 ₹{cv:,.0f} | {secs:.1f}s")
    result["forecasts"] = len(c99)

    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (title, data, target) in zip(
        axes, [("Missed the 99% band (promised 1%)", miss99, 0.01),
               ("…of which crashes below it (promised 0.5%)", miss99_dn, 0.005)], strict=True):
        for k, sym in enumerate(syms):
            ax.plot(data[sym], y + (k - 0.5) * 0.18, "o", ms=9, color=plots.COLORS[sym],
                    label=sym)
        ax.axvline(target, color=plots.TEXT, ls="--", lw=1.2)
        ax.annotate("target", (target, len(names) - 0.45), xytext=(4, 0),
                    textcoords="offset points", fontsize=9, color=plots.TEXT)
        ax.set_xlim(0, max(max(v) for v in data.values()) * 1.25)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[1].legend(loc="lower right")
    fig.suptitle("Walk-forward, 2016–2026: which model gets the extremes right?", x=0.01,
                 ha="left", fontsize=13, fontweight="bold")
    plots.save(fig, out_dir, "models-coverage")

    fig, ax = plt.subplots(figsize=(8, 4))
    for k, sym in enumerate(syms):
        ax.plot(cvar[sym], y + (k - 0.5) * 0.18, "o", ms=9, color=plots.COLORS[sym], label=sym)
        ax.axvline(real_cvar[sym], color=plots.COLORS[sym], ls="--", lw=1.2)
        ax.annotate(f"real {sym}", (real_cvar[sym], -0.55 + k * 0.3), xytext=(-4, 0),
                    ha="right", textcoords="offset points", fontsize=9, color=plots.TEXT)
    ax.set_yticks(y, names)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_rupees))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Average loss in the worst 1% of months, ₹1 lakh position")
    ax.set_ylim(len(names) - 0.5, -0.8)
    ax.set_title("How deep each model thinks a really bad month goes")
    ax.legend(loc="lower left")
    plots.save(fig, out_dir, "models-tail")
    return result


REPORTS = {"data": report_data, "returns": report_returns, "gbm": report_gbm, "risk": report_risk,
           "validate": report_validate, "models": report_models}


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
