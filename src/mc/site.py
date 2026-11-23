"""Interactive dashboard: a Monte Carlo simulator for the five stocks and NIFTY, as one static page.

Python fits each model's settings per stock and history window (the Student-t degrees of freedom
need a maximum-likelihood fit); the browser simulates the paths with the same step rules as
mc.validate and mc.models.
"""

import json
import re
from pathlib import Path
from string import Template

import numpy as np
import pandas as pd
from scipy import stats

from mc import plots

FONT = "system-ui, -apple-system, \"Segoe UI\", Roboto, sans-serif"
ICON = (
    "data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22%3E"
    "%3Cpath d=%22M1 8 5 6 9 3 15 1M1 8 5 9 9 10 15 13M1 8 6 8 10 7 15 7%22 fill=%22none%22"
    " stroke=%22%232a78d6%22 stroke-width=%221.5%22/%3E%3C/svg%3E"
)
HERE = Path(__file__).parent
JS = (HERE / "sim.js").read_text(encoding="utf-8")
SIMULATOR = (HERE / "simulator.html").read_text(encoding="utf-8")
PAGE = Template((HERE / "page.html").read_text(encoding="utf-8"))
REPO = "https://github.com/Dhirajdas18/monte-carlo-stock-sim"
SCALE = 1e-5  # log returns stored as integers in units of 0.001%
WINDOWS = {"3y": 756, "5y": 1260, "all": None}  # trading days of history the models learn from
# Dark-mode steps of the same palette (as in the other learning repos)
DARK = {plots.SURFACE: "#1a1a19", plots.TEXT: "#f0efec", plots.TEXT_2: "#c3c2b7",
        plots.GRID: "#2e2d2b", plots.SERIES[0]: "#3987e5", plots.SERIES[1]: "#d95926",
        plots.SERIES[2]: "#199e70", plots.SERIES[3]: "#c98500"}


def model_params(hist: np.ndarray) -> dict:
    """Settings of the five models for one history, computed exactly as mc.models does."""
    hist = np.asarray(hist, dtype=float)
    sd = float(np.std(hist, ddof=1))
    dev = hist - hist.mean()
    jump = np.abs(dev) > 3.0 * sd
    ok = bool(jump.sum() >= 2)
    return {"n": int(len(hist)), "sd": sd, "df": max(float(stats.t.fit(hist)[0]), 3.0),
            "jumpOk": ok, "lam": float(jump.mean()) if ok else 0.0,
            "mJ": float(dev[jump].mean()) if ok else 0.0,
            "sJ": float(dev[jump].std(ddof=1)) if ok else 0.0,
            "sdD": float(dev[~jump].std(ddof=1)) if ok else sd}


def page_data(prices: pd.DataFrame) -> dict:
    from mc.data import INDEX, SYMBOLS

    names = SYMBOLS + [INDEX]
    out = {"names": names, "windows": list(WINDOWS), "r": {}, "params": {}, "last": {},
           "asof": f"{prices.index[-1]:%d %b %Y}"}
    for s in names:
        lr = np.log(prices[s]).diff().dropna()
        out["r"][s] = [int(round(x / SCALE)) for x in lr]
        out["last"][s] = float(prices[s].iloc[-1])
        out["params"][s] = {w: model_params(lr.to_numpy() if n is None else lr.to_numpy()[-n:])
                            for w, n in WINDOWS.items()}
    return out


def simulator_section(prices: pd.DataFrame) -> str:
    return (SIMULATOR.replace("$js", JS)
            .replace("$data", json.dumps(page_data(prices), separators=(",", ":"))))


SECTIONS = [
    ("Simulator",
     "Pick a stock or NIFTY, a model and how much history it learns from, and the page "
     "simulates 2,000 futures for your rupee amount. Every model has zero drift (the expected "
     "price stays at today’s): it estimates the spread of outcomes, not the direction. Compare "
     "the chance of ever touching a loss level with the chance of ending below it.",
     simulator_section),
]


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build_html(prices: pd.DataFrame) -> str:
    parts, nav = [], []
    for title, blurb, make in SECTIONS:
        sid = _slug(title)
        nav.append(f'<li><a href="#{sid}">{title}</a></li>')
        parts.append(f'<section id="{sid}">\n<h2>{title}</h2>\n<p>{blurb}</p>\n{make(prices)}\n'
                     "</section>")
    return PAGE.substitute(
        icon=ICON, font=FONT, repo=REPO, updated=f"{prices.index[-1]:%d %b %Y}",
        surface=plots.SURFACE, text=plots.TEXT, text2=plots.TEXT_2, grid=plots.GRID,
        accent=plots.SERIES[0], muted=plots.MUTED, d_surface=DARK[plots.SURFACE],
        d_text=DARK[plots.TEXT], d_text2=DARK[plots.TEXT_2], d_grid=DARK[plots.GRID],
        d_accent=DARK[plots.SERIES[0]], d_muted="#6b6a64",
        dark=json.dumps({k.lower(): v for k, v in DARK.items()}), dark_lines="{}",
        nav="\n".join(nav), sections="\n".join(parts),
    )
