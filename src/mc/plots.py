"""Shared chart style. Colours follow the entity (a stock keeps its colour in every chart)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
# Categorical slots in fixed order (validated: CVD + normal-vision separation pass).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COLORS = {
    "RELIANCE": SERIES[0],
    "HDFCBANK": SERIES[1],
    "INFY": SERIES[2],
    "TCS": SERIES[3],
    "ITC": SERIES[4],
    "NIFTY": SERIES[5],
}
MUTED = "#a3a29c"


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_2,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": TEXT_2,
            "ytick.color": TEXT_2,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "legend.labelcolor": TEXT_2,
            "font.size": 10,
            "figure.dpi": 110,
        }
    )


def label_line_end(ax, x, y, text: str, color: str) -> None:
    """Direct label at a line's right end (identity never by colour alone)."""
    ax.annotate(
        text,
        (x, y),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=TEXT,
        bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": color, "lw": 1},
    )


def save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
