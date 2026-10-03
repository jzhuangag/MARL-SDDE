"""Shared publication style for every experimental figure in the TSP-V3 paper."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.ticker import FormatStrFormatter


TEXT_WIDTH_IN = 7.15
COLUMN_WIDTH_IN = 3.50
DOUBLE_FIGURE_HEIGHT_IN = 2.25

FONT_FAMILY = "Times New Roman"
BASE_FONT_SIZE = 8.0
TICK_FONT_SIZE = 7.0
LEGEND_FONT_SIZE = 7.2
TITLE_FONT_SIZE = 8.0

GRID_COLOR = "#9E9E9E"
GRID_ALPHA = 0.28
GRID_LINEWIDTH = 0.55
CONFIDENCE_ALPHA = 0.14

PROPOSED_COLOR = "#0072B2"
Q_COLORS = {
    1: "#6E6E6E",
    2: "#56B4E9",
    4: "#009E73",
    8: "#E69F00",
    16: "#CC79A7",
    32: "#D55E00",
}
Q_LINESTYLES = {
    1: "--",
    2: ":",
    4: "-.",
    8: (0, (5, 1.4)),
    16: (0, (3, 1, 1, 1)),
    32: (0, (1.2, 1.2)),
}

RESOURCE_TICKS = (0.0, 0.25, 0.50, 0.75, 1.0)


def apply_publication_style() -> None:
    """Configure deterministic, IEEE-readable typography and line rendering."""

    mpl.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.serif": [FONT_FAMILY],
            "font.size": BASE_FONT_SIZE,
            "axes.labelsize": BASE_FONT_SIZE,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": TICK_FONT_SIZE,
            "ytick.labelsize": TICK_FONT_SIZE,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "legend.fontsize": LEGEND_FONT_SIZE,
            "lines.linewidth": 1.45,
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY,
            "mathtext.it": f"{FONT_FAMILY}:italic",
            "mathtext.bf": f"{FONT_FAMILY}:bold",
            "mathtext.sf": FONT_FAMILY,
            "mathtext.tt": FONT_FAMILY,
            "mathtext.cal": FONT_FAMILY,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def style_axis(ax: Axes, *, grid_axis: str = "both") -> None:
    """Apply the common grid, layer ordering, and spine treatment."""

    ax.grid(
        True,
        axis=grid_axis,
        color=GRID_COLOR,
        alpha=GRID_ALPHA,
        linewidth=GRID_LINEWIDTH,
    )
    ax.set_axisbelow(True)


def set_fraction_axis(
    ax: Axes,
    *,
    label: str,
    ticks: Iterable[float] = RESOURCE_TICKS,
) -> None:
    """Use the same charged-resource axis in every convergence plot."""

    ax.set_xlim(0.0, 1.0)
    ax.set_xticks(tuple(ticks))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.set_xlabel(label)


PDF_METADATA = {
    "Creator": "MARL-SDDE TSP-V3 figure pipeline",
    "Producer": "Matplotlib",
    "CreationDate": None,
    "ModDate": None,
}
