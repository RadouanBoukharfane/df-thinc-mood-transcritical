"""Common figure style: sizes, fonts, colours, line styles, and colormaps."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, MaxNLocator, NullLocator

# Text width of the elsarticle preprint layout (345 pt); figures are included at scale 1.
WIDTH = 345.0 / 72.27
PANEL_H = 1.55

INK = "#1a1a1a"
MUTED = "#6b6a66"
GRID = "#e4e3dd"
REF = "#1a1a1a"
REF_FINE = "#8a8984"

# Validated categorical palette (white surface, all pairs pass CVD and contrast checks).
BLUE, GREEN, ORANGE = "#256abf", "#199e70", "#d95926"
VIOLET = "#6250d6"

VARIANTS = {
    "df_ho": dict(color=BLUE, ls="-", marker="o", label="DF + HO"),
    "df_thinc": dict(color=GREEN, ls=(0, (4.5, 1.8)), marker="^", label="DF + THINC-BVD"),
    "full": dict(color=ORANGE, ls=(0, (6, 1.6, 1.2, 1.6)), marker="s", label="Complete method"),
}

# Single-hue blue ramp (light -> dark) for ordered series and scalar fields.
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def ordinal(n):
    """n colours from the blue ramp, readable on white (light end clipped)."""
    lo, hi = 2, len(BLUE_RAMP) - 1
    if n == 1:
        return [BLUE_RAMP[5]]
    return [BLUE_RAMP[round(lo + k * (hi - lo) / (n - 1))] for k in range(n)]


FIELD_CMAP = LinearSegmentedColormap.from_list("field", ["#f7f9fc"] + BLUE_RAMP[1:])
ACTIVE = "#184f95"
BINARY_CMAP = ListedColormap(["#f3f2ee", ACTIVE])
MOOD_CMAP = ListedColormap(["#0d366b", "#3987e5", "#9ec5f4", "#f3f2ee"])  # level 0 .. 3

LINE_W = 1.1
MARKER_S = 3.6


def apply():
    """Set the matplotlib rcParams of all paper figures."""
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{newtxtext}\usepackage{newtxmath}\usepackage{siunitx}",
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "axes.titlepad": 4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        "lines.linewidth": LINE_W,
        "lines.markersize": MARKER_S,
        "lines.markeredgewidth": 0.9,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "butt",
        "legend.frameon": False,
        "legend.handlelength": 2.4,
        "legend.borderaxespad": 0.3,
        "legend.labelspacing": 0.3,
        "legend.columnspacing": 1.2,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "image.cmap": "field",
    })
    mpl.colormaps.register(FIELD_CMAP, force=True)


def style(variant, markers=False, **kw):
    """Line keyword arguments of a method variant."""
    v = VARIANTS[variant]
    out = dict(color=v["color"], ls=v["ls"], label=v["label"], lw=LINE_W)
    if markers:
        out.update(marker=v["marker"], mfc="white", mec=v["color"], ms=MARKER_S)
    out.update(kw)
    return out


def reference(fine=False, **kw):
    """Line keyword arguments of an exact or fine-grid reference."""
    out = dict(color=REF_FINE, ls=(0, (1.2, 1.4)), lw=1.0) if fine else dict(color=REF, ls="-", lw=0.8)
    out.update(kw)
    return out


def panel_label(ax, letter, title=None, **_):
    """Panel letter '(a)', optionally followed by a short title, above the upper-left corner."""
    ax.set_title(f"({letter})" + (f" {title}" if title else ""), loc="left", fontsize=8, pad=3)


def figure(nrows=1, ncols=1, height=None, width=WIDTH, **kw):
    """Figure at the paper text width."""
    h = height if height is not None else PANEL_H * nrows
    kw.setdefault("constrained_layout", True)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, h), **kw)
    if kw["constrained_layout"]:
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.03)
    return fig, axes


def field_axes(ax):
    """Axes of a 2-D colour map: no grid, full frame, equal aspect."""
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.5)
    ax.set_aspect("equal")


def save(fig, stem, out="paper_figures"):
    """Save the figure as PDF (and PNG preview) in the output directory."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{stem}.pdf")
    fig.savefig(out / f"{stem}.png", dpi=200)
    plt.close(fig)


def cell_extent(x, y):
    """imshow extent of cell-centred data on the coordinates x, y."""
    return (x[0] - 0.5 * (x[1] - x[0]), x[-1] + 0.5 * (x[-1] - x[-2]), y[0], y[-1])


def activation_maps(fig, axes, extent, maps, titles, xlabel, aspect="auto"):
    """Activation maps sharing one legend (MOOD levels and active cells)."""
    for k, (ax, (key, arr), title) in enumerate(zip(axes, maps, titles)):
        if key == "mood":
            ax.imshow(arr, origin="lower", aspect=aspect, extent=extent, cmap=MOOD_CMAP,
                      vmin=-0.5, vmax=3.5, interpolation="nearest", rasterized=True)
        else:
            ax.imshow((arr > 0).astype(float), origin="lower", aspect=aspect, extent=extent,
                      cmap=BINARY_CMAP, vmin=0, vmax=1, interpolation="nearest", rasterized=True)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.5)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2, steps=[5, 10]))
        if k:
            ax.tick_params(labelleft=False)
        panel_label(ax, "abcd"[k], title)
    handles = [Patch(fc=MOOD_CMAP(l), ec=MUTED, lw=0.3, label=rf"MOOD $\ell={l}$") for l in range(4)]
    handles.append(Patch(fc=ACTIVE, ec=MUTED, lw=0.3, label="active"))
    fig.legend(handles=handles, loc="outside lower center", ncol=5, handlelength=1.2, handleheight=0.9)


def resolution_axis(ax, N, labelled=None):
    """Logarithmic grid-resolution axis with ticks at the computed grids."""
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(N))
    ax.xaxis.set_minor_locator(NullLocator())
    shown = set(N if labelled is None else labelled)
    ax.set_xticklabels([str(int(n)) if n in shown else "" for n in N])
