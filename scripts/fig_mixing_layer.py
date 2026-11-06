"""Figure of the temporal nitrogen / n-dodecane mixing layer (Section 8.6, Fig. 19)."""
from pathlib import Path

import numpy as np

import paperstyle as ps

RES = Path("mixing_layer_results")
VMIN, VMAX = 22.0, 642.0


def load(method, n):
    with np.load(RES / f"ml_{method}_N{n}.npz") as z:
        return {k: z[k] for k in z.files}


def panel(ax, d, k, title, letter, ylabel):
    x, y = d["x"] * 1e3, d["y"] * 1e3
    im = ax.imshow(d["snap_rho"][k], origin="lower", extent=ps.cell_extent(x, y), cmap=ps.FIELD_CMAP,
                   vmin=VMIN, vmax=VMAX, interpolation="bilinear", rasterized=True)
    ax.contour(x, y, d["snap_Y"][k], levels=[0.5], colors=ps.ORANGE, linewidths=0.35)
    ps.field_axes(ax)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_xlabel(r"$x$ [mm]")
    if ylabel:
        ax.set_ylabel(r"$y$ [mm]")
    else:
        ax.tick_params(labelleft=False)
    ps.panel_label(ax, letter, title)
    return im


if __name__ == "__main__":
    ps.apply()
    fig = ps.plt.figure(figsize=(ps.WIDTH, 3.3), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.05)
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.0, 1.45])
    d = load("full", 192)
    axes = top.subplots(1, 5)
    for i, (ax, k) in enumerate(zip(axes, range(1, 6))):
        im = panel(ax, d, k, rf"$t={d['snap_times'][k] * 1e6:.0f}\,\mu$s", "abcde"[i], i == 0)
    top.suptitle(r"Complete method, $192^2$", fontsize=8)
    axes = bottom.subplots(1, 2)
    for i, (ax, m) in enumerate(zip(axes, ("df_ho", "full"))):
        panel(ax, load(m, 128), -1, ps.VARIANTS[m]["label"], "fg"[i], i == 0)
    bottom.suptitle(r"$t=50\,\mu$s, $128^2$", fontsize=8)
    cb = bottom.colorbar(im, ax=axes, shrink=0.85, pad=0.03, aspect=16)
    cb.set_label(r"$\rho$ [kg\,m$^{-3}$]")
    cb.outline.set_linewidth(0.5)
    ps.save(fig, "mixing_layer")
