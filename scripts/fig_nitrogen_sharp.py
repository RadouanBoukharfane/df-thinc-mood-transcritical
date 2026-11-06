"""Figures of the discontinuous nitrogen-interface advection (Section 8.1, Figs. 2-5)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

import paperstyle as ps  # noqa: E402
from thermo import get_T_from_p_rho  # noqa: E402

DATA = Path("nitrogen_results/sharp")
METHODS = ("df_ho", "df_thinc", "full")


def load(method):
    path = DATA / f"sharp_{method}_N150.npz"
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    meta = json.loads(path.with_suffix(".json").read_text())
    return dict(data=data, config=meta["config"], metrics=meta["metrics"])


def periods(case):
    return case["data"]["time"] / float(case["metrics"]["period"])


def top_legend(fig, handles=None, labels=None, ncol=4):
    fig.legend(handles, labels, loc="outside upper center", ncol=ncol, handlelength=2.6)


def fig_profiles(cases):
    fig, (a, b) = ps.figure(1, 2, height=1.85)
    ref = cases["df_ho"]["data"]
    for ax, key, exact in ((a, "rho", "rho_exact"), (b, "T", "T_exact")):
        ax.plot(ref["x"], ref[exact], **ps.reference(label="Exact"))
        zoom = ax.inset_axes([0.52, 0.30, 0.42, 0.42]) if key == "rho" else ax.inset_axes([0.52, 0.45, 0.42, 0.42])
        zoom.plot(ref["x"], ref[exact], **ps.reference())
        for m in METHODS:
            d = cases[m]["data"]
            y = d["V"][0] if key == "rho" else get_T_from_p_rho(d["V"][2], d["V"][0])
            ax.plot(d["x"], y, **ps.style(m))
            zoom.plot(d["x"], y, **ps.style(m, label=None, lw=0.9))
        zoom.set_xlim(0.19, 0.31)
        lo, hi = ax.get_ylim()
        zoom.set_ylim(lo, hi)
        zoom.tick_params(labelsize=5.5, length=1.8, pad=1)
        zoom.grid(False)
        for s in zoom.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.5)
            s.set_color(ps.MUTED)
        ax.add_patch(Rectangle((0.19, lo), 0.12, hi - lo, fill=False, ec=ps.MUTED, lw=0.5, ls=(0, (2, 1.5))))
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$x$ [m]")
    a.set_ylabel(r"$\rho$ [kg\,m$^{-3}$]")
    b.set_ylabel(r"$T$ [K]")
    ps.panel_label(a, "a")
    ps.panel_label(b, "b")
    top_legend(fig, *a.get_legend_handles_labels())
    ps.save(fig, "sharp_rho_temperature")


def fig_equilibrium(cases):
    fig, (a, b) = ps.figure(1, 2, height=1.65)
    for m in METHODS:
        c = cases[m]
        d = c["data"]
        a.plot(d["x"], (d["V"][1] - c["config"]["velocity"]) * 1e10, **ps.style(m, lw=0.9))
        b.plot(d["x"], (d["V"][2] - c["config"]["pressure"]) / c["config"]["pressure"] * 1e11, **ps.style(m, lw=0.9))
    for ax in (a, b):
        ax.axhline(0, color=ps.MUTED, lw=0.5, zorder=0)
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$x$ [m]")
    a.set_ylabel(r"$(u-u_0)\times10^{10}$ [m\,s$^{-1}$]")
    b.set_ylabel(r"$(p-p_0)/p_0\times10^{11}$")
    ps.panel_label(a, "a")
    ps.panel_label(b, "b")
    top_legend(fig, *a.get_legend_handles_labels(), ncol=3)
    ps.save(fig, "sharp_velocity_pressure_errors")


def fig_energy_width(cases):
    fig, (a, b) = ps.figure(1, 2, height=1.65)
    for m in METHODS:
        c = cases[m]
        t = periods(c)
        dx = c["config"]["length"] / c["config"]["nx"]
        a.semilogy(t[1:], c["data"]["energy_error"][1:], **ps.style(m))
        b.plot(t, c["data"]["width_gradient"] / dx, **ps.style(m, lw=0.8))
    a.set_ylim(1e-4, 3e-2)
    a.set_ylabel(r"Energy defect, $\varepsilon_E$")
    b.set_ylabel(r"Interface thickness, $\delta_\rho/\Delta x$")
    for ax, letter in ((a, "a"), (b, "b")):
        ax.set_xlim(0, 1)
        ax.set_xlabel("Advection periods")
        ps.panel_label(ax, letter)
    top_legend(fig, *a.get_legend_handles_labels(), ncol=3)
    ps.save(fig, "sharp_energy_width")


def fig_activation(case):
    fig, axes = ps.figure(1, 4, height=1.85, sharey=True)
    d = case["data"]
    t = periods(case)
    maps = [("mood", d["mood_map"]), ("thinc", d["thinc_map"]), ("riemann", d["riemann_map"]), ("dmp", d["dmp_map"])]
    ps.activation_maps(fig, axes, ps.cell_extent(d["x"], t), maps, ["MOOD level", "THINC-BVD", "Dissipative faces", "DMP/FCT"], r"$x$ [m]")
    axes[0].set_ylabel("Advection periods")
    ps.save(fig, "sharp_activation_maps")


if __name__ == "__main__":
    ps.apply()
    cases = {m: load(m) for m in METHODS}
    fig_profiles(cases)
    fig_equilibrium(cases)
    fig_energy_width(cases)
    fig_activation(cases["full"])
