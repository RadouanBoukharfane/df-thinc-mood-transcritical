"""Figures of the transcritical nitrogen shock tube (Section 8.2, Figs. 8-10)."""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

import paperstyle as ps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import exact_riemann_nitrogen as exact  # noqa: E402

T_FINAL = 5.0e-4

CASES = Path("shock_tube_extended_results")
REFINEMENT = Path("shock_tube_refinement_results")
METHODS = ("df_ho", "df_thinc", "full")


def load(path):
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    return dict(data=data, config=meta["config"], metrics=meta.get("metrics", {}))


def exact_case(sol, n=2000):
    """Exact PR solution, in the same layout as a computed case."""
    x = (np.arange(n) + 0.5) / n
    e = exact.profile(sol, x, T_FINAL)
    return dict(data=dict(x=x, V=np.stack((e["rho"], e["u"], e["p"])), T=e["T"]))


def wave_crossings(case, sol, contact=(0.48, 0.64), shock=(0.64, 0.82)):
    """Sub-cell positions where density (contact) and pressure (shock) cross their mid values."""
    x, V = case["data"]["x"], case["data"]["V"]

    def crossing(q, window, lo, hi):
        m = (x >= window[0]) & (x <= window[1])
        xx, qq, mid = x[m], q[m], 0.5 * (lo + hi)
        k = np.flatnonzero((qq[:-1] - mid) * (qq[1:] - mid) <= 0)[0]
        return xx[k] + (mid - qq[k]) * (xx[k + 1] - xx[k]) / (qq[k + 1] - qq[k])

    return (crossing(V[0], contact, sol["rhoLs"], sol["rhoRs"]),
            crossing(V[2], shock, sol["pstar"], sol["pR"]))


def fig_profiles(cases, ref):
    fig, axes = ps.figure(2, 2, height=2.75, sharex=True)
    fields = [(lambda d: d["V"][0], r"$\rho$ [kg\,m$^{-3}$]"), (lambda d: d["V"][1], r"$u$ [m\,s$^{-1}$]"),
              (lambda d: d["V"][2] / 1e6, r"$p$ [MPa]"), (lambda d: d["T"], r"$T$ [K]")]
    for k, (ax, (get, ylabel)) in enumerate(zip(axes.flat, fields)):
        ax.plot(ref["data"]["x"], get(ref["data"]), **ps.reference(label="Exact"))
        for m in METHODS:
            ax.plot(cases[m]["data"]["x"], get(cases[m]["data"]), **ps.style(m, lw=0.95))
        ax.set_xlim(0, 1)
        ax.set_ylabel(ylabel)
        if k >= 2:
            ax.set_xlabel(r"$x$ [m]")
        ps.panel_label(ax, "abcd"[k], x=0.88)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=4, handlelength=2.6)
    ps.save(fig, "shock_tube_profiles")


def fig_activation(case):
    fig, axes = ps.figure(1, 3, height=1.85, sharey=True)
    d = case["data"]
    t = d["map_time"] / float(case["config"]["t_final"])
    maps = [("mood", d["mood_map"]), ("thinc", d["thinc_map"]), ("riemann", d["dissipative_map"][:, :-1])]
    ps.activation_maps(fig, axes, ps.cell_extent(d["x"], t), maps, ["MOOD level", "THINC-BVD", "Dissipative faces"], r"$x$ [m]")
    axes[0].set_ylabel(r"$t/t_f$")
    ps.save(fig, "shock_tube_activation_maps")


def fig_energy_refinement(runs, ref):
    fig, (a, b, c) = ps.figure(1, 3, height=1.75)
    N = np.array([r["config"]["nx"] for r in runs], float)
    colors = ps.ordinal(len(runs))
    for r, col in zip(runs, colors):
        t = r["data"]["history_time"] / float(r["config"]["t_final"])
        a.semilogy(t[1:], r["data"]["history_energy_error"][1:], color=col, label=str(r["config"]["nx"]))
    a.set_xlim(0, 1)
    a.set_ylim(3e-5, 2e-2)
    a.set_xlabel(r"$t/t_f$")
    a.set_ylabel(r"Energy defect, $\varepsilon_E$")
    a.legend(loc="lower right", ncol=2, fontsize=6.5, handlelength=1.2, columnspacing=0.8, title=r"$N$",
             title_fontsize=6.5)

    dx = 1.0 / N
    xc0, xs0 = 0.5 + ref["ustar"] * T_FINAL, 0.5 + ref["shock_speed"] * T_FINAL
    pos = np.array([wave_crossings(r, ref) for r in runs])
    b.loglog(N, np.abs(pos[:, 1] - xs0), color=ps.BLUE, marker="o", mfc="white", label="Shock")
    b.loglog(N, np.abs(pos[:, 0] - xc0), color=ps.INK, ls=(0, (4, 1.8)), marker="^", mfc="white", label="Contact")
    b.loglog(N, np.abs(pos[-1, 1] - xs0) * dx / dx[-1], color=ps.MUTED, lw=0.7, ls=(0, (1.2, 1.4)),
             label="First order")
    b.set_ylabel("Wave-position error [m]")
    b.set_ylim(top=0.1)
    b.legend(loc="upper right", fontsize=6.5, handlelength=2.0)

    eps = np.array([r["data"]["history_energy_error"][-1] for r in runs])
    c.loglog(N, eps * 1e3, color=ps.BLUE, marker="o", mfc="white")
    c.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4]))
    c.yaxis.set_minor_locator(NullLocator())
    c.yaxis.set_major_formatter(ScalarFormatter())
    order = -np.polyfit(np.log(N), np.log(eps), 1)[0]
    c.text(0.95, 0.9, rf"$r={order:.2f}$", transform=c.transAxes, ha="right", va="top", fontsize=7)
    c.set_ylabel(r"$\varepsilon_E(t_f)\times10^{3}$")
    for ax in (b, c):
        ps.resolution_axis(ax, N)
        ax.set_xlabel(r"Grid resolution, $N$")
    for ax, letter in ((a, "a"), (b, "b"), (c, "c")):
        ps.panel_label(ax, letter, x=0.84)
    ps.save(fig, "shock_tube_energy_refinement")


if __name__ == "__main__":
    ps.apply()
    cases = {m: load(CASES / f"shock_tube_{m}_N512.npz") for m in METHODS}
    sol = exact.solve()
    runs = sorted((load(p) for p in REFINEMENT.glob("shock_tube_full_N*.npz")), key=lambda r: r["config"]["nx"])
    runs.append(load(CASES / "shock_tube_reference_N4096.npz"))
    fig_profiles(cases, exact_case(sol))
    fig_activation(cases["full"])
    fig_energy_refinement(runs, sol)
