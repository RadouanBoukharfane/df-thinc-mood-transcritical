"""Figures of the nitrogen / n-dodecane shock tubes (Section 8.3, Figs. 11-12)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

import exact_riemann_mixture as ex  # noqa: E402
import paperstyle as ps  # noqa: E402

RES = Path("shock_tube_mixture_results")
T_FINAL = 5.0e-4
CASES = {"p60": 60.0e6, "p7": 7.0e6}
TITLES = {"p60": r"N$_2$ at 60 MPa / $n$-C$_{12}$H$_{26}$ at 6 MPa",
          "p7": r"N$_2$ at 7 MPa / $n$-C$_{12}$H$_{26}$ at 6 MPa"}
METHODS = ("df_ho", "df_thinc", "full")
GRIDS = (200, 400, 800, 1600, 6400)


def load(name):
    with np.load(RES / f"{name}.npz") as z:
        data = {k: z[k] for k in z.files}
    return data, json.loads((RES / f"{name}.json").read_text())["metrics"]


def fig_profiles(sols):
    fig = ps.plt.figure(figsize=(ps.WIDTH, 3.45), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.06)
    rows = fig.subfigures(2, 1)
    fields = [("rho", r"$\rho$ [kg\,m$^{-3}$]", 1.0), ("u", r"$u$ [m\,s$^{-1}$]", 1.0),
              ("p", r"$p$ [MPa]", 1e-6), ("T", r"$T$ [K]", 1.0)]
    xf = np.linspace(0.0, 1.0, 4001)
    letters = iter("abcdefgh")
    for row, case in zip(rows, CASES):
        row.suptitle(TITLES[case], fontsize=8)
        axes = row.subplots(1, 4)
        exact = ex.profile(sols[case], xf, T_FINAL)
        ref, _ = load(f"{case}_full_N6400")
        runs = {m: load(f"{case}_{m}_N200")[0] for m in METHODS}
        for ax, (k, label, sc) in zip(axes, fields):
            ax.plot(ref["x"], ref[k] * sc, **ps.reference(fine=True, label=r"Complete, $N=6400$"))
            ax.plot(xf, exact[k] * sc, **ps.reference(label="Exact"))
            for m in METHODS:
                ax.plot(runs[m]["x"], runs[m][k] * sc, **ps.style(m, lw=0.95))
            vals = np.concatenate([exact[k]] + [runs[m][k] for m in METHODS]) * sc
            pad = 0.08 * (vals.max() - vals.min())
            ax.set_ylim(vals.min() - pad, vals.max() + pad)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 0.5, 1])
            ax.set_xlabel(r"$x$ [m]")
            ax.set_ylabel(label, labelpad=1)
            ps.panel_label(ax, next(letters))
    handles, labels = axes[0].get_legend_handles_labels()
    order = [1, 0, 2, 3, 4]
    fig.legend([handles[i] for i in order], [labels[i].replace("Complete method", "Complete method") for i in order],
               loc="outside upper center", ncol=5, handlelength=2.2, columnspacing=0.9)
    ps.save(fig, "mixture_shock_tube_profiles")


def fig_refinement(sols):
    fig, axes = ps.figure(1, 3, height=1.7)
    for case, kw in (("p7", dict(color=ps.BLUE, marker="o", mfc="white", label=r"N$_2$ at 7 MPa")),
                     ("p60", dict(color=ps.INK, ls=(0, (4, 1.8)), marker="s", mfc="white", label=r"N$_2$ at 60 MPa"))):
        l1, serr, eps = [], [], []
        for n in GRIDS:
            d, m = load(f"{case}_full_N{n}")
            e = ex.profile(sols[case], d["x"], T_FINAL)
            l1.append(np.mean(np.abs(d["rho"] - e["rho"])))
            win = (0.7, 0.95) if case == "p60" else (0.6, 0.75)
            sel = (d["x"] > win[0]) & (d["x"] < win[1])
            g = np.abs(np.gradient(d["p"], d["x"]))
            xs = d["x"][np.flatnonzero(sel)[np.argmax(g[sel])]]
            serr.append(abs(xs - (0.5 + sols[case]["shock_speed"] * T_FINAL)))
            eps.append(m["energy_rel"])
        axes[0].loglog(GRIDS, l1, **kw)
        axes[1].loglog(GRIDS, np.maximum(serr, 1e-5), **kw)
        axes[2].loglog(GRIDS, eps, **kw)
    for ax, ylabel, letter in zip(axes, (r"$L_1$ error of $\rho$ [kg\,m$^{-3}$]", "Shock-position error [m]",
                                         r"$\varepsilon_E(t_f)$"), "abc"):
        ps.resolution_axis(ax, np.array(GRIDS, float), labelled=(200, 800, 6400))
        ax.set_xlabel(r"Grid resolution, $N$")
        ax.set_ylabel(ylabel)
        ps.panel_label(ax, letter)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=2)
    ps.save(fig, "mixture_shock_tube_refinement")


if __name__ == "__main__":
    ps.apply()
    sols = {c: ex.solve(p) for c, p in CASES.items()}
    fig_profiles(sols)
    fig_refinement(sols)
