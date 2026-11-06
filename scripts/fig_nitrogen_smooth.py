"""Figures of the smooth nitrogen-profile refinement (Section 8.1, Figs. 6-7)."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import paperstyle as ps

CLIPPED = Path("nitrogen_results/refinement_smooth")
UNCLIPPED = Path("nitrogen_results/refinement_smooth_unclipped")
FIT_MIN_N = 100
NORMS = [("L1_rho", r"$L_1$ error of $\rho$"), ("L2_rho", r"$L_2$ error of $\rho$"),
         ("Linf_rho", r"$L_\infty$ error of $\rho$"), ("L1_T", r"$L_1$ error of $T$")]


def load(directory):
    rows = []
    for p in directory.glob("*_N*.json"):
        m = json.loads(p.read_text())
        row = dict(N=int(re.search(r"_N(\d+)", p.stem).group(1)))
        row.update(m["metrics"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)


def fit(N, e):
    ok = N >= FIT_MIN_N
    c = np.polyfit(np.log(N[ok]), np.log(e[ok]), 1)
    return -c[0], N[ok], np.exp(c[1]) * N[ok] ** c[0]


def fig_convergence(clipped, unclipped):
    fig, axes = ps.figure(2, 2, height=2.9, sharex=True)
    series = [(unclipped, dict(color=ps.BLUE, ls="-", marker="o", mfc=ps.BLUE, mec=ps.BLUE)),
              (clipped, dict(color=ps.MUTED, ls=(0, (4, 1.8)), marker="s", mfc="white", mec=ps.MUTED))]
    for k, (ax, (key, ylabel)) in enumerate(zip(axes.flat, NORMS)):
        for frame, st in series:
            N = frame["N"].to_numpy(float)
            e = frame[key].to_numpy(float)
            ax.loglog(N, e, **st)
            r, Nf, ef = fit(N, e)
            ax.loglog(Nf, ef * 0.45, color=ps.INK, lw=0.6, ls="-")
            ax.text(np.sqrt(Nf[0] * Nf[-1]), np.sqrt(ef[0] * ef[-1]) * 0.13, rf"$r={r:.2f}$",
                    ha="center", va="top", fontsize=7, color=st["color"] if st["color"] != ps.MUTED else ps.INK)
        ps.resolution_axis(ax, N)
        ax.set_ylabel(ylabel)
        ps.panel_label(ax, "abcd"[k], x=0.88)
        if k >= 2:
            ax.set_xlabel(r"Grid resolution, $N$")
    handles = [Line2D([], [], **series[0][1], label="Unclipped reconstruction"),
               Line2D([], [], **series[1][1], label="Clipped reconstruction"),
               Line2D([], [], color=ps.INK, lw=0.6, label=rf"Least-squares fit, $N\geq{FIT_MIN_N}$ (offset)")]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    ps.save(fig, "smooth_accuracy_convergence")


def fig_robustness(frame):
    fig, (a, b, c) = ps.figure(1, 3, height=1.6)
    N = frame["N"].to_numpy(float)
    a.semilogy(N, frame["pressure_error_final"], color=ps.BLUE, marker="o", mfc="white", label=r"$\|p-p_0\|_\infty/p_0$")
    a.semilogy(N, frame["velocity_error_final"], color=ps.INK, ls=(0, (4, 1.8)), marker="s", mfc="white",
               label=r"$\|u-u_0\|_\infty$ [m\,s$^{-1}$]")
    a.set_ylabel("Equilibrium error")
    b.semilogy(N, frame["energy_error_final"], color=ps.BLUE, marker="o", mfc="white")
    b.set_ylabel(r"Energy defect, $\varepsilon_E(t_f)$")
    for key, label, color, ls, mk in (("max_thinc_fraction", "THINC-BVD", ps.GREEN, "-", "^"),
                                      ("max_riemann_fraction", "Dissipative faces", ps.BLUE, (0, (4, 1.8)), "o"),
                                      ("max_dmp_fraction", "DMP/FCT", ps.ORANGE, (0, (1.2, 1.4)), "s")):
        c.plot(N, frame[key], color=color, ls=ls, marker=mk, mfc="white", label=label)
    c.set_ylabel("Max. active fraction")
    for ax, letter in ((a, "a"), (b, "b"), (c, "c")):
        ps.resolution_axis(ax, N)
        ax.set_xlabel(r"$N$")
        ps.panel_label(ax, letter, x=0.86)
    a.set_ylim(top=3e-8)
    a.legend(loc="upper left", fontsize=6.5, handlelength=2.0)
    c.set_ylim(-0.01, 0.2)
    c.legend(loc="upper right", fontsize=6.5, handlelength=2.0)
    ps.save(fig, "smooth_robustness")


if __name__ == "__main__":
    ps.apply()
    clipped, unclipped = load(CLIPPED), load(UNCLIPPED)
    fig_convergence(clipped, unclipped)
    fig_robustness(unclipped)
