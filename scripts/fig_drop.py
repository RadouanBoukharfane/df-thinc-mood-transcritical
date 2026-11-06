"""Figures of the two-dimensional n-dodecane drop (Section 8.5, Figs. 14-18)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import paperstyle as ps

RES = Path("drop_euler_results")
N_CMP = 100
D = 0.5e-3
WINDOW = (-0.8, 0.8)
METHODS = ("df_ho", "df_thinc", "full")


def load(name):
    path = RES / f"{name}.npz"
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    meta = json.loads(path.with_suffix(".json").read_text())
    return dict(data=data, config=meta["config"], metrics=meta["metrics"])


def centred(case):
    """Coordinates normalized by D and centred on the final drop centre."""
    m = case["metrics"]
    return (case["data"]["x"] - m["final_center_x"]) / D, (case["data"]["y"] - m["final_center_y"]) / D


def bilinear(x, y, f, xq, yq):
    ix = np.clip(np.searchsorted(x, xq) - 1, 0, len(x) - 2)
    iy = np.clip(np.searchsorted(y, yq) - 1, 0, len(y) - 2)
    tx = (xq - x[ix]) / (x[ix + 1] - x[ix])
    ty = (yq - y[iy]) / (y[iy + 1] - y[iy])
    return ((1 - tx) * (1 - ty) * f[iy, ix] + tx * (1 - ty) * f[iy, ix + 1]
            + (1 - tx) * ty * f[iy + 1, ix] + tx * ty * f[iy + 1, ix + 1])


def map_axes(ax, k, title):
    ps.field_axes(ax)
    ax.set_xlim(*WINDOW)
    ax.set_ylim(*WINDOW)
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.set_xlabel(r"$(x-x_c)/D$")
    if k == 0:
        ax.set_ylabel(r"$(y-y_c)/D$")
    else:
        ax.tick_params(labelleft=False)
    ps.panel_label(ax, "abcdefgh"[k], title)


def density_maps(cases, titles, stem):
    fig, axes = ps.figure(1, len(cases), height=1.85)
    vmin = min(c["data"]["rho_final"].min() for c in cases)
    vmax = max(c["data"]["rho_final"].max() for c in cases)
    for k, (ax, c, title) in enumerate(zip(axes, cases, titles)):
        xs, ys = centred(c)
        im = ax.imshow(c["data"]["rho_final"], origin="lower", extent=ps.cell_extent(xs, ys), cmap=ps.FIELD_CMAP,
                       vmin=vmin, vmax=vmax, interpolation="nearest", rasterized=True)
        ex = c["data"]["rho_exact"]
        ax.contour(xs, ys, ex, levels=[0.5 * (ex.min() + ex.max())], colors=ps.INK, linewidths=0.6,
                   linestyles=[(0, (3, 2))])
        map_axes(ax, k, title)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, aspect=18)
    cb.set_label(r"$\rho$ [kg\,m$^{-3}$]")
    cb.outline.set_linewidth(0.5)
    ps.save(fig, stem)


def fig_equilibrium(cases):
    fig, axes = ps.figure(1, 3, height=1.65)
    for m in METHODS:
        d = cases[m]["data"]
        t = (d["time"] / cases[m]["metrics"]["period"])[1:]
        axes[0].semilogy(t, d["history_pressure_error"][1:], **ps.style(m, lw=0.8))
        axes[1].semilogy(t, np.maximum(d["history_u_error"], d["history_v_error"])[1:], **ps.style(m, lw=0.8))
        axes[2].plot(t, d["history_energy_error"][1:], **ps.style(m))
    axes[0].set_ylim(1e-13, 1e-9)
    axes[1].set_ylim(1e-12, 1e-8)
    axes[0].set_ylabel(r"$\max|p-p_0|/p_0$")
    axes[1].set_ylabel(r"$\max(|u-u_0|,|v-v_0|)$ [m/s]")
    axes[2].set_ylabel(r"Energy defect, $\varepsilon_E$")
    for ax, letter in zip(axes, "abc"):
        ax.set_xlim(0, 1)
        ax.set_xlabel("Advection periods")
        ps.panel_label(ax, letter)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
    ps.save(fig, "drop_equilibrium")


def fig_cut_refinement_conservation(cases, frame):
    fig, axes = ps.figure(2, 2, height=3.15)
    a, b, c, d = axes.flat
    for i, m in enumerate(METHODS):
        case = cases[m]
        x, y = case["data"]["x"], case["data"]["y"]
        cx, cy = case["metrics"]["final_center_x"], case["metrics"]["final_center_y"]
        lo, hi = max(x.min(), cx + y.min() - cy), min(x.max(), cx + y.max() - cy)
        xq = np.linspace(lo, hi, 1500)
        yq = cy + xq - cx
        s = np.sqrt(2.0) * (xq - cx) / D  # arc length along the diagonal
        if i == 0:
            a.plot(s, bilinear(x, y, case["data"]["rho_exact"], xq, yq), **ps.reference(label="Exact"))
        a.plot(s, bilinear(x, y, case["data"]["rho_final"], xq, yq), **ps.style(m))
    a.set_xlim(*WINDOW)
    a.set_xlabel(r"Diagonal coordinate, $s/D$")
    a.set_ylabel(r"$\rho$ [kg\,m$^{-3}$]")


    full = cases["full"]["data"]
    t = full["time"] / cases["full"]["metrics"]["period"]
    b.semilogy(t[1:], np.abs(full["history_mass_error"][1:]), color=ps.BLUE, label="Total mass")
    b.semilogy(t[1:], np.abs(full["history_fuel_mass_error"][1:]), color=ps.INK, ls=(0, (4, 1.8)), label="Fuel mass")
    b.set_xlim(0, 1)
    b.set_ylim(1e-15, 1e-11)
    b.set_xlabel("Advection periods")
    b.set_ylabel("Relative conservation error")
    b.legend(loc="lower right", fontsize=6.5)

    frame = frame.sort_values("N")
    N = frame["N"].to_numpy(float)
    for key, label, kw in (("rho_L1", r"$L_1(\rho)$ [kg\,m$^{-3}$]", dict(color=ps.BLUE, marker="o")),
                           ("Y_L1", r"$L_1(Y)$", dict(color=ps.INK, ls=(0, (4, 1.8)), marker="^"))):
        e = frame[key].to_numpy(float)
        r = -np.polyfit(np.log(N), np.log(e), 1)[0]
        c.loglog(N, e, mfc="white", label=rf"{label}, $r={r:.2f}$", **kw)
    c.set_ylim(5e-3, 80)
    c.set_ylabel(r"$L_1$ error")
    c.legend(loc="center left", bbox_to_anchor=(0.0, 0.42), fontsize=6.5)

    d.plot(N, frame["mixed_width_cells"], color=ps.BLUE, marker="o", mfc="white")
    d.set_ylim(0, 9)
    d.set_ylabel(r"Mixed width, $\delta/\Delta x$")
    for ax in (c, d):
        ps.resolution_axis(ax, N)
        ax.set_xlabel(r"Grid resolution, $N$")
    for ax, letter in zip(axes.flat, "abcd"):
        ps.panel_label(ax, letter)
    fig.legend(*a.get_legend_handles_labels(), loc="outside upper center", ncol=4)
    ps.save(fig, "drop_cut_refinement_conservation")


def fig_activation(case):
    fig, axes = ps.figure(1, 4, height=1.75)
    d = case["data"]
    xs, ys = centred(case)
    maps = [("mood", d["mood_map"]), ("thinc", d["thinc_map"]), ("riemann", d["dissipative_map"]), ("dmp", d["dmp_map"])]
    ps.activation_maps(fig, axes, ps.cell_extent(xs, ys), maps,
                       ["MOOD level", "THINC-BVD", "Dissipative cells", "DMP/FCT"], r"$(x-x_c)/D$", aspect="equal")
    for k, ax in enumerate(axes):
        ax.set_xlim(*WINDOW)
        ax.set_ylim(*WINDOW)
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_yticks([-0.5, 0, 0.5])
    axes[0].set_ylabel(r"$(y-y_c)/D$")
    ps.save(fig, "drop_activation_maps")


if __name__ == "__main__":
    ps.apply()
    cases = {m: load(f"drop_{m}_N{N_CMP}") for m in METHODS}
    names = [ps.VARIANTS[m]["label"] for m in METHODS]
    density_maps([cases[m] for m in METHODS], names, "drop_density_contours")
    fig_equilibrium(cases)
    fig_cut_refinement_conservation(cases, pd.read_csv(RES / "refinement_metrics.csv"))
    fig_activation(cases["full"])
    obl = [load(f"drop_full_N{N_CMP}_oblique"), load(f"drop_df_ho_N{N_CMP}_oblique")]
    density_maps([cases["full"]] + obl, ["Complete, horizontal", "Complete, oblique", "DF + HO, oblique"],
                 "drop_oblique_density")
