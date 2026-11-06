"""Temporal nitrogen / n-dodecane mixing layer (unsplit solver)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import time

import numpy as np

import drop_euler_2d as de
import drop_euler_2d_unsplit as du
import thermo_binary_pr as pr


def setup(n, U=100.0, delta=50.0e-6, eps=0.05, p0=6.0e6, L=1.0e-3):
    """Initial shear layer with a localized transverse perturbation."""
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    y = -L / 2 + (np.arange(n) + 0.5) * dx
    X, Yc = np.meshgrid(x, y)
    fuel = (Yc < 0.0).astype(float)
    T = np.where(fuel > 0.5, 363.0, 900.0)
    rho = pr.density_from_T_p_Y(T, p0, fuel)
    u = U * np.tanh(Yc / delta)
    v = eps * U * np.sin(2 * np.pi * X / L) * np.exp(-(Yc / delta) ** 2)
    V = np.stack((rho, u, v, np.full_like(rho, p0), fuel))
    return dict(V=V, x=x, y=y, dx=dx, L=L, U=U, delta=delta, p0=p0)


def momentum_thickness(V, y, U):
    """Momentum thickness of the mean profile."""
    rho, u = V[0].mean(axis=1), V[1].mean(axis=1)
    rho_m = 0.5 * (rho.max() + rho.min())
    w = (U - u) * (U + u) / (4 * U**2)
    return float(np.sum(rho / rho_m * w) * (y[1] - y[0]))


def run(method, n, t_final, snapshots):
    """Run the mixing layer and store snapshots and histories."""
    cfg = de.Config(method=method, n=n).normalized()
    s = setup(n)
    V, dx = s["V"], s["dx"]
    bcs = ("periodic", "wall")
    _, _, _, _, E0, _ = de.project(V)
    mass0, fuel0, E0s, E0a = V[0].sum(), (V[0] * V[4]).sum(), E0.sum(), np.abs(E0).sum()
    hist = {k: [] for k in ("time", "mass_error", "fuel_mass_error", "energy_error", "theta",
                            "Y_min", "Y_max", "p_min", "p_max", "mood_min", "thinc_frac", "dmp_frac",
                            "mood_demoted_frac", "fallback")}
    snaps, snap_times = [], sorted(snapshots)
    t, step, tic = 0.0, 0, time.perf_counter()

    def record(stages):
        _, _, T, c, E, _ = de.project(V)
        hist["time"].append(t)
        hist["mass_error"].append(abs(V[0].sum() - mass0) / mass0)
        hist["fuel_mass_error"].append(abs((V[0] * V[4]).sum() - fuel0) / fuel0)
        hist["energy_error"].append(abs(E.sum() - E0s) / E0a)
        hist["theta"].append(momentum_thickness(V, s["y"], s["U"]))
        hist["Y_min"].append(float(V[4].min()))
        hist["Y_max"].append(float(V[4].max()))
        hist["p_min"].append(float(V[3].min()))
        hist["p_max"].append(float(V[3].max()))
        if stages:
            lev = np.minimum.reduce([d["cell_level"] for d in stages])
            top = 3 if cfg.use_thinc else 2
            hist["mood_min"].append(int(lev.min()))
            hist["mood_demoted_frac"].append(float(np.mean(lev < top)))
            hist["thinc_frac"].append(float(np.mean(np.logical_or.reduce([d["thinc"] for d in stages]))))
            hist["dmp_frac"].append(float(np.mean(np.logical_or.reduce([d["dmp"] for d in stages]))))
            hist["fallback"].append(int(sum(d["fallback"] for d in stages)))
        else:
            for k, v0 in (("mood_min", 3), ("mood_demoted_frac", 0.0), ("thinc_frac", 0.0),
                          ("dmp_frac", 0.0), ("fallback", 0)):
                hist[k].append(v0)
        return T

    T = record(None)
    snaps.append(dict(t=0.0, rho=V[0].copy(), Y=V[4].copy(), T=T.copy(), p=V[3].copy(),
                      u=V[1].copy(), v=V[2].copy()))
    while t < t_final * (1.0 - 1e-12):
        gam, e0, _, c, E, Rgas = de.project(V)
        dt = cfg.cfl / np.max((np.abs(V[1]) + c) / dx + (np.abs(V[2]) + c) / dx)
        nxt = next((ts for ts in snap_times if ts > t * (1 + 1e-12)), t_final)
        dt = min(dt, nxt - t, t_final - t)
        Q = np.stack((V[0], V[0] * V[1], V[0] * V[2], E, V[0] * V[4]))
        op = du.Unsplit(cfg, bcs, dx, gam, e0, Rgas)
        Qn, stages = op.advance(Q, dt)
        W = op.primitives(Qn)
        V = np.stack((W[0], W[1], W[2], W[3], W[4]))
        t += dt
        step += 1
        if step % 20 == 0 or any(abs(t - ts) < 1e-15 for ts in snap_times) or t >= t_final * (1 - 1e-12):
            T = record(stages)
            if any(abs(t - ts) < 1e-15 for ts in snap_times):
                snaps.append(dict(t=t, rho=V[0].copy(), Y=V[4].copy(), T=T.copy(), p=V[3].copy(),
                                  u=V[1].copy(), v=V[2].copy()))
        if step % 500 == 0:
            print(f"[mixing layer {method} N={n}] step {step} t={t:.3e} theta={hist['theta'][-1]:.3e} "
                  f"eps_E={hist['energy_error'][-1]:.3e} Y=[{hist['Y_min'][-1]:.1e},{hist['Y_max'][-1] - 1:.1e}] "
                  f"({time.perf_counter() - tic:.0f}s)", flush=True)
    metrics = dict(method=method, N=n, steps=step, t_final=t, runtime_s=time.perf_counter() - tic,
                   mass_rel=hist["mass_error"][-1], fuel_mass_rel=hist["fuel_mass_error"][-1],
                   energy_rel=hist["energy_error"][-1], theta_final=hist["theta"][-1],
                   Y_min=float(np.min(hist["Y_min"])), Y_max=float(np.max(hist["Y_max"])),
                   p_min=float(np.min(hist["p_min"])), p_max=float(np.max(hist["p_max"])),
                   mood_min_level=int(np.min(hist["mood_min"])),
                   max_thinc_fraction=float(np.max(hist["thinc_frac"])),
                   max_dmp_fraction=float(np.max(hist["dmp_frac"])),
                   max_mood_demoted_fraction=float(np.max(hist["mood_demoted_frac"])),
                   total_fct_fallback=int(np.sum(hist["fallback"])))
    arrays = dict(x=s["x"], y=s["y"], snap_times=np.array([sn["t"] for sn in snaps]),
                  **{f"snap_{k}": np.array([sn[k] for sn in snaps]) for k in ("rho", "Y", "T", "p", "u", "v")},
                  **{f"history_{k}": np.asarray(v) for k, v in hist.items()})
    return metrics, arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", default=["full"])
    ap.add_argument("--grids", nargs="+", type=int, default=[128])
    ap.add_argument("--t-final", type=float, default=5.0e-5)
    ap.add_argument("--output", type=Path, default=Path("mixing_layer_results"))
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    snaps = [1.0e-5, 2.0e-5, 3.0e-5, 4.0e-5, 5.0e-5]
    for m in a.methods:
        for n in a.grids:
            try:
                metrics, arrays = run(m, n, a.t_final, [t for t in snaps if t <= a.t_final])
                metrics["status"] = "completed"
            except RuntimeError as exc:
                print("FAILED:", exc, flush=True)
                metrics, arrays = dict(method=m, N=n, status="failed", failure=str(exc)), {}
            stem = f"ml_{m}_N{n}"
            if arrays:
                np.savez_compressed(a.output / f"{stem}.npz", **arrays)
            (a.output / f"{stem}.json").write_text(json.dumps(metrics, indent=2, default=str))
            print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
