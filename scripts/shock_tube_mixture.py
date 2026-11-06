"""Nitrogen / n-dodecane shock tubes of Boyd & Jarrahbashi (2021), Section 8.3."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

import drop_euler_2d as de
import thermo_binary_pr as pr

CASES = {"p60": 60.0e6, "p7": 7.0e6}


def setup(case, n):
    """Initial states at 650 K (nitrogen left, n-dodecane right)."""
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) * dx
    left = x < 0.5
    T = np.full(n, 650.0)
    p = np.where(left, CASES[case], 6.0e6)
    Y = np.where(left, 0.0, 1.0)
    rho = pr.density_from_T_p_Y(T, p, Y)
    V = np.stack((rho, np.zeros(n), np.zeros(n), p, Y))[:, None, :]
    return x, dx, V


def run(case, method, n, t_final=5.0e-4, cfl=0.8, map_stride=5):
    """Run one shock tube and collect histories, maps, and metrics."""
    cfg = de.Config(method=method, n=n).normalized()
    x, dx, V = setup(case, n)
    _, _, _, _, E0, _ = de.project(V)
    mass0, fuel0 = V[0].sum(), (V[0] * V[4]).sum()
    energy0, energy0_abs = E0.sum(), np.abs(E0).sum()
    hist = dict(time=[0.0], energy_error=[0.0], mass_error=[0.0], fuel_mass_error=[0.0])
    maps = dict(time=[], mood=[], thinc=[], diss=[], dmp=[])
    failure = ""
    t, step, tic = 0.0, 0, time.perf_counter()
    while t < t_final * (1.0 - 1e-12):
        gam, e0, _, c, E, Rgas = de.project(V)
        dt = min(cfl * dx / np.max(np.abs(V[1]) + c), t_final - t)
        rho, u, v, p, Y = V
        Q = np.stack((rho, rho * u, rho * v, E, rho * Y))
        op = de.Sweep(cfg, "wall", dx, gam, e0, Rgas)
        try:
            Qn, stages = op.advance(Q, dt)
            W = op.primitives(Qn)
            Vt = np.stack((W[0], W[1], W[2], W[3], W[4]))
            de.project(Vt)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            print(f"[{case}/{method} N={n}] FAILED at t={t:.3e}: {failure}", flush=True)
            break
        V = Vt
        t += dt
        step += 1
        _, _, _, _, E, _ = de.project(V)
        hist["time"].append(t)
        hist["energy_error"].append(abs(E.sum() - energy0) / energy0_abs)
        hist["mass_error"].append(abs(V[0].sum() - mass0) / mass0)
        hist["fuel_mass_error"].append(abs((V[0] * V[4]).sum() - fuel0) / fuel0)
        if step % map_stride == 0 or t >= t_final * (1.0 - 1e-12):
            maps["time"].append(t)
            maps["mood"].append(np.minimum.reduce([d["cell_level"][0] for d in stages]))
            maps["thinc"].append(np.logical_or.reduce([d["thinc_cell"][0] for d in stages]))
            f = [d["diss_face"][0] for d in stages]
            maps["diss"].append(np.logical_or.reduce([a[:-1] | a[1:] for a in f]))
            f = [d["dmp_face"][0] for d in stages]
            maps["dmp"].append(np.logical_or.reduce([a[:-1] | a[1:] for a in f]))
    gam, e0, T, c, E, _ = de.project(V)
    rho, u, _, p, Y = (a[0] for a in V)
    metrics = dict(case=case, method=method, N=n, steps=step, t_final=t,
                   status="failed" if failure else "completed", failure=failure,
                   runtime_s=time.perf_counter() - tic,
                   energy_rel=hist["energy_error"][-1], mass_rel=hist["mass_error"][-1],
                   fuel_mass_rel=hist["fuel_mass_error"][-1],
                   Y_min=float(Y.min()), Y_max=float(Y.max()),
                   p_min=float(p.min()), p_max=float(p.max()),
                   T_min=float(T.min()), T_max=float(T.max()),
                   mood_min_level=int(min(m.min() for m in maps["mood"])) if maps["mood"] else -1,
                   max_thinc_fraction=float(max(m.mean() for m in maps["thinc"])) if maps["thinc"] else 0.0,
                   max_dmp_fraction=float(max(m.mean() for m in maps["dmp"])) if maps["dmp"] else 0.0)
    arrays = dict(x=x, rho=rho, u=u, p=p, Y=Y, T=T[0], gamma=gam[0], e0=e0[0],
                  history_time=np.array(hist["time"]),
                  history_energy_error=np.array(hist["energy_error"]),
                  history_mass_error=np.array(hist["mass_error"]),
                  map_time=np.array(maps["time"]),
                  mood_map=np.array(maps["mood"], dtype=np.int8),
                  thinc_map=np.array(maps["thinc"], dtype=np.int8),
                  dissipative_map=np.array(maps["diss"], dtype=np.int8),
                  dmp_map=np.array(maps["dmp"], dtype=np.int8))
    return dict(config=dict(asdict(cfg), case=case), metrics=metrics, arrays=arrays)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", choices=sorted(CASES), default="p60")
    ap.add_argument("--methods", nargs="+", default=["df_ho", "df_thinc", "full"])
    ap.add_argument("--grids", nargs="+", type=int, default=[200])
    ap.add_argument("--output", type=Path, default=Path("shock_tube_mixture_results"))
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for m in a.methods:
        for n in a.grids:
            res = run(a.case, m, n)
            stem = f"{a.case}_{m}_N{n}"
            np.savez_compressed(a.output / f"{stem}.npz", **res["arrays"])
            (a.output / f"{stem}.json").write_text(json.dumps(
                dict(config=res["config"], metrics=res["metrics"]), indent=2, default=str))
            print(json.dumps(res["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
