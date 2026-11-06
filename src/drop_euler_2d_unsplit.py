"""Unsplit variant of the two-dimensional drop solver."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

import drop_euler_2d as de


def _swap(Q):
    return np.stack((Q[0].T, Q[2].T, Q[1].T, Q[3].T, Q[4].T))


def _increment(F, lam):
    d = F[..., 1:] - F[..., :-1]
    return -lam * np.stack((d[0], d[1], d[2], F[3][..., 1:] - F[4][..., :-1], d[5]))


def _from_y(inc):
    return np.stack((inc[0].T, inc[2].T, inc[1].T, inc[3].T, inc[4].T))


class Unsplit:
    """Unsplit operator: x and y residuals added at every stage."""
    def __init__(self, cfg, bcs, dx, gam, e0, Rgas):
        self.cfg, self.dx = cfg, dx
        self.sx = de.Sweep(cfg, bcs[0], dx, gam, e0, Rgas)
        self.sy = de.Sweep(cfg, bcs[1], dx, gam.T, e0.T, Rgas.T)
        self.gam, self.e0, self.R = gam, e0, Rgas
        self.bcs = bcs

    def primitives(self, Q):
        return self.sx.primitives(Q)

    def update(self, Qb, Fx, Fy, dt):
        lam = dt / self.dx
        return Qb + _increment(Fx, lam) + _from_y(_increment(Fy, lam))

    def detect(self, Qa, W_eval):
        """Admissibility detectors (entropy bound over the 5-point stencil)."""
        c = self.cfg
        finite = np.all(np.isfinite(Qa), axis=0)
        bad = ~finite | ~(Qa[0] > 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            Wa = self.primitives(Qa)
        Y = Wa[4]
        bad |= ~(Wa[3] > 0.0) | ~np.all(np.isfinite(Wa), axis=0)
        bad |= (Y < -c.y_tol) | (Y > 1.0 + c.y_tol)
        if c.check_entropy:
            with np.errstate(invalid="ignore", divide="ignore"):
                s_new = np.log(Wa[3]) - self.gam * np.log(Wa[0] * self.R)
                neigh = []
                for axis, bc in ((1, self.bcs[0]), (0, self.bcs[1])):
                    for shift in (-1, 1):
                        if bc == "periodic":
                            r = np.roll(W_eval[0], shift, axis=axis)
                            p = np.roll(W_eval[3], shift, axis=axis)
                        else:
                            r = _shift_edge(W_eval[0], shift, axis)
                            p = _shift_edge(W_eval[3], shift, axis)
                        neigh.append(np.log(p) - self.gam * np.log(r * self.R))
                neigh.append(np.log(W_eval[3]) - self.gam * np.log(W_eval[0] * self.R))
                bound = np.minimum.reduce(neigh)
            tol = c.entropy_atol + c.entropy_rtol * np.maximum(1.0, np.abs(bound))
            bad |= ~np.isfinite(s_new) | (s_new < bound - tol)
        return bad, Wa

    def euler(self, Q, dt):
        """Forward-Euler stage with cell-based MOOD levels."""
        c = self.cfg
        levx = self.sx.level_fluxes(Q)
        levy = self.sy.level_fluxes(_swap(Q))
        W_eval = levx["W"]
        top = 3 if c.use_thinc else 2
        cell = np.full(Q.shape[1:], top, dtype=np.int8)

        def mixed(lev, cell_dir, ext):
            ce = ext(cell_dir)
            face = np.minimum(ce[..., :-1], ce[..., 1:])
            F = lev["F"][0].copy()
            for l in range(1, top + 1):
                F = np.where((face == l)[None], lev["F"][l], F)
            return F, face

        for _ in range(4 * cell.size + 4):
            Fx, fx = mixed(levx, cell, self.sx.ext)
            Fy, fy = mixed(levy, cell.T, self.sy.ext)
            Qa = self.update(Q, Fx, Fy, dt)
            bad, Wa = self.detect(Qa, W_eval)
            if not bad.any():
                break
            if np.any(bad & (cell == 0)):
                raise RuntimeError("MOOD: inadmissible cells at the terminal level")
            cell = np.where(bad, cell - 1, cell).astype(np.int8)
        else:
            raise RuntimeError("MOOD loop did not converge")

        diag = dict(cell_level=cell,
                    thinc=(levx["thinc_cell"] | levy["thinc_cell"].T) & (cell == 3),
                    diss=_cells_of(fx < top, None) | _cells_of(fy < top, None).T
                    | _cells_of(_diss_faces(levx, fx, top), None) | _cells_of(_diss_faces(levy, fy, top), None).T,
                    dmp=np.zeros(cell.shape, bool), fallback=0)
        if c.use_fct:
            Qa, Wa, dmp, nfb = self.fct(Q, Fx, Fy, levx, levy, dt, W_eval)
            diag["dmp"], diag["fallback"] = dmp, nfb
        return Qa, Wa, diag

    def fct(self, Qb, FHx, FHy, levx, levy, dt, W_eval):
        """Two-dimensional Zalesak DMP/FCT with post-correction fallback."""
        c = self.cfg
        lam = dt / self.dx
        FLx, FLy = levx["F"][0], levy["F"][0]
        rho = Qb[0]
        neigh = [rho]
        for axis, bc in ((1, self.bcs[0]), (0, self.bcs[1])):
            for shift in (-1, 1):
                neigh.append(np.roll(rho, shift, axis=axis) if bc == "periodic" else _shift_edge(rho, shift, axis))
        lo, hi = np.minimum.reduce(neigh), np.maximum.reduce(neigh)
        delta = c.dmp_abs_tolerance * np.maximum(1.0, np.maximum(np.abs(lo), np.abs(hi)))
        rlow = rho - lam * (FLx[0][..., 1:] - FLx[0][..., :-1]) - lam * (FLy[0][..., 1:] - FLy[0][..., :-1]).T
        Ax, Ay = FHx[0] - FLx[0], FHy[0] - FLy[0]
        contrib = [lam * Ax[..., :-1], -lam * Ax[..., 1:], (lam * Ay[..., :-1]).T, (-lam * Ay[..., 1:]).T]
        Pp = sum(np.maximum(a, 0) for a in contrib)
        Pm = sum(np.minimum(a, 0) for a in contrib)
        Qp, Qm = hi + delta - rlow, lo - delta - rlow
        Rp = np.where(Pp > 0, np.minimum(1.0, np.maximum(0.0, Qp) / np.where(Pp > 0, Pp, 1.0)), 1.0)
        Rm = np.where(Pm < 0, np.minimum(1.0, np.minimum(0.0, Qm) / np.where(Pm < 0, Pm, -1.0)), 1.0)

        def theta_dir(A, Rp_d, Rm_d, sweep, active_d):
            Rpe, Rme = sweep.ext(Rp_d), sweep.ext(Rm_d)
            th = np.where(A > 0, np.minimum(Rme[..., :-1], Rpe[..., 1:]),
                          np.where(A < 0, np.minimum(Rpe[..., :-1], Rme[..., 1:]), 1.0))
            ae = sweep.ext(active_d)
            return np.where(ae[..., :-1] | ae[..., 1:], th, 1.0)

        active = (levx["contact"] | levy["contact"].T) & ~(levx["shock"] | levy["shock"].T)
        thx = theta_dir(Ax, Rp, Rm, self.sx, active)
        thy = theta_dir(Ay, Rp.T, Rm.T, self.sy, active.T)
        nfb = 0
        for _ in range(Qb[0].size + 2):
            Fx = FLx + thx[None] * (FHx - FLx)
            Fy = FLy + thy[None] * (FHy - FLy)
            Qa = self.update(Qb, Fx, Fy, dt)
            bad, Wa = self.detect(Qa, W_eval)
            if not bad.any():
                break
            bx, by = self.sx.ext(bad), self.sy.ext(bad.T)
            fx, fy = bx[..., :-1] | bx[..., 1:], by[..., :-1] | by[..., 1:]
            new = np.count_nonzero(fx & (thx > 0)) + np.count_nonzero(fy & (thy > 0))
            if new == 0:
                raise RuntimeError("FCT: inadmissible with low-order flux on all faces")
            nfb += new
            thx, thy = np.where(fx, 0.0, thx), np.where(fy, 0.0, thy)
        else:
            raise RuntimeError("FCT fallback did not converge")
        dmp = _cells_of(thx < 1 - 1e-14, None) | _cells_of(thy < 1 - 1e-14, None).T
        return Qa, Wa, dmp, nfb

    def advance(self, Q, dt):
        """One SSP-RK3 step."""
        Q1, _, d1 = self.euler(Q, dt)
        Q2e, _, d2 = self.euler(Q1, dt)
        Q2 = 0.75 * Q + 0.25 * Q2e
        Q3e, _, d3 = self.euler(Q2, dt)
        return Q / 3.0 + 2.0 / 3.0 * Q3e, (d1, d2, d3)


def _shift_edge(a, shift, axis):
    out = np.roll(a, shift, axis=axis)
    idx = [slice(None)] * a.ndim
    idx[axis] = 0 if shift == 1 else -1
    out[tuple(idx)] = a[tuple(idx)]
    return out


def _cells_of(face_mask, _):
    return face_mask[..., :-1] | face_mask[..., 1:]


def _diss_faces(lev, face, top):
    out = np.zeros_like(face, dtype=bool)
    for l in range(top + 1):
        out |= (face == l) & lev["diss"][l]
    return out


def run(cfg: de.Config):
    """Run one drop computation."""
    cfg = cfg.normalized()
    s = de.setup(cfg)
    n, dx = cfg.n, s["dx"]
    bcs = ("periodic", "periodic" if cfg.case == "oblique" else "wall")
    V = np.stack((s["rho"], np.full_like(s["rho"], s["u0"]), np.full_like(s["rho"], s["v0"]),
                  np.full_like(s["rho"], s["p0"]), s["Y"]))
    _, _, _, _, E0, _ = de.project(V)
    rho0, Y0 = s["rho"].copy(), s["Y"].copy()
    mass0, fuel0, energy0, energy0_abs = rho0.sum(), (rho0 * Y0).sum(), E0.sum(), np.abs(E0).sum()
    period = s["L"] / max(abs(s["u0"]), abs(s["v0"]))
    t_final = cfg.periods * period
    hist = {k: [] for k in ("time", "energy_error", "pressure_error", "u_error", "v_error", "mass_error")}
    t, step, tic = 0.0, 0, time.perf_counter()
    while t < t_final * (1.0 - 1e-12):
        gam, e0, _, c, E, Rgas = de.project(V)
        dt = min(cfg.cfl / np.max((np.abs(V[1]) + c) / dx + (np.abs(V[2]) + c) / dx), t_final - t)
        Q = np.stack((V[0], V[0] * V[1], V[0] * V[2], E, V[0] * V[4]))
        op = Unsplit(cfg, bcs, dx, gam, e0, Rgas)
        Qn, stages = op.advance(Q, dt)
        W = op.primitives(Qn)
        V = np.stack((W[0], W[1], W[2], W[3], W[4]))
        t += dt
        step += 1
        _, _, _, _, E, _ = de.project(V)
        hist["time"].append(t)
        hist["energy_error"].append(abs(E.sum() - energy0) / energy0_abs)
        hist["pressure_error"].append(float(np.max(np.abs(V[3] / s["p0"] - 1.0))))
        hist["u_error"].append(float(np.max(np.abs(V[1] - s["u0"]))))
        hist["v_error"].append(float(np.max(np.abs(V[2] - s["v0"]))))
        hist["mass_error"].append(abs(V[0].sum() - mass0) / mass0)
        if step % 200 == 0:
            print(f"[unsplit {cfg.case}/{cfg.method} N={n}] step {step} t/T={t / period:.3f} "
                  f"p_err={hist['pressure_error'][-1]:.2e} ({time.perf_counter() - tic:.0f}s)", flush=True)
    rho, u, v, p, Y = V
    jump = rho0.max() - rho0.min()
    prof = Y[n // 2]
    metrics = dict(
        status="completed", N=n, method=cfg.method, case=cfg.case, splitting="unsplit", steps=step,
        period=period, runtime_s=time.perf_counter() - tic, dx_m=dx,
        rho_L1=float(np.mean(np.abs(rho - rho0))), Y_L1=float(np.mean(np.abs(Y - Y0))),
        Y_min=float(Y.min()), Y_max=float(Y.max()), rho_min=float(rho.min()), rho_max=float(rho.max()),
        overshoot_pct_jump=100.0 * max(0.0, rho.max() - rho0.max(), rho0.min() - rho.min()) / jump,
        mass_rel=hist["mass_error"][-1], fuel_mass_rel=float(abs((rho * Y).sum() - fuel0) / fuel0),
        energy_rel=hist["energy_error"][-1],
        pressure_error_max=float(np.max(hist["pressure_error"])),
        u_error_max=float(np.max(hist["u_error"])), v_error_max=float(np.max(hist["v_error"])),
        centerline_mixed_width_m=np.count_nonzero((prof > 0.05) & (prof < 0.95)) * dx,
        final_center_x=s["L"] / 2, final_center_y=s["L"] / 2,
        mood_min_level_final=int(min(d["cell_level"].min() for d in stages)),
        total_fct_fallback_faces_last_step=int(sum(d["fallback"] for d in stages)))
    arrays = dict(x=s["x"], y=s["y"], rho_final=rho, rho_exact=rho0, Yfuel_final=Y, Yfuel_exact=Y0,
                  u_final=u, v_final=v, p_final=p,
                  **{("history_" + k if k != "time" else "time"): np.asarray(v_) for k, v_ in hist.items()})
    return dict(config=asdict(cfg), metrics=metrics, arrays=arrays)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", default=["full"])
    ap.add_argument("--grids", nargs="+", type=int, default=[100])
    ap.add_argument("--case", default="horizontal")
    ap.add_argument("--periods", type=float, default=1.0)
    ap.add_argument("--output", type=Path, default=Path("drop_euler_results_unsplit"))
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for m in a.methods:
        for n in a.grids:
            res = run(de.Config(method=m, case=a.case, n=n, periods=a.periods))
            stem = f"drop_{m}_N{n}" + ("_oblique" if a.case == "oblique" else "")
            np.savez_compressed(a.output / f"{stem}.npz", **res["arrays"])
            (a.output / f"{stem}.json").write_text(json.dumps(
                dict(config=res["config"], metrics=res["metrics"]), indent=2, default=str))
            print(json.dumps(res["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
