"""Two-dimensional double-flux Euler solver for the nitrogen / n-dodecane mixture.

Directional (Strang) splitting, THINC-BVD on density and mass fraction,
local multilevel MOOD, full-vector DMP/FCT with post-correction check, and
projection onto the Peng--Robinson EoS between sweeps.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import thermo_binary_pr as pr

NG = 2
RU = pr.RU


@dataclass
class Config:
    """Numerical parameters of a run."""
    method: str = "full"
    case: str = "horizontal"
    n: int = 100
    cfl: float = 0.8
    periods: float = 1.0
    zeta: float = 0.2
    zeta_p: float | None = None
    beta: float = 1.6
    limit_ho: bool | None = None
    check_entropy: bool = True
    entropy_atol: float = 1.0e-12
    entropy_rtol: float = 1.0e-6
    y_tol: float = 1.0e-9
    contact_threshold: float = 5.0e-2
    contact_weights: tuple = (20.0, 20.0)
    velocity_floor: float = 1.0
    shock_pressure_threshold: float = 5.0e-2
    shock_compression_threshold: float = 1.0e-3
    shock_velocity_threshold: float = 5.0e-2
    dmp_abs_tolerance: float = 1.0e-12
    shock_mask_threshold: float = 2.0e-2
    shock_mask_buffer: int = 2
    history_stride: int = 1

    def normalized(self) -> "Config":
        cfg = Config(**asdict(self))
        if cfg.method not in ("df_ho", "df_thinc", "full", "full_nomood"):
            raise ValueError("method must be df_ho, df_thinc, full or full_nomood")
        if cfg.case not in ("horizontal", "oblique"):
            raise ValueError("case must be horizontal or oblique")
        if cfg.limit_ho is None:
            cfg.limit_ho = False
        return cfg

    @property
    def use_thinc(self):
        return self.method in ("df_thinc", "full", "full_nomood")

    @property
    def use_fct(self):
        return self.method in ("full", "full_nomood")

    @property
    def use_mood(self):
        return self.method != "full_nomood"


def pad(a, bc, negate=False):
    """Pad the last axis with ghost cells (periodic, transmissive, or wall)."""
    if bc == "periodic":
        left, right = a[..., -NG:], a[..., :NG]
    elif bc == "transmissive":
        left = np.repeat(a[..., :1], NG, axis=-1)
        right = np.repeat(a[..., -1:], NG, axis=-1)
    else:
        left, right = a[..., NG - 1::-1], a[..., :-NG - 1:-1]
        if negate:
            left, right = -left, -right
    return np.concatenate((left, a, right), axis=-1)


def ext_index(n, bc):
    """Physical index of the extended cells -1..n."""
    if bc == "periodic":
        return np.concatenate(([n - 1], np.arange(n), [0]))
    return np.concatenate(([0], np.arange(n), [n - 1]))


def eno2_traces(q):
    """ENO2 left/right traces of the extended cells."""
    qm, q0, qp = q[..., :-2], q[..., 1:-1], q[..., 2:]
    dm, dp = q0 - qm, qp - q0
    right = q0 + 0.5 * np.where(np.abs(dm) <= np.abs(dp), dm, dp)
    left = q0 - 0.5 * np.where(np.abs(dp) <= np.abs(dm), dp, dm)
    return left, right


def thinc_traces(q, beta):
    """THINC left/right traces of the extended cells."""
    qm, q0, qp = q[..., :-2], q[..., 1:-1], q[..., 2:]
    qmin, qmax = np.minimum(qm, qp), np.maximum(qm, qp)
    dq = qmax - qmin
    scale = np.maximum.reduce([np.ones_like(q0), np.abs(qm), np.abs(q0), np.abs(qp)])
    active = ((q0 - qm) * (qp - q0) > 0.0) & (dq > 1.0e-14 * scale)
    theta = np.sign(qp - qm)
    C = np.divide(q0 - qmin, dq, out=np.full_like(q0, 0.5), where=dq > 0)
    tiny = np.finfo(float).eps
    C = np.clip(C, tiny, 1.0 - tiny)
    hbar = np.clip(theta * (2.0 * C - 1.0), -1.0 + tiny, 1.0 - tiny)
    r = np.exp(beta * hbar)
    floor = np.finfo(float).tiny
    y = np.maximum(np.exp(beta) - r, floor) / np.maximum(r - np.exp(-beta), floor)
    xi0 = 0.5 * np.log(y) / beta
    left = np.clip(qmin + 0.5 * dq * (1.0 + theta * np.tanh(-beta * xi0)), qmin, qmax)
    right = np.clip(qmin + 0.5 * dq * (1.0 + theta * np.tanh(beta * (1.0 - xi0))), qmin, qmax)
    left = np.where(active, left, q0)
    right = np.where(active, right, q0)
    return left, right


def central_face(q, limit):
    """Centred four-point face values, optionally clipped."""
    a, b, c, d = q[..., :-3], q[..., 1:-2], q[..., 2:-1], q[..., 3:]
    f = (-a + 7.0 * b + 7.0 * c - d) / 12.0
    if limit:
        f = np.clip(f, np.minimum.reduce([a, b, c, d]), np.maximum.reduce([a, b, c, d]))
    return f


def owner_energy(rho, un, ut, p, gam, e0):
    """Total energy from the frozen affine relation of the owner cell."""
    return p / (gam - 1.0) + rho * e0 + 0.5 * rho * (un * un + ut * ut)


def flux_central(W, gL, eL, gR, eR):
    """Central flux with owner-dependent energies."""
    rho, un, ut, p, Y = W
    mass = rho * un
    EL = owner_energy(rho, un, ut, p, gL, eL)
    ER = owner_energy(rho, un, ut, p, gR, eR)
    return np.stack((mass, mass * un + p, mass * ut, un * (EL + p), un * (ER + p), mass * Y))


def _sgn(s):
    return (s >= 0).astype(float) - (s <= 0).astype(float)


def flux_hllc(WL, WR, gL, eL, gR, eR):
    """HLLC flux with owner-dependent energies."""
    rL, uL, tL, pL, YL = WL
    rR, uR, tR, pR, YR = WR
    aL = np.sqrt(gL * pL / rL)
    aR = np.sqrt(gR * pR / rR)
    sL, sR = uL - aL, uR + aR
    sM = (pR - pL + rL * uL * (sL - uL) - rR * uR * (sR - uR)) / (rL * (sL - uL) - rR * (sR - uR))
    pM = rR * (uR - sR) * (uR - sM) + pR
    sm, sp, sg = np.minimum(0.0, sL), np.maximum(0.0, sR), _sgn(sM)
    wl, wr = 0.5 * (1.0 + sg), 0.5 * (1.0 - sg)

    rsL = (sL - uL) / (sL - sM) * rL
    rsR = (sR - uR) / (sR - sM) * rR
    mL, mR = rL * uL, rR * uR
    mass = wl * (mL + sm * (rsL - rL)) + wr * (mR + sp * (rsR - rR))
    momn = wl * (mL * uL + pL + sm * (((sL - uL) * mL + (pM - pL)) / (sL - sM) - mL)) \
        + wr * (mR * uR + pR + sp * (((sR - uR) * mR + (pM - pR)) / (sR - sM) - mR))
    momt = wl * (mL * tL + sm * (rsL * tL - rL * tL)) + wr * (mR * tR + sp * (rsR * tR - rR * tR))
    spec = wl * (mL * YL + sm * (rsL * YL - rL * YL)) + wr * (mR * YR + sp * (rsR * YR - rR * YR))

    def energy(g, e):
        EL = owner_energy(rL, uL, tL, pL, g, e)
        ER = owner_energy(rR, uR, tR, pR, g, e)
        EsL = ((sL - uL) * EL - pL * uL + pM * sM) / (sL - sM)
        EsR = ((sR - uR) * ER - pR * uR + pM * sM) / (sR - sM)
        return wl * (uL * (EL + pL) + sm * (EsL - EL)) + wr * (uR * (ER + pR) + sp * (EsR - ER))

    return np.stack((mass, momn, momt, energy(gL, eL), energy(gR, eR), spec))


def flux_llf(WL, WR, gL, eL, gR, eR):
    """Local Lax--Friedrichs flux; species flux = upwind Y times mass flux."""
    rL, uL, tL, pL, YL = WL
    rR, uR, tR, pR, YR = WR
    aL = np.sqrt(gL * pL / rL)
    aR = np.sqrt(gR * pR / rR)
    alpha = np.maximum(np.abs(uL) + aL, np.abs(uR) + aR)
    mL, mR = rL * uL, rR * uR
    mass = 0.5 * (mL + mR) - 0.5 * alpha * (rR - rL)
    momn = 0.5 * (mL * uL + pL + mR * uR + pR) - 0.5 * alpha * (mR - mL)
    momt = 0.5 * (mL * tL + mR * tR) - 0.5 * alpha * (rR * tR - rL * tL)

    def energy(g, e):
        EL = owner_energy(rL, uL, tL, pL, g, e)
        ER = owner_energy(rR, uR, tR, pR, g, e)
        return 0.5 * (uL * (EL + pL) + uR * (ER + pR)) - 0.5 * alpha * (ER - EL)

    spec = np.where(mass >= 0.0, YL, YR) * mass
    return np.stack((mass, momn, momt, energy(gL, eL), energy(gR, eR), spec))


class Sweep:
    """Directional operator along the last array axis."""

    def __init__(self, cfg, bc, dx, gam, e0, Rgas):
        self.cfg, self.bc, self.dx = cfg, bc, dx
        self.n = gam.shape[-1]
        self.cmap = ext_index(self.n, bc)
        g_ext, e_ext = gam[..., self.cmap], e0[..., self.cmap]
        self.gL, self.eL = g_ext[..., :-1], e_ext[..., :-1]
        self.gR, self.eR = g_ext[..., 1:], e_ext[..., 1:]
        self.gam, self.e0, self.R = gam, e0, Rgas

    def primitives(self, Q):
        """Primitive variables from the frozen affine relation."""
        rho, mn, mt, E, Z = Q
        un, ut, Y = mn / rho, mt / rho, Z / rho
        p = (self.gam - 1.0) * (E - 0.5 * rho * (un * un + ut * ut) - rho * self.e0)
        return np.stack((rho, un, ut, p, Y))

    def ext(self, a):
        """Values of a cell field on the extended cells."""
        return a[..., self.cmap]

    def classify(self, Wp):
        """Contact and shock flags (including the dilated shock mask)."""
        c = self.cfg
        rho, un, _, p, _ = Wp
        rL, rR = rho[..., NG - 1:-NG], rho[..., NG:-NG + 1 or None]
        uL, uR = un[..., NG - 1:-NG], un[..., NG:-NG + 1 or None]
        pL, pR = p[..., NG - 1:-NG], p[..., NG:-NG + 1 or None]
        Sr = np.abs(rR - rL) / (np.abs(rR) + np.abs(rL) + 1e-14)
        Sp = np.abs(pR - pL) / (np.abs(pR) + np.abs(pL) + 1e-14)
        us = np.maximum(np.abs(uR) + np.abs(uL), c.velocity_floor)
        Su = np.abs(uR - uL) / (us + 1e-14)
        comp = np.maximum(0.0, uL - uR) / (us + 1e-14)
        score = Sr * np.exp(-c.contact_weights[0] * Sp - c.contact_weights[1] * Su)
        shock_f = (Sp > c.shock_pressure_threshold) & (
            (comp > c.shock_compression_threshold) | (Su > c.shock_velocity_threshold))
        contact_f = (~shock_f) & (score > c.contact_threshold)
        mask = Sp >= c.shock_mask_threshold
        if self.bc == "periodic":
            base = mask[..., :-1].copy()
            m = base.copy()
            for k in range(1, int(c.shock_mask_buffer) + 1):
                m |= np.roll(base, k, axis=-1) | np.roll(base, -k, axis=-1)
            mask = np.concatenate((m, m[..., :1]), axis=-1)
        else:
            base = mask.copy()
            for k in range(1, int(c.shock_mask_buffer) + 1):
                mask[..., k:] |= base[..., :-k]
                mask[..., :-k] |= base[..., k:]
        shock_f = shock_f | mask
        shock_c = shock_f[..., :-1] | shock_f[..., 1:]
        contact_c = (~shock_c) & (contact_f[..., :-1] | contact_f[..., 1:])
        return contact_c, shock_c, shock_f

    def level_fluxes(self, Q):
        """Face fluxes of all MOOD levels."""
        c = self.cfg
        W = self.primitives(Q)
        Wp = np.stack([pad(W[k], self.bc, negate=(k == 1)) for k in range(5)])
        WL0, WR0 = Wp[..., NG - 1:-NG], Wp[..., NG:-NG + 1 or None]
        F0 = flux_llf(WL0, WR0, self.gL, self.eL, self.gR, self.eR)

        tr = [eno2_traces(Wp[k]) for k in range(5)]
        left = np.stack([t[0] for t in tr])
        right = np.stack([t[1] for t in tr])
        WLe, WRe = right[..., :-1], left[..., 1:]
        F1 = flux_hllc(WLe, WRe, self.gL, self.eL, self.gR, self.eR)

        Wc = np.stack([central_face(Wp[k], c.limit_ho) for k in range(5)])
        Fc = flux_central(Wc, self.gL, self.eL, self.gR, self.eR)
        rho_c = Wc[0]
        rs = np.maximum(np.abs(rho_c - WL0[0]) / WL0[0], np.abs(rho_c - WR0[0]) / WR0[0]) > c.zeta
        if c.zeta_p is not None:
            p_c = Wc[3]
            rs |= np.maximum(np.abs(p_c - WL0[3]) / WL0[3], np.abs(p_c - WR0[3]) / WR0[3]) > c.zeta_p

        contact_c, shock_c, shock_f = self.classify(Wp)
        diss2 = rs | shock_f
        F2 = np.where(diss2[None], F1, Fc)

        thinc_cell = np.zeros(W.shape[1:], dtype=bool)
        F3, diss3 = F2, diss2
        if c.use_thinc:
            rl_t, rr_t = thinc_traces(Wp[0], c.beta)
            yl_t, yr_t = thinc_traces(Wp[4], c.beta)
            jE = np.abs(left[0][..., 1:] - right[0][..., :-1])
            jT = np.abs(rl_t[..., 1:] - rr_t[..., :-1])
            tbvE, tbvT = jE[..., :-1] + jE[..., 1:], jT[..., :-1] + jT[..., 1:]
            prefer = tbvT + (1e-14 + 1e-12 * np.maximum(tbvE, tbvT)) < tbvE
            thinc_cell = prefer & contact_c
            use_ext = self.ext(thinc_cell)
            leftT, rightT = left.copy(), right.copy()
            leftT[0] = np.where(use_ext, rl_t, left[0])
            rightT[0] = np.where(use_ext, rr_t, right[0])
            leftT[4] = np.where(use_ext, yl_t, left[4])
            rightT[4] = np.where(use_ext, yr_t, right[4])
            thinc_face = use_ext[..., :-1] | use_ext[..., 1:]
            F1t = flux_hllc(rightT[..., :-1], leftT[..., 1:], self.gL, self.eL, self.gR, self.eR)
            diss3 = diss2 | thinc_face
            F3 = np.where(thinc_face[None], F1t, F2)

        return dict(F={0: F0, 1: F1, 2: F2, 3: F3}, diss={0: np.ones_like(diss2), 1: np.ones_like(diss2), 2: diss2, 3: diss3},
                    thinc_cell=thinc_cell, contact=contact_c, shock=shock_c, W=W)

    def update(self, Qb, F, dt):
        """Forward-Euler update with owner-dependent energy fluxes."""
        lam = dt / self.dx
        d = F[..., 1:] - F[..., :-1]
        out = np.empty_like(Qb)
        out[0] = Qb[0] - lam * d[0]
        out[1] = Qb[1] - lam * d[1]
        out[2] = Qb[2] - lam * d[2]
        out[3] = Qb[3] - lam * (F[3][..., 1:] - F[4][..., :-1])
        out[4] = Qb[4] - lam * d[5]
        return out

    def entropy(self, rho, p):
        """Surrogate entropy with the frozen exponent."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log(p) - self.gam * np.log(rho * self.R)

    def detect(self, Qa, W_eval):
        """Admissibility detectors of a candidate state."""
        c = self.cfg
        finite = np.all(np.isfinite(Qa), axis=0)
        rho = Qa[0]
        bad = ~finite | ~(rho > 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            Wa = self.primitives(Qa)
        Y = Wa[4]
        bad |= ~(Wa[3] > 0.0) | ~np.all(np.isfinite(Wa), axis=0)
        bad |= (Y < -c.y_tol) | (Y > 1.0 + c.y_tol)
        if c.check_entropy:
            s_new = self.entropy(Wa[0], Wa[3])
            rp = pad(W_eval[0], self.bc)
            pp = pad(W_eval[3], self.bc)
            sl = slice(NG - 1, -NG - 1)
            sc = slice(NG, -NG)
            sr = slice(NG + 1, None if NG == 1 else -NG + 1)
            g = self.gam
            R = self.R
            with np.errstate(invalid="ignore", divide="ignore"):
                s3 = [np.log(pp[..., k]) - g * np.log(rp[..., k] * R) for k in (sl, sc, sr)]
            bound = np.minimum.reduce(s3)
            tol = c.entropy_atol + c.entropy_rtol * np.maximum(1.0, np.abs(bound))
            bad |= ~np.isfinite(s_new) | (s_new < bound - tol)
        return bad, Wa

    def euler(self, Q_eval, Q_before, dt):
        """Forward-Euler stage with local MOOD and DMP/FCT."""
        c = self.cfg
        lev = self.level_fluxes(Q_eval)
        F, W_eval = lev["F"], lev["W"]
        top = 3 if c.use_thinc else 2
        cell = np.full(W_eval.shape[1:], top, dtype=np.int8)
        for _ in range(4 * self.n + 4):
            ce = self.ext(cell)
            face = np.minimum(ce[..., :-1], ce[..., 1:])
            Fm = F[0].copy()
            for l in (1, 2, 3):
                if l <= top:
                    Fm = np.where((face == l)[None], F[l], Fm)
            Qa = self.update(Q_before, Fm, dt)
            bad, Wa = self.detect(Qa, W_eval)
            if not bad.any() or not c.use_mood:
                break
            if np.any(bad & (cell == 0)):
                raise RuntimeError("MOOD: inadmissible cells at the terminal level")
            cell = np.where(bad, cell - 1, cell).astype(np.int8)
        else:
            raise RuntimeError("MOOD loop did not converge")

        diss = np.zeros_like(face, dtype=bool)
        for l in range(top + 1):
            diss |= (face == l) & lev["diss"][l]
        thinc_cell = lev["thinc_cell"] & (cell == 3)
        dmp_face = np.zeros_like(face, dtype=bool)
        fallback = np.zeros_like(face, dtype=bool)

        if c.use_fct:
            Qa, dmp_face, fallback, Wa = self.fct(Q_before, Fm, F[0], lev, dt, W_eval)

        return Qa, Wa, dict(cell_level=cell, diss_face=diss, thinc_cell=thinc_cell,
                            dmp_face=dmp_face, fallback_face=fallback)

    def fct(self, Qb, FH, FL, lev, dt, W_eval):
        """Full-vector density DMP/FCT with post-correction fallback."""
        c = self.cfg
        lam = dt / self.dx
        rho = Qb[0]
        rp = pad(rho, self.bc)
        lo = np.minimum.reduce([rp[..., NG - 1:-NG - 1], rp[..., NG:-NG], rp[..., NG + 1:-NG + 1 or None]])
        hi = np.maximum.reduce([rp[..., NG - 1:-NG - 1], rp[..., NG:-NG], rp[..., NG + 1:-NG + 1 or None]])
        delta = c.dmp_abs_tolerance * np.maximum(1.0, np.maximum(np.abs(lo), np.abs(hi)))
        rlow = rho - lam * (FL[0][..., 1:] - FL[0][..., :-1])
        A = FH[0] - FL[0]
        cin, cout = lam * A[..., :-1], -lam * A[..., 1:]
        Pp = np.maximum(cin, 0) + np.maximum(cout, 0)
        Pm = np.minimum(cin, 0) + np.minimum(cout, 0)
        Qp, Qm = hi + delta - rlow, lo - delta - rlow
        Rp = np.where(Pp > 0, np.minimum(1.0, np.maximum(0.0, Qp) / np.where(Pp > 0, Pp, 1.0)), 1.0)
        Rm = np.where(Pm < 0, np.minimum(1.0, np.minimum(0.0, Qm) / np.where(Pm < 0, Pm, -1.0)), 1.0)
        Rpe, Rme = self.ext(Rp), self.ext(Rm)
        theta = np.where(A > 0, np.minimum(Rme[..., :-1], Rpe[..., 1:]),
                         np.where(A < 0, np.minimum(Rpe[..., :-1], Rme[..., 1:]), 1.0))
        active = self.ext(lev["contact"] & ~lev["shock"])
        theta = np.where(active[..., :-1] | active[..., 1:], theta, 1.0)
        fallback = np.zeros_like(theta, dtype=bool)
        for _ in range(self.n + 2):
            F = FL + theta[None] * (FH - FL)
            Qa = self.update(Qb, F, dt)
            bad, Wa = self.detect(Qa, W_eval)
            if not bad.any() or not c.use_mood:
                break
            be = self.ext(bad)
            faces = be[..., :-1] | be[..., 1:]
            new = faces & (theta > 0.0)
            if not new.any():
                raise RuntimeError("FCT: inadmissible with low-order flux on all faces")
            theta = np.where(faces, 0.0, theta)
            fallback |= new
        else:
            raise RuntimeError("FCT fallback did not converge")
        return Qa, theta < 1.0 - 1e-14, fallback, Wa

    def advance(self, Q, dt):
        """One SSP-RK3 step of the directional operator."""
        Q1, _, d1 = self.euler(Q, Q, dt)
        Q2e, _, d2 = self.euler(Q1, Q1, dt)
        Q2 = 0.75 * Q + 0.25 * Q2e
        Q3e, _, d3 = self.euler(Q2, Q2, dt)
        Qn = Q / 3.0 + 2.0 / 3.0 * Q3e
        return Qn, (d1, d2, d3)


def setup(cfg):
    """Initial drop configuration."""
    L, D, p0 = 1.0e-3, 0.5e-3, 6.0e6
    u0 = 50.0
    v0 = 50.0 if cfg.case == "oblique" else 0.0
    n = cfg.n
    dx = L / n
    x = (np.arange(n) + 0.5) * dx
    X, Yc = np.meshgrid(x, x)
    fuel = ((X - L / 2) ** 2 + (Yc - L / 2) ** 2 <= (D / 2) ** 2).astype(float)
    T = np.where(fuel > 0.5, 363.0, 900.0)
    rho = pr.density_from_T_p_Y(T, p0, fuel)
    return dict(L=L, D=D, p0=p0, u0=u0, v0=v0, dx=dx, x=x, y=x.copy(), rho=rho, Y=fuel, T=T)


def project(V):
    """Projection onto the real EoS: gamma*, e0*, T, c, total energy, R."""
    rho, u, v, p, Y = V
    gam, e0, T = pr.gamma_e0_from_rho_p_Y(rho, p, Y)
    e = pr.internal_energy_from_T_rho_Y(T, rho, Y)
    c = np.sqrt(gam * p / rho)
    if not (np.all(np.isfinite(T)) and np.all(T > 0) and np.all(np.isfinite(c)) and np.all(c > 0)):
        raise RuntimeError("real-fluid projection failed (T or c not admissible)")
    E = rho * e + 0.5 * rho * (u * u + v * v)
    _, _, MW = pr.mass_to_mole_fraction(Y)
    return gam, e0, T, c, E, RU / MW


def sweep(V, dt, direction, cfg, bcs, dx, diag):
    """One directional sweep starting from the projected state."""
    gam, e0, _, _, E, Rgas = project(V)
    rho, u, v, p, Y = V
    if direction == "x":
        tr = lambda a: a
        un, ut, bc = u, v, bcs[0]
    else:
        tr = lambda a: a.T
        un, ut, bc = v, u, bcs[1]
    Q = np.stack([tr(rho), tr(rho * un), tr(rho * ut), tr(E), tr(rho * Y)])
    op = Sweep(cfg, bc, dx, tr(gam), tr(e0), tr(Rgas))
    Qn, stages = op.advance(Q, dt)
    W = op.primitives(Qn)
    rho_n, un_n, ut_n, p_n, Y_n = (tr(W[k]) for k in range(5))
    if direction == "x":
        Vn = np.stack((rho_n, un_n, ut_n, p_n, Y_n))
    else:
        Vn = np.stack((rho_n, ut_n, un_n, p_n, Y_n))
    for d in stages:
        lev = tr(d["cell_level"])
        diag["mood"] = np.minimum(diag["mood"], lev)
        diag["thinc"] |= tr(d["thinc_cell"])
        for key, name in (("diss_face", "diss"), ("dmp_face", "dmp"), ("fallback_face", "fallback")):
            f = d[key]
            cellf = f[..., :-1] | f[..., 1:]
            diag[name] |= tr(cellf)
        diag["fallback_count"] += int(np.count_nonzero(d["fallback_face"]))
    return Vn, Qn


def run(cfg: Config):
    """Run one drop computation and collect diagnostics."""
    cfg = cfg.normalized()
    s = setup(cfg)
    n, dx = cfg.n, s["dx"]
    bcs = ("periodic", "periodic" if cfg.case == "oblique" else "wall")
    V = np.stack((s["rho"], np.full_like(s["rho"], s["u0"]), np.full_like(s["rho"], s["v0"]),
                  np.full_like(s["rho"], s["p0"]), s["Y"]))
    gam0, e00, T0, c0, E0, _ = project(V)
    rho0, Y0 = s["rho"].copy(), s["Y"].copy()
    mass0, fuel0, energy0 = rho0.sum(), (rho0 * Y0).sum(), E0.sum()
    energy0_abs = np.abs(E0).sum()
    area0 = np.count_nonzero(Y0 > 0.5)
    period = s["L"] / max(abs(s["u0"]), abs(s["v0"]))
    t_final = cfg.periods * period

    hist = {k: [] for k in ("time", "mass_error", "fuel_mass_error", "energy_error", "area_error",
                            "pressure_error", "u_error", "v_error", "Y_min", "Y_max", "rho_min", "rho_max")}

    def record(t, V, E):
        rho, u, v, p, Y = V
        hist["time"].append(t)
        hist["mass_error"].append(abs(rho.sum() - mass0) / mass0)
        hist["fuel_mass_error"].append(abs((rho * Y).sum() - fuel0) / fuel0)
        hist["energy_error"].append(abs(E.sum() - energy0) / energy0_abs)
        hist["area_error"].append(abs(np.count_nonzero(Y > 0.5) - area0) / area0)
        hist["pressure_error"].append(float(np.max(np.abs(p / s["p0"] - 1.0))))
        hist["u_error"].append(float(np.max(np.abs(u - s["u0"]))))
        hist["v_error"].append(float(np.max(np.abs(v - s["v0"]))))
        hist["Y_min"].append(float(Y.min()))
        hist["Y_max"].append(float(Y.max()))
        hist["rho_min"].append(float(rho.min()))
        hist["rho_max"].append(float(rho.max()))

    record(0.0, V, E0)
    failure = ""
    t, step, tic = 0.0, 0, time.perf_counter()
    while t < t_final * (1.0 - 1e-12):
        _, _, _, c, _, _ = project(V)
        u, v = V[1], V[2]
        dt = cfg.cfl / np.max((np.abs(u) + c) / dx + (np.abs(v) + c) / dx)
        dt = min(dt, t_final - t)
        diag = dict(mood=np.full((n, n), 3, np.int8), thinc=np.zeros((n, n), bool),
                    diss=np.zeros((n, n), bool), dmp=np.zeros((n, n), bool),
                    fallback=np.zeros((n, n), bool), fallback_count=0)
        order = ("x", "y", "x") if step % 2 == 0 else ("y", "x", "y")
        try:
            for k, d in enumerate(order):
                V, _ = sweep(V, dt / 2 if k != 1 else dt, d, cfg, bcs, dx, diag)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            failure = f"{type(exc).__name__}: {exc}"
            print(f"[{cfg.case}/{cfg.method} N={n}] FAILED at t/T={t / period:.4f}: {failure}", flush=True)
            break
        if not (np.all(np.isfinite(V)) and np.all(V[0] > 0) and np.all(V[3] > 0)):
            failure = "inadmissible state (rho or p <= 0, or non-finite)"
            print(f"[{cfg.case}/{cfg.method} N={n}] FAILED at t/T={t / period:.4f}: {failure}", flush=True)
            break
        t += dt
        step += 1
        if step % cfg.history_stride == 0 or t >= t_final * (1.0 - 1e-12):
            _, _, _, _, E, _ = project(V)
            record(t, V, E)
        if step % 200 == 0:
            print(f"[{cfg.case}/{cfg.method} N={n}] step {step} t/T={t / period:.3f} "
                  f"p_err={hist['pressure_error'][-1]:.2e} ({time.perf_counter() - tic:.0f}s)", flush=True)

    rho, u, v, p, Y = V
    _, _, T, _, E, _ = project(V)
    jump = rho0.max() - rho0.min()
    j = n // 2
    prof = Y[j]
    mixed = np.count_nonzero((prof > 0.05) & (prof < 0.95)) * dx
    metrics = dict(
        status="failed" if failure else "completed", failure=failure, N=n, method=cfg.method, case=cfg.case, steps=step,
        period=period, time_final=t, runtime_s=time.perf_counter() - tic,
        dx_m=dx, dy_m=dx,
        rho_L1=float(np.mean(np.abs(rho - rho0))),
        Y_L1=float(np.mean(np.abs(Y - Y0))),
        rho_min=float(rho.min()), rho_max=float(rho.max()),
        Y_min=float(Y.min()), Y_max=float(Y.max()),
        Y_min_history=float(np.min(hist["Y_min"])), Y_max_history=float(np.max(hist["Y_max"])),
        mass_rel=hist["mass_error"][-1], fuel_mass_rel=hist["fuel_mass_error"][-1],
        energy_rel=hist["energy_error"][-1],
        pressure_error_max=float(np.max(hist["pressure_error"])),
        u_error_max=float(np.max(hist["u_error"])),
        v_error_max=float(np.max(hist["v_error"])),
        centerline_mixed_width_m=mixed,
        mixed_cell_fraction=float(np.mean((Y > 0.05) & (Y < 0.95))),
        overshoot_pct_jump=100.0 * max(0.0, rho.max() - rho0.max(), rho0.min() - rho.min()) / jump,
        final_center_x=s["L"] / 2, final_center_y=s["L"] / 2,
        max_thinc_cell_fraction_final=float(np.mean(diag["thinc"])),
        dissipative_cell_fraction_final=float(np.mean(diag["diss"])),
        dmp_cell_fraction_final=float(np.mean(diag["dmp"])),
        mood_min_level_final=int(diag["mood"].min()),
        mood_demoted_fraction_final=float(np.mean(diag["mood"] < (3 if cfg.use_thinc else 2))),
        total_fct_fallback_faces_last_step=int(diag["fallback_count"]),
        mood_available=True,
    )
    arrays = dict(
        x=s["x"], y=s["y"], rho_final=rho, rho_exact=rho0, Yfuel_final=Y, Yfuel_exact=Y0,
        u_final=u, v_final=v, p_final=p, T_final=T,
        thinc_map=diag["thinc"].astype(np.int8), dissipative_map=diag["diss"].astype(np.int8),
        dmp_map=diag["dmp"].astype(np.int8), mood_map=diag["mood"],
        **{("history_" + k if k != "time" else "time"): np.asarray(v_) for k, v_ in hist.items()},
    )
    return dict(config=asdict(cfg), metrics=metrics, arrays=arrays)


def save(result, out: Path):
    """Save arrays (npz) and metrics (json)."""
    out.mkdir(parents=True, exist_ok=True)
    c = result["config"]
    stem = f"drop_{c['method']}_N{c['n']}" + ("_oblique" if c["case"] == "oblique" else "")
    np.savez_compressed(out / f"{stem}.npz", **result["arrays"])
    (out / f"{stem}.json").write_text(json.dumps(
        dict(config=c, metrics=result["metrics"]), indent=2, default=str))
    return out / f"{stem}.npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--methods", nargs="+", default=["full"])
    ap.add_argument("--grids", nargs="+", type=int, default=[100])
    ap.add_argument("--case", default="horizontal")
    ap.add_argument("--periods", type=float, default=1.0)
    ap.add_argument("--no-entropy", action="store_true")
    ap.add_argument("--output", type=Path, default=Path("drop_euler_results"))
    a = ap.parse_args()
    for m in a.methods:
        for n in a.grids:
            cfg = Config(method=m, case=a.case, n=n, periods=a.periods, check_entropy=not a.no_entropy)
            res = run(cfg)
            path = save(res, a.output)
            print(json.dumps(res["metrics"], indent=2))
            print("saved", path)


if __name__ == "__main__":
    main()
