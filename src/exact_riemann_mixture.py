"""Exact Riemann solution of the nitrogen / n-dodecane shock tubes (same EoS)."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

import thermo_binary_pr as pr


def p_of(T, rho, Y):
    return float(pr.pressure_from_T_rho_Y(T, rho, Y))


def e_of(T, rho, Y):
    return float(pr.internal_energy_from_T_rho_Y(T, rho, Y))


def c_of(T, rho, Y):
    return float(pr.sound_speed_from_T_rho_Y(T, rho, Y))


def dTdrho_isentrope(T, rho, Y):
    """Temperature derivative along an isentrope."""
    hT, hr = 1e-4 * T, 1e-6 * rho
    deT = (e_of(T + hT, rho, Y) - e_of(T - hT, rho, Y)) / (2 * hT)
    der = (e_of(T, rho + hr, Y) - e_of(T, rho - hr, Y)) / (2 * hr)
    return (p_of(T, rho, Y) / rho**2 - der) / deT


class Rarefaction:
    """Nitrogen isentrope parametrized by density."""

    def __init__(self, rhoL, TL, Y, rho_min):
        self.Y = Y
        rhs = lambda r, z: [dTdrho_isentrope(z[0], r, Y), -c_of(z[0], r, Y) / r]
        sol = solve_ivp(rhs, (rhoL, rho_min), [TL, 0.0], rtol=1e-10, atol=1e-10,
                        dense_output=True, max_step=(rhoL - rho_min) / 400)
        self.sol, self.rhoL, self.rho_min = sol, rhoL, rho_min

    def state(self, rho):
        T, u = self.sol.sol(rho)
        return T, u, p_of(T, rho, self.Y)

    def rho_from_p(self, p):
        return brentq(lambda r: self.state(r)[2] - p, self.rho_min, self.rhoL, xtol=1e-12)


def hugoniot(p, rhoR, TR, Y):
    """Post-shock state of n-dodecane for a given pressure."""
    pR, eR = p_of(TR, rhoR, Y), e_of(TR, rhoR, Y)

    def residual(rho):
        T = brentq(lambda T: p_of(T, rho, Y) - p, 200.0, 3000.0, xtol=1e-10)
        return e_of(T, rho, Y) - eR - 0.5 * (p + pR) * (1.0 / rhoR - 1.0 / rho), T

    rho = brentq(lambda r: residual(r)[0], rhoR * (1 + 1e-9), rhoR * 1.6, xtol=1e-12)
    T = residual(rho)[1]
    u = np.sqrt((p - pR) * (1.0 / rhoR - 1.0 / rho))
    return rho, T, u


def solve(pL, pR=6.0e6, T0=650.0):
    """Star state and wave speeds."""
    rhoL = float(pr.density_from_T_p_Y(np.array([T0]), np.array([pL]), np.array([0.0]))[0])
    rhoR = float(pr.density_from_T_p_Y(np.array([T0]), np.array([pR]), np.array([1.0]))[0])
    raref = Rarefaction(rhoL, T0, 0.0, rho_min=0.05 * rhoL)

    def mismatch(p):
        rhoLs = raref.rho_from_p(p)
        uL = raref.state(rhoLs)[1]
        uR = hugoniot(p, rhoR, T0, 1.0)[2]
        return uL - uR

    pstar = brentq(mismatch, pR * (1 + 1e-6), pL * (1 - 1e-6), xtol=1e-3)
    rhoLs = raref.rho_from_p(pstar)
    TLs, ustar, _ = raref.state(rhoLs)
    rhoRs, TRs, _ = hugoniot(pstar, rhoR, T0, 1.0)
    S = rhoRs * ustar / (rhoRs - rhoR)
    return dict(pL=pL, pR=pR, rhoL=rhoL, rhoR=rhoR, pstar=pstar, ustar=ustar,
                rhoLs=rhoLs, TLs=TLs, rhoRs=rhoRs, TRs=TRs, shock_speed=S,
                cL=c_of(T0, rhoL, 0.0), cLs=c_of(TLs, rhoLs, 0.0), raref=raref, T0=T0)


def profile(sol, x, t, x0=0.5):
    """Exact profiles at time t."""
    xi = (x - x0) / t
    out = {k: np.empty_like(x) for k in ("rho", "u", "p", "T", "Y")}
    head, tail = -sol["cL"], sol["ustar"] - sol["cLs"]
    raref = sol["raref"]
    for i, s in enumerate(xi):
        if s < head:
            r, u, p, T, Y = sol["rhoL"], 0.0, sol["pL"], sol["T0"], 0.0
        elif s < tail:
            r = brentq(lambda rr: raref.state(rr)[1] - c_of(raref.state(rr)[0], rr, 0.0) - s,
                       sol["rhoLs"], sol["rhoL"], xtol=1e-12)
            T, u, p = raref.state(r)
            Y = 0.0
        elif s < sol["ustar"]:
            r, u, p, T, Y = sol["rhoLs"], sol["ustar"], sol["pstar"], sol["TLs"], 0.0
        elif s < sol["shock_speed"]:
            r, u, p, T, Y = sol["rhoRs"], sol["ustar"], sol["pstar"], sol["TRs"], 1.0
        else:
            r, u, p, T, Y = sol["rhoR"], 0.0, sol["pR"], sol["T0"], 1.0
        out["rho"][i], out["u"][i], out["p"][i], out["T"][i], out["Y"][i] = r, u, p, T, Y
    return out


if __name__ == "__main__":
    for pL in (60.0e6, 7.0e6):
        s = solve(pL)
        print({k: (round(v, 4) if isinstance(v, float) else None) for k, v in s.items() if k != "raref"})
        t = 5.0e-4
        print("  contact at x =", 0.5 + s["ustar"] * t, " shock at x =", 0.5 + s["shock_speed"] * t,
              " rarefaction head/tail at", 0.5 - s["cL"] * t, 0.5 + (s["ustar"] - s["cLs"]) * t)
