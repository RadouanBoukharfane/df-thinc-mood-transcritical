"""Exact Riemann solution of the transcritical nitrogen shock tube (Peng--Robinson, Section 8.2)."""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from thermo import get_energy_from_T_rho, get_p_from_T_rho, get_sos, get_T_from_p_rho


def p_of(T, rho):
    return float(get_p_from_T_rho(np.array([T]), np.array([rho]))[0])


def e_of(T, rho):
    return float(get_energy_from_T_rho(np.array([T]), np.array([rho]))[0])


def c_of(T, rho):
    V = np.array([[rho], [0.0], [p_of(T, rho)]])
    return float(get_sos(V)[0])


def T_of(p, rho):
    return float(get_T_from_p_rho(np.array([p]), np.array([rho]))[0])


def dTdrho_isentrope(T, rho):
    """dT/drho along an isentrope, from de = p/rho^2 drho."""
    hT, hr = 1e-5 * T, 1e-7 * rho
    deT = (e_of(T + hT, rho) - e_of(T - hT, rho)) / (2 * hT)
    der = (e_of(T, rho + hr) - e_of(T, rho - hr)) / (2 * hr)
    return (p_of(T, rho) / rho**2 - der) / deT


class Rarefaction:
    """Left-facing isentrope parametrized by density; u = -int c/rho drho."""

    def __init__(self, rhoL, TL, rho_min):
        rhs = lambda r, z: [dTdrho_isentrope(z[0], r), -c_of(z[0], r) / r]
        self.sol = solve_ivp(rhs, (rhoL, rho_min), [TL, 0.0], method="DOP853", rtol=1e-11,
                             atol=1e-11, dense_output=True, max_step=(rhoL - rho_min) / 800)
        self.rhoL, self.rho_min = rhoL, rho_min

    def state(self, rho):
        T, u = self.sol.sol(rho)
        return T, u, p_of(T, rho)

    def rho_from_p(self, p):
        return brentq(lambda r: self.state(r)[2] - p, self.rho_min, self.rhoL, xtol=1e-13)


def hugoniot(p, rhoR, TR):
    """Post-shock state of a right-facing shock into a quiescent state."""
    pR, eR = p_of(TR, rhoR), e_of(TR, rhoR)

    def residual(rho):
        T = T_of(p, rho)
        return e_of(T, rho) - eR - 0.5 * (p + pR) * (1.0 / rhoR - 1.0 / rho)

    rho = brentq(residual, rhoR * (1 + 1e-10), rhoR * 8.0, xtol=1e-13)
    u = np.sqrt((p - pR) * (1.0 / rhoR - 1.0 / rho))
    return rho, T_of(p, rho), u


def solve(rhoL=800.0, pL=60.0e6, rhoR=80.0, pR=6.0e6):
    """Star state and wave speeds (left rarefaction, contact, right shock)."""
    TL, TR = T_of(pL, rhoL), T_of(pR, rhoR)
    # the PR isentrope leaves the mechanically stable branch below ~0.72 rhoL; 0.75 rhoL suffices here
    raref = Rarefaction(rhoL, TL, rho_min=0.75 * rhoL)
    mismatch = lambda p: raref.state(raref.rho_from_p(p))[1] - hugoniot(p, rhoR, TR)[2]
    pstar = brentq(mismatch, pR * (1 + 1e-6), pL * (1 - 1e-6), xtol=1e-4)
    rhoLs = raref.rho_from_p(pstar)
    TLs, ustar, _ = raref.state(rhoLs)
    rhoRs, TRs, _ = hugoniot(pstar, rhoR, TR)
    return dict(rhoL=rhoL, pL=pL, TL=TL, rhoR=rhoR, pR=pR, TR=TR, pstar=pstar, ustar=ustar,
                rhoLs=rhoLs, TLs=TLs, rhoRs=rhoRs, TRs=TRs,
                shock_speed=rhoRs * ustar / (rhoRs - rhoR),
                cL=c_of(TL, rhoL), cLs=c_of(TLs, rhoLs), raref=raref)


def profile(sol, x, t, x0=0.5):
    """Exact (rho, u, p, T) at time t."""
    xi = (np.asarray(x) - x0) / t
    out = {k: np.empty(len(xi)) for k in ("rho", "u", "p", "T")}
    head, tail, raref = -sol["cL"], sol["ustar"] - sol["cLs"], sol["raref"]
    for i, s in enumerate(xi):
        if s < head:
            r, u, p, T = sol["rhoL"], 0.0, sol["pL"], sol["TL"]
        elif s < tail:
            r = brentq(lambda rr: raref.state(rr)[1] - c_of(raref.state(rr)[0], rr) - s,
                       sol["rhoLs"], sol["rhoL"], xtol=1e-13)
            T, u, p = raref.state(r)
        elif s < sol["ustar"]:
            r, u, p, T = sol["rhoLs"], sol["ustar"], sol["pstar"], sol["TLs"]
        elif s < sol["shock_speed"]:
            r, u, p, T = sol["rhoRs"], sol["ustar"], sol["pstar"], sol["TRs"]
        else:
            r, u, p, T = sol["rhoR"], 0.0, sol["pR"], sol["TR"]
        out["rho"][i], out["u"][i], out["p"][i], out["T"][i] = r, u, p, T
    return out


def cell_average_profile(sol, x, dx, t, nsub=16):
    """Exact cell averages of rho, rho*u (as u), p, by sub-cell quadrature."""
    xs = (x[:, None] + dx * ((np.arange(nsub) + 0.5) / nsub - 0.5)[None, :]).ravel()
    prof = profile(sol, xs, t)
    return {k: v.reshape(len(x), nsub).mean(axis=1) for k, v in prof.items()}


if __name__ == "__main__":
    s = solve()
    print({k: round(v, 6) for k, v in s.items() if k != "raref"})
    t = 5.0e-4
    print("contact x =", 0.5 + s["ustar"] * t, " shock x =", 0.5 + s["shock_speed"] * t,
          " rarefaction head/tail =", 0.5 - s["cL"] * t, 0.5 + (s["ustar"] - s["cLs"]) * t)
