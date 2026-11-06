"""Phase-stability (tangent-plane-distance) analysis of nitrogen/n-dodecane PR states."""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

import thermo_binary_pr as pr

RU = pr.RU
S2 = np.sqrt(2.0)


def aij_matrix(T):
    """Cross attraction parameters a_ij(T) of the corresponding-states rule."""
    T = float(T)
    A = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            tc, pc, c = pr.TCIJ[i, j], pr.PCIJ[i, j], pr.CIJ[i, j]
            A[i, j] = 0.457236 * (RU * tc) ** 2 / pc * (1.0 + c * (1.0 - np.sqrt(T / tc))) ** 2
    return A


def z_roots(T, p, x, A):
    """Physical compressibility roots and corresponding a, b."""
    a = x @ A @ x
    b = x @ pr.BI
    Am, Bm = a * p / (RU * T) ** 2, b * p / (RU * T)
    r = np.roots([1.0, -(1.0 - Bm), Am - 3 * Bm**2 - 2 * Bm, -(Am * Bm - Bm**2 - Bm**3)])
    r = np.real(r[np.abs(np.imag(r)) < 1e-10])
    return r[r > Bm * (1 + 1e-12)], a, b, Am, Bm


def ln_phi(Z, x, A, a, b, Am, Bm):
    """Fugacity coefficients (PR, composition-independent a_ij)."""
    bi = pr.BI
    s = A @ x
    return (bi / b * (Z - 1.0) - np.log(Z - Bm)
            - Am / (2 * S2 * Bm) * (2 * s / a - bi / b) * np.log((Z + (1 + S2) * Bm) / (Z + (1 - S2) * Bm)))


def lnphi_min_gibbs(T, p, x, A):
    """ln(phi) of the root with the lowest Gibbs energy."""
    Zs, a, b, Am, Bm = z_roots(T, p, x, A)
    best = None
    for Z in Zs:
        lp = ln_phi(Z, x, A, a, b, Am, Bm)
        g = x @ lp
        if best is None or g < best[0]:
            best = (g, lp, Z)
    return best[1], best[2]


def lnphi_at_rho(T, p, x, A, rho, MWmix):
    """ln(phi) of the root closest to the given density (the actual cell state)."""
    Zs, a, b, Am, Bm = z_roots(T, p, x, A)
    Z_cell = p * MWmix / (rho * RU * T)
    Z = Zs[np.argmin(np.abs(Zs - Z_cell))]
    return ln_phi(Z, x, A, a, b, Am, Bm)


def tpd_min(T, p, Xf, rho=None, ntrial=400):
    """Minimum tangent-plane distance over trial compositions (binary: 1-D scan).

    Negative value: the state is thermodynamically unstable (would split into two phases).
    """
    A = aij_matrix(T)
    z = np.array([1.0 - Xf, Xf])
    if rho is None:
        lpz, _ = lnphi_min_gibbs(T, p, z, A)
    else:
        MWmix = z @ pr.MW
        lpz = lnphi_at_rho(T, p, z, A, rho, MWmix)
    dz = np.log(np.clip(z, 1e-300, None)) + lpz
    w1 = np.concatenate((np.logspace(-8, -1, ntrial // 4), np.linspace(0.1, 0.9, ntrial // 2),
                         1.0 - np.logspace(-1, -8, ntrial // 4)))
    best = np.inf
    for wf in w1:
        w = np.array([1.0 - wf, wf])
        lpw, _ = lnphi_min_gibbs(T, p, w, A)
        tpd = w @ (np.log(w) + lpw - dz)
        best = min(best, tpd)
    return best


def enthalpy(T, p, Yf):
    """Specific enthalpy at (T, p, Y) using the minimum-Gibbs root."""
    Xn, Xf, MWmix = pr.mass_to_mole_fraction(np.array(Yf))
    x = np.array([float(Xn), float(Xf)])
    A = aij_matrix(T)
    _, Z = lnphi_min_gibbs(T, p, x, A)
    rho = p * float(MWmix) / (Z * RU * T)
    e = float(pr.internal_energy_from_T_rho_Y(T, rho, Yf))
    return e + p / rho, rho


def adiabatic_mixing_line(p=6.0e6, T_f=363.0, T_ox=900.0, nZ=41):
    """Adiabatic (isobaric) mixing states and their TPD stability."""
    h_f, _ = enthalpy(T_f, p, 1.0)
    h_o, _ = enthalpy(T_ox, p, 0.0)
    out = []
    for Zm in np.linspace(0.0, 1.0, nZ):
        h = Zm * h_f + (1 - Zm) * h_o
        T = brentq(lambda TT: enthalpy(TT, p, Zm)[0] - h, 150.0, 1200.0, xtol=1e-6)
        _, rho = enthalpy(T, p, Zm)
        Xf = float(pr.mass_to_mole_fraction(np.array(Zm))[1])
        tpd = tpd_min(T, p, Xf) if 0.0 < Zm < 1.0 else 0.0
        out.append(dict(Y=Zm, X=Xf, T=T, rho=rho, tpd=tpd))
    return out
