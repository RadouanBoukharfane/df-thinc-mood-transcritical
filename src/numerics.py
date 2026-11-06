"""Primitive/conservative conversions and first-order HLLC (double flux)."""

import numpy as np

from thermo import get_T_from_p_rho, get_energy_from_T_rho


def primitives_to_conservatives(primitives):
    """Conservative variables from (rho, u, p) with the real EoS."""
    rho = primitives[0, :]
    u = primitives[1, :]
    p = primitives[2, :]

    T = get_T_from_p_rho(p, rho)

    conservatives = np.empty_like(primitives)
    conservatives[0, :] = rho
    conservatives[1, :] = rho * u
    conservatives[2, :] = rho * get_energy_from_T_rho(T, rho) + 0.5 * rho * u**2
    return conservatives


def update_primitives_df(W, gammaS, e0S):
    """Primitive variables from the frozen affine relation."""
    rho = W[0, :]
    u = W[1, :] / W[0, :]
    E = W[2, :]
    p = (gammaS - 1) * (E - rho * e0S - 0.5 * rho * u**2)

    V = np.zeros_like(W)
    V[0, :] = rho
    V[1, :] = u
    V[2, :] = p
    return V


def hllc_df(V_L, V_R, gammaS, e0S):
    """First-order HLLC flux, double flux."""

    def sign(s):
        return (s >= 0).astype(float) - (s <= 0).astype(float)

    u_L, u_R = V_L[1, :], V_R[1, :]
    rho_L, rho_R = V_L[0, :], V_R[0, :]
    p_L, p_R = V_L[2, :], V_R[2, :]

    a_L = np.sqrt(gammaS * p_L / rho_L)
    a_R = np.sqrt(gammaS * p_R / rho_R)

    E_L = p_L / (gammaS - 1) + rho_L * e0S + 0.5 * rho_L * u_L**2
    E_R = p_R / (gammaS - 1) + rho_R * e0S + 0.5 * rho_R * u_R**2

    f_L = np.vstack([rho_L * u_L, rho_L * u_L**2 + p_L, u_L * (E_L + p_L)])
    f_R = np.vstack([rho_R * u_R, rho_R * u_R**2 + p_R, u_R * (E_R + p_R)])

    W_L = np.vstack([rho_L, rho_L * u_L, E_L])
    W_R = np.vstack([rho_R, rho_R * u_R, E_R])

    s_L = u_L - a_L
    s_R = u_R + a_R

    sM = (p_L - p_R - rho_L * u_L * (s_L - u_L) + rho_R * u_R * (s_R - u_R)) / (
        rho_R * (s_R - u_R) - rho_L * (s_L - u_L)
    )
    pM = rho_R * (u_R - s_R) * (u_R - sM) + p_R

    WML = np.vstack([
        (s_L - u_L) / (s_L - sM) * rho_L,
        ((s_L - u_L) * rho_L * u_L + (pM - p_L)) / (s_L - sM),
        ((s_L - u_L) * E_L - p_L * u_L + pM * sM) / (s_L - sM),
    ])

    WMR = np.vstack([
        (s_R - u_R) / (s_R - sM) * rho_R,
        ((s_R - u_R) * rho_R * u_R + (pM - p_R)) / (s_R - sM),
        ((s_R - u_R) * E_R - p_R * u_R + pM * sM) / (s_R - sM),
    ])

    s_minus = np.minimum(0, s_L)
    s_plus = np.maximum(0, s_R)

    f_HLLC = (1 + sign(sM)) / 2 * (f_L + s_minus * (WML - W_L)) + \
             (1 - sign(sM)) / 2 * (f_R + s_plus * (WMR - W_R))

    return f_HLLC


def get_rhs_df(W, V, gammaS, e0S, dx):
    """First-order periodic residual, double flux."""
    Nx = V.shape[1]

    fa_L = np.hstack([V[:, [Nx - 1]], V[:, 0:Nx]])
    fa_R = np.hstack([V[:, 0:Nx], V[:, [0]]])

    flux_L = hllc_df(fa_L[:, 0:Nx], fa_R[:, 0:Nx], gammaS, e0S)
    flux_R = hllc_df(fa_L[:, 1:Nx + 1], fa_R[:, 1:Nx + 1], gammaS, e0S)

    RHS = -(flux_R - flux_L) / dx
    return RHS
