"""Primitive/conservative conversions and first-order HLLC (fully conservative)."""

import numpy as np

from thermo import get_gammaS_e0S, get_T_from_e_rho, get_p_from_T_rho


def primitives_to_conservatives_fc(primitives):
    """Conservative variables from (rho, u, p)."""
    from numerics import primitives_to_conservatives
    return primitives_to_conservatives(primitives)


def update_primitives_fc(W):
    """Primitive variables by inversion of the real EoS."""
    rho = W[0, :]
    u = W[1, :] / W[0, :]
    E = W[2, :]
    e = E / rho - 0.5 * u**2

    T = get_T_from_e_rho(e, rho)
    p = get_p_from_T_rho(T, rho)

    V = np.zeros_like(W)
    V[0, :] = rho
    V[1, :] = u
    V[2, :] = p
    return V


def _sign(s):
    return (s >= 0).astype(float) - (s <= 0).astype(float)


def hllc_fc(V_L, V_R):
    """First-order HLLC flux, fully conservative."""
    rho_L, rho_R = V_L[0, :], V_R[0, :]
    u_L, u_R = V_L[1, :], V_R[1, :]
    p_L, p_R = V_L[2, :], V_R[2, :]

    gammaS_L, e0S_L = get_gammaS_e0S(V_L)
    gammaS_R, e0S_R = get_gammaS_e0S(V_R)

    a_L = np.sqrt(gammaS_L * p_L / rho_L)
    a_R = np.sqrt(gammaS_R * p_R / rho_R)

    E_L = p_L / (gammaS_L - 1) + rho_L * e0S_L + 0.5 * rho_L * u_L**2
    E_R = p_R / (gammaS_R - 1) + rho_R * e0S_R + 0.5 * rho_R * u_R**2

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
    sgn = _sign(sM)

    f_HLLC = (1 + sgn) / 2 * (f_L + s_minus * (WML - W_L)) + \
             (1 - sgn) / 2 * (f_R + s_plus * (WMR - W_R))

    return f_HLLC


def get_rhs_fc(V, dx):
    """First-order periodic residual, fully conservative."""
    V_m = np.roll(V, shift=1, axis=1)
    V_p = np.roll(V, shift=-1, axis=1)

    flux_left_face = hllc_fc(V_m, V)
    flux_right_face = hllc_fc(V, V_p)

    RHS = -(flux_right_face - flux_left_face) / dx
    return RHS
