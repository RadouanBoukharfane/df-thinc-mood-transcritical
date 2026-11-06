"""Peng--Robinson thermodynamics of pure nitrogen (NASA ideal-gas part)."""

import numpy as np

NASA_N2 = np.array([
    3.531005280e+00,
    -1.236609870e-04,
    -5.029994370e-07,
    2.435306120e-09,
    -1.408812350e-12,
    -1.046976280e+03,
    2.967474680e+00,
])


def get_thermo(T):
    """PR parameters a, b, R and temperature derivatives of a."""
    T = np.asarray(T, dtype=float)

    MW = 28.0134e-3
    Tc = 126.19
    pc = 3.3958e+6
    omega = 0.03720

    c = 0.37464 + 1.54226 * omega - 0.26992 * omega**2

    R = 8.314 / MW
    a = 0.457236 * (R * Tc) ** 2 / pc * (1 + c * (1 - np.sqrt(T / Tc))) ** 2
    b = 0.077796 * R * Tc / pc
    G = c * np.sqrt(T / Tc) / (1 + c * (1 - np.sqrt(T / Tc)))
    dadT = -1.0 / T * a * G
    d2adT2 = 0.457236 * R**2 / T / 2 * c * (1 + c) * Tc / pc * np.sqrt(Tc / T)

    return a, b, R, dadT, d2adT2


def get_p_from_T_rho(T, rho):
    """Pressure from (T, rho)."""
    T = np.asarray(T, dtype=float)
    rho = np.asarray(rho, dtype=float)

    a, b, R, dadT, d2adT2 = get_thermo(T)
    v = 1.0 / rho
    p = R * T / (v - b) - a / (v**2 + 2 * v * b - b**2)
    return p


def get_energy_from_T_rho(T, rho):
    """Specific internal energy from (T, rho)."""
    T = np.asarray(T, dtype=float)
    rho = np.asarray(rho, dtype=float)
    an = NASA_N2

    a, b, R, dadT, d2adT2 = get_thermo(T)
    h_ideal = R * T * (
        an[0] + an[1] * T / 2 + an[2] * T**2 / 3 + an[3] * T**3 / 4
        + an[4] * T**4 / 5 + an[5] / T
    )

    v = 1.0 / rho
    K1 = 1 / np.sqrt(8) / b * np.log((v + (1 - np.sqrt(2)) * b) / (v + (1 + np.sqrt(2)) * b))
    dep = (a - T * dadT) * K1

    e = h_ideal - R * T + dep
    return e


def get_T_from_p_rho(p, rho):
    """Temperature from (p, rho) by Newton iteration."""
    p = np.asarray(p, dtype=float)
    rho = np.asarray(rho, dtype=float)

    CRIT = 1.0e-6

    v = 1.0 / rho
    T = np.full_like(rho, 300.0)
    p_n = get_p_from_T_rho(T, rho)
    diff = p_n - p

    while np.max(np.abs(diff)) > CRIT:
        a, b, R, dadT, d2adT2 = get_thermo(T)
        dpdT = R / (v - b) - 1.0 / (v**2 + 2 * v * b - b**2) * dadT
        T = T - diff / dpdT
        p_n = get_p_from_T_rho(T, rho)
        diff = p_n - p

    return T


def get_rho_from_T_p(T, p):
    """Density from (T, p) (Cardano root)."""
    T = np.asarray(T, dtype=float)
    p = np.asarray(p, dtype=float)

    a, b, R, dadT, d2adT2 = get_thermo(T)

    a0 = p * b**3 + b**2 * R * T - a * b
    a1 = -3 * p * b**2 - 2 * b * R * T + a
    a2 = p * b - R * T
    a3 = p

    l = a2 / a3
    m = a1 / a3
    n = a0 / a3
    A = 1.0 / 3 * (3 * m - l**2)
    B = 1.0 / 27 * (2 * l**3 - 9 * l * m + 27 * n)
    D = A**3 / 27 + B**2 / 4

    def signed_cbrt(x):
        return np.sign(x) * np.abs(x) ** (1.0 / 3)

    M = signed_cbrt(-B / 2 + np.sqrt(D))
    N = signed_cbrt(-B / 2 - np.sqrt(D))

    v = M + N - l / 3
    rho = 1.0 / v
    return rho


def get_sos(V):
    """Sound speed of primitive states (rho, u, p)."""
    an = NASA_N2

    rho = V[0, :]
    p = V[2, :]
    v = 1.0 / rho

    T = get_T_from_p_rho(p, rho)
    a, b, R, dadT, d2adT2 = get_thermo(T)

    cp_ideal = R * (an[0] + an[1] * T + an[2] * T**2 + an[3] * T**3 + an[4] * T**4)
    cv_ideal = cp_ideal - R

    dpdT = R / (v - b) - dadT / (v**2 + 2 * v * b - b**2)
    dpdv = -R * T / (v - b) ** 2 * (
        1 - 2 * a * ((R * T * (v + b) * ((v**2 + 2 * v * b - b**2) / (v**2 - b**2)) ** 2) ** (-1))
    )
    K1 = 1 / np.sqrt(8) / b * np.log((v + (1 - np.sqrt(2)) * b) / (v + (1 + np.sqrt(2)) * b))
    cp = cv_ideal - T * dpdT**2 / dpdv - K1 * T * d2adT2

    kT = -1.0 / v / dpdv
    av = -dpdT / v / dpdv
    ks = kT - v * T * av**2 / cp

    sos = np.sqrt(1.0 / rho / ks)
    return sos


def get_gammaS_e0S(V):
    """Double-flux parameters gamma* = rho c^2 / p and e0*."""
    rho = V[0, :]
    p = V[2, :]

    T = get_T_from_p_rho(p, rho)
    e = get_energy_from_T_rho(T, rho)
    sos = get_sos(V)

    gammaS = sos**2 * rho / p
    e0S = e - p / rho / (gammaS - 1)
    return gammaS, e0S


def get_cv_from_T_rho(T, rho):
    """Isochoric heat capacity."""
    T = np.asarray(T, dtype=float)
    rho = np.asarray(rho, dtype=float)
    an = NASA_N2

    a, b, R, dadT, d2adT2 = get_thermo(T)
    cp_ideal = R * (an[0] + an[1] * T + an[2] * T**2 + an[3] * T**3 + an[4] * T**4)
    cv_ideal = cp_ideal - R

    v = 1.0 / rho
    K1 = 1 / np.sqrt(8) / b * np.log((v + (1 - np.sqrt(2)) * b) / (v + (1 + np.sqrt(2)) * b))

    cv = cv_ideal - K1 * T * d2adT2
    return cv


def get_T_from_e_rho(e, rho, T_init=300.0, max_iter=100, rtol=1e-10):
    """Temperature from (e, rho) (fully conservative formulation only)."""
    e = np.asarray(e, dtype=float)
    rho = np.asarray(rho, dtype=float)

    if np.any(~np.isfinite(rho)) or np.any(rho <= 0.0):
        raise ValueError(
            f"invalid density passed to get_T_from_e_rho: "
            f"min(rho)={np.min(rho) if rho.size else float('nan')}"
        )
    if np.any(~np.isfinite(e)):
        raise ValueError("non-finite internal energy passed to get_T_from_e_rho")

    T = np.full_like(rho, T_init)
    e_n = get_energy_from_T_rho(T, rho)
    diff = e_n - e

    scale = np.maximum(np.abs(e), 1.0)
    it = 0
    while np.max(np.abs(diff) / scale) > rtol and it < max_iter:
        cv = get_cv_from_T_rho(T, rho)

        if np.any(~np.isfinite(cv)):
            raise RuntimeError("non-finite cv during the inversion T(e, rho)")
        if np.any(cv <= 0.0):
            raise RuntimeError(
                f"invalid thermodynamic state during the inversion T(e, rho): "
                f"min(cv)={np.min(cv):.6e} (cv must be positive)"
            )
        cv_safe = np.maximum(cv, 1.0e-6)

        dT = -diff / cv_safe
        dT = np.clip(dT, -0.25 * T, 0.25 * T)
        T = T + dT
        T = np.maximum(T, 1.0)
        e_n = get_energy_from_T_rho(T, rho)
        diff = e_n - e
        it += 1

    residual = np.max(np.abs(diff) / scale)
    if residual > rtol:
        raise RuntimeError(
            f"inversion T(e, rho) not converged after {max_iter} iterations : "
            f"residu relatif max = {residual:.3e} (tolerance = {rtol:.1e})"
        )

    return T
