"""Surrogate double-flux entropy and discrete entropy-minimum criterion."""

import numpy as np

_MW_N2 = 28.0134e-3
R_N2 = 8.314 / _MW_N2


def entropy_double_flux(V, gamma_effective, R=R_N2):
    """Surrogate entropy s = ln p - gamma* ln(rho R) with frozen gamma*."""
    rho = V[0, :] if V.ndim == 2 else V[0]
    p = V[2, :] if V.ndim == 2 else V[2]

    s = np.log(p) - gamma_effective * np.log(rho * R)
    return s


def neighbor_entropy_bound(s_prev):
    """Minimum entropy over the three-cell stencil."""
    s_left = np.roll(s_prev, 1)
    s_right = np.roll(s_prev, -1)
    return np.minimum(np.minimum(s_left, s_prev), s_right)


def entropy_minimum_check(s_candidate, s_prev, atol=1.0e-12, rtol=1.0e-10):
    """Cells violating the discrete entropy-minimum principle."""
    s_min = neighbor_entropy_bound(s_prev)
    tol = atol + rtol * np.maximum(1.0, np.abs(s_min))
    invalid = ~np.isfinite(s_candidate)
    troubled = invalid | (s_candidate < (s_min - tol))
    return troubled
