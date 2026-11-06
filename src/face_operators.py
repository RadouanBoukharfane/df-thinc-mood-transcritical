"""One-dimensional face fluxes with owner-dependent double-flux energies."""

import numpy as np
from dataclasses import dataclass

from thermo import get_sos


@dataclass
class FaceFluxes:
    """Shared mass/momentum fluxes and one energy flux per owner cell."""
    mass: np.ndarray
    momentum: np.ndarray
    energy_left_owner: np.ndarray
    energy_right_owner: np.ndarray
    scheme_tag: str

    def energy_conservation_error(self):
        return self.energy_right_owner - self.energy_left_owner


def flux_ho(V_face, gammaS_left, e0S_left, gammaS_right, e0S_right):
    """Central flux of the high-order face state."""
    rho, u, p = V_face[0, :], V_face[1, :], V_face[2, :]

    mass = rho * u
    momentum = rho * u**2 + p

    E_left = p / (gammaS_left - 1) + rho * e0S_left + 0.5 * rho * u**2
    E_right = p / (gammaS_right - 1) + rho * e0S_right + 0.5 * rho * u**2

    energy_left_owner = u * (E_left + p)
    energy_right_owner = u * (E_right + p)

    return FaceFluxes(mass, momentum, energy_left_owner, energy_right_owner, 'HO')


def _sign(s):
    return (s >= 0).astype(float) - (s <= 0).astype(float)


def flux_hllc_df(V_L, V_R, gammaS_left, e0S_left, gammaS_right, e0S_right):
    """HLLC flux, double-flux energies."""
    rho_L, rho_R = V_L[0, :], V_R[0, :]
    u_L, u_R = V_L[1, :], V_R[1, :]
    p_L, p_R = V_L[2, :], V_R[2, :]

    a_L = get_sos(V_L)
    a_R = get_sos(V_R)

    s_L = u_L - a_L
    s_R = u_R + a_R

    sM = (p_L - p_R - rho_L * u_L * (s_L - u_L) + rho_R * u_R * (s_R - u_R)) / (
        rho_R * (s_R - u_R) - rho_L * (s_L - u_L)
    )
    pM = rho_R * (u_R - s_R) * (u_R - sM) + p_R

    s_minus = np.minimum(0, s_L)
    s_plus = np.maximum(0, s_R)
    sgn = _sign(sM)

    f_mass_L = rho_L * u_L
    f_mass_R = rho_R * u_R
    f_mom_L = rho_L * u_L**2 + p_L
    f_mom_R = rho_R * u_R**2 + p_R

    rho_starL = (s_L - u_L) / (s_L - sM) * rho_L
    rho_starR = (s_R - u_R) / (s_R - sM) * rho_R
    mom_starL = ((s_L - u_L) * rho_L * u_L + (pM - p_L)) / (s_L - sM)
    mom_starR = ((s_R - u_R) * rho_R * u_R + (pM - p_R)) / (s_R - sM)

    mass = (1 + sgn) / 2 * (f_mass_L + s_minus * (rho_starL - rho_L)) + \
           (1 - sgn) / 2 * (f_mass_R + s_plus * (rho_starR - rho_R))
    momentum = (1 + sgn) / 2 * (f_mom_L + s_minus * (mom_starL - rho_L * u_L)) + \
               (1 - sgn) / 2 * (f_mom_R + s_plus * (mom_starR - rho_R * u_R))

    def energy_flux_for_owner(gammaS_o, e0S_o):
        E_L = p_L / (gammaS_o - 1) + rho_L * e0S_o + 0.5 * rho_L * u_L**2
        E_R = p_R / (gammaS_o - 1) + rho_R * e0S_o + 0.5 * rho_R * u_R**2

        f_E_L = u_L * (E_L + p_L)
        f_E_R = u_R * (E_R + p_R)

        E_starL = ((s_L - u_L) * E_L - p_L * u_L + pM * sM) / (s_L - sM)
        E_starR = ((s_R - u_R) * E_R - p_R * u_R + pM * sM) / (s_R - sM)

        f_E = (1 + sgn) / 2 * (f_E_L + s_minus * (E_starL - E_L)) + \
              (1 - sgn) / 2 * (f_E_R + s_plus * (E_starR - E_R))
        return f_E

    energy_left_owner = energy_flux_for_owner(gammaS_left, e0S_left)
    energy_right_owner = energy_flux_for_owner(gammaS_right, e0S_right)

    return FaceFluxes(mass, momentum, energy_left_owner, energy_right_owner, 'ENO_HLLC')


def flux_llf_df(V_L, V_R, gammaS_left, e0S_left, gammaS_right, e0S_right):
    """Local Lax--Friedrichs flux, double-flux energies."""
    rho_L, rho_R = V_L[0, :], V_R[0, :]
    u_L, u_R = V_L[1, :], V_R[1, :]
    p_L, p_R = V_L[2, :], V_R[2, :]

    a_L = get_sos(V_L)
    a_R = get_sos(V_R)
    alpha = np.maximum(np.abs(u_L) + a_L, np.abs(u_R) + a_R)

    f_mass_L = rho_L * u_L
    f_mass_R = rho_R * u_R
    f_mom_L = rho_L * u_L**2 + p_L
    f_mom_R = rho_R * u_R**2 + p_R

    mass = 0.5 * (f_mass_L + f_mass_R) - 0.5 * alpha * (rho_R - rho_L)
    momentum = 0.5 * (f_mom_L + f_mom_R) - 0.5 * alpha * (rho_R * u_R - rho_L * u_L)

    def energy_flux_for_owner(gammaS_o, e0S_o):
        E_L = p_L / (gammaS_o - 1) + rho_L * e0S_o + 0.5 * rho_L * u_L**2
        E_R = p_R / (gammaS_o - 1) + rho_R * e0S_o + 0.5 * rho_R * u_R**2
        f_E_L = u_L * (E_L + p_L)
        f_E_R = u_R * (E_R + p_R)
        return 0.5 * (f_E_L + f_E_R) - 0.5 * alpha * (E_R - E_L)

    energy_left_owner = energy_flux_for_owner(gammaS_left, e0S_left)
    energy_right_owner = energy_flux_for_owner(gammaS_right, e0S_right)

    return FaceFluxes(mass, momentum, energy_left_owner, energy_right_owner, 'ENO_LF')


def flux_ho_fc(V_face):
    """Central flux, fully conservative energy."""
    from thermo import get_gammaS_e0S

    rho, u, p = V_face[0, :], V_face[1, :], V_face[2, :]

    mass = rho * u
    momentum = rho * u**2 + p

    gammaS, e0S = get_gammaS_e0S(V_face)
    E = p / (gammaS - 1) + rho * e0S + 0.5 * rho * u**2
    energy = u * (E + p)

    return FaceFluxes(mass, momentum, energy.copy(), energy.copy(), 'HO_FC')


def flux_hllc_fc(V_L, V_R):
    """HLLC flux, fully conservative energy."""
    from thermo import get_gammaS_e0S

    rho_L, rho_R = V_L[0, :], V_R[0, :]
    u_L, u_R = V_L[1, :], V_R[1, :]
    p_L, p_R = V_L[2, :], V_R[2, :]

    a_L = get_sos(V_L)
    a_R = get_sos(V_R)

    s_L = u_L - a_L
    s_R = u_R + a_R

    sM = (p_L - p_R - rho_L * u_L * (s_L - u_L) + rho_R * u_R * (s_R - u_R)) / (
        rho_R * (s_R - u_R) - rho_L * (s_L - u_L)
    )
    pM = rho_R * (u_R - s_R) * (u_R - sM) + p_R

    s_minus = np.minimum(0, s_L)
    s_plus = np.maximum(0, s_R)
    sgn = _sign(sM)

    f_mass_L = rho_L * u_L
    f_mass_R = rho_R * u_R
    f_mom_L = rho_L * u_L**2 + p_L
    f_mom_R = rho_R * u_R**2 + p_R

    rho_starL = (s_L - u_L) / (s_L - sM) * rho_L
    rho_starR = (s_R - u_R) / (s_R - sM) * rho_R
    mom_starL = ((s_L - u_L) * rho_L * u_L + (pM - p_L)) / (s_L - sM)
    mom_starR = ((s_R - u_R) * rho_R * u_R + (pM - p_R)) / (s_R - sM)

    mass = (1 + sgn) / 2 * (f_mass_L + s_minus * (rho_starL - rho_L)) + \
           (1 - sgn) / 2 * (f_mass_R + s_plus * (rho_starR - rho_R))
    momentum = (1 + sgn) / 2 * (f_mom_L + s_minus * (mom_starL - rho_L * u_L)) + \
               (1 - sgn) / 2 * (f_mom_R + s_plus * (mom_starR - rho_R * u_R))

    gammaS_L, e0S_L = get_gammaS_e0S(V_L)
    gammaS_R, e0S_R = get_gammaS_e0S(V_R)

    E_L = p_L / (gammaS_L - 1) + rho_L * e0S_L + 0.5 * rho_L * u_L**2
    E_R = p_R / (gammaS_R - 1) + rho_R * e0S_R + 0.5 * rho_R * u_R**2

    f_E_L = u_L * (E_L + p_L)
    f_E_R = u_R * (E_R + p_R)

    E_starL = ((s_L - u_L) * E_L - p_L * u_L + pM * sM) / (s_L - sM)
    E_starR = ((s_R - u_R) * E_R - p_R * u_R + pM * sM) / (s_R - sM)

    energy = (1 + sgn) / 2 * (f_E_L + s_minus * (E_starL - E_L)) + \
             (1 - sgn) / 2 * (f_E_R + s_plus * (E_starR - E_R))

    return FaceFluxes(mass, momentum, energy.copy(), energy.copy(), 'ENO_HLLC_FC')


def flux_llf_fc(V_L, V_R):
    """Local Lax--Friedrichs flux, fully conservative energy."""
    from thermo import get_gammaS_e0S

    rho_L, rho_R = V_L[0, :], V_R[0, :]
    u_L, u_R = V_L[1, :], V_R[1, :]
    p_L, p_R = V_L[2, :], V_R[2, :]

    a_L = get_sos(V_L)
    a_R = get_sos(V_R)
    alpha = np.maximum(np.abs(u_L) + a_L, np.abs(u_R) + a_R)

    f_mass_L = rho_L * u_L
    f_mass_R = rho_R * u_R
    f_mom_L = rho_L * u_L**2 + p_L
    f_mom_R = rho_R * u_R**2 + p_R

    mass = 0.5 * (f_mass_L + f_mass_R) - 0.5 * alpha * (rho_R - rho_L)
    momentum = 0.5 * (f_mom_L + f_mom_R) - 0.5 * alpha * (rho_R * u_R - rho_L * u_L)

    gammaS_L, e0S_L = get_gammaS_e0S(V_L)
    gammaS_R, e0S_R = get_gammaS_e0S(V_R)
    E_L = p_L / (gammaS_L - 1) + rho_L * e0S_L + 0.5 * rho_L * u_L**2
    E_R = p_R / (gammaS_R - 1) + rho_R * e0S_R + 0.5 * rho_R * u_R**2
    f_E_L = u_L * (E_L + p_L)
    f_E_R = u_R * (E_R + p_R)
    energy = 0.5 * (f_E_L + f_E_R) - 0.5 * alpha * (E_R - E_L)

    return FaceFluxes(mass, momentum, energy.copy(), energy.copy(), 'ENO_LF_FC')


flux_eno_hllc = flux_hllc_df
flux_eno_lf = flux_llf_df
flux_eno_hllc_fc = flux_hllc_fc
flux_eno_lf_fc = flux_llf_fc
