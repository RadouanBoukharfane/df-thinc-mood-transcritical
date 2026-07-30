"""Operateurs de face Euler 2D pour le modele double-flux.

Les solveurs sont ecrits dans un repere normal/transverse, puis remappes vers
les composantes cartesiennes. Ils sont donc strictement rotationnels entre
les directions x et y.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from thermo import get_sos
from numerics_2d import modified_total_energy_2d

Direction = Literal["x", "y"]


@dataclass(frozen=True)
class FaceFluxes2D:
    mass: np.ndarray
    momentum_x: np.ndarray
    momentum_y: np.ndarray
    energy_left_owner: np.ndarray
    energy_right_owner: np.ndarray
    scheme_tag: str

    def conservative_left_owner(self) -> np.ndarray:
        return np.stack(
            [self.mass, self.momentum_x, self.momentum_y, self.energy_left_owner],
            axis=0,
        )

    def conservative_right_owner(self) -> np.ndarray:
        return np.stack(
            [self.mass, self.momentum_x, self.momentum_y, self.energy_right_owner],
            axis=0,
        )


def _validate(V: np.ndarray, name: str) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] != 4:
        raise ValueError(f"{name} doit avoir la forme (4, Nf)")
    if not np.all(np.isfinite(V)):
        raise ValueError(f"{name} contient des valeurs non finies")
    if np.any(V[0] <= 0.0) or np.any(V[3] <= 0.0):
        raise ValueError(f"{name}: rho et p doivent etre positifs")
    return V


def _normal_transverse(V: np.ndarray, direction: Direction):
    rho, u, v, p = V
    if direction == "x":
        return rho, u, v, p
    if direction == "y":
        return rho, v, u, p
    raise ValueError("direction doit valoir 'x' ou 'y'")


def _map_momenta(normal_flux, transverse_flux, direction: Direction):
    if direction == "x":
        return normal_flux, transverse_flux
    return transverse_flux, normal_flux


def _thermo_triplet(V: np.ndarray, direction: Direction) -> np.ndarray:
    rho, un, _, p = _normal_transverse(V, direction)
    return np.vstack([rho, un, p])


def flux_ho_df_2d(
    V_face: np.ndarray,
    gammaS_left: np.ndarray,
    e0S_left: np.ndarray,
    gammaS_right: np.ndarray,
    e0S_right: np.ndarray,
    *,
    direction: Direction,
) -> FaceFluxes2D:
    """Flux physique central sur un etat unique de face."""
    V_face = _validate(V_face, "V_face")
    rho, un, wt, p = _normal_transverse(V_face, direction)

    mass = rho * un
    mom_n = rho * un * un + p
    mom_t = rho * un * wt
    mom_x, mom_y = _map_momenta(mom_n, mom_t, direction)

    E_left = modified_total_energy_2d(V_face, gammaS_left, e0S_left)
    E_right = modified_total_energy_2d(V_face, gammaS_right, e0S_right)
    energy_left = un * (E_left + p)
    energy_right = un * (E_right + p)

    return FaceFluxes2D(mass, mom_x, mom_y, energy_left, energy_right, f"HO_DF_{direction.upper()}")


def flux_hllc_df_2d(
    V_L: np.ndarray,
    V_R: np.ndarray,
    gammaS_left: np.ndarray,
    e0S_left: np.ndarray,
    gammaS_right: np.ndarray,
    e0S_right: np.ndarray,
    *,
    direction: Direction,
) -> FaceFluxes2D:
    """HLLC 2D rotationnel avec energie double-flux owner-dependante.

    La vitesse transverse reste constante a travers chacune des ondes
    acoustiques HLLC. Le contact transporte donc la composante transverse
    avec le flux massique du meme probleme de Riemann.
    """
    V_L = _validate(V_L, "V_L")
    V_R = _validate(V_R, "V_R")
    if V_L.shape != V_R.shape:
        raise ValueError("V_L et V_R doivent avoir la meme forme")

    rho_L, un_L, wt_L, p_L = _normal_transverse(V_L, direction)
    rho_R, un_R, wt_R, p_R = _normal_transverse(V_R, direction)

    a_L = get_sos(_thermo_triplet(V_L, direction))
    a_R = get_sos(_thermo_triplet(V_R, direction))
    s_L = un_L - a_L
    s_R = un_R + a_R

    denom = rho_R * (s_R - un_R) - rho_L * (s_L - un_L)
    s_M = (
        p_L - p_R
        - rho_L * un_L * (s_L - un_L)
        + rho_R * un_R * (s_R - un_R)
    ) / denom
    p_M = rho_R * (un_R - s_R) * (un_R - s_M) + p_R

    s_minus = np.minimum(0.0, s_L)
    s_plus = np.maximum(0.0, s_R)
    choose_left = s_M >= 0.0

    rho_star_L = rho_L * (s_L - un_L) / (s_L - s_M)
    rho_star_R = rho_R * (s_R - un_R) / (s_R - s_M)
    momn_star_L = ((s_L - un_L) * rho_L * un_L + (p_M - p_L)) / (s_L - s_M)
    momn_star_R = ((s_R - un_R) * rho_R * un_R + (p_M - p_R)) / (s_R - s_M)
    momt_star_L = rho_star_L * wt_L
    momt_star_R = rho_star_R * wt_R

    f_mass_L = rho_L * un_L
    f_mass_R = rho_R * un_R
    f_momn_L = rho_L * un_L * un_L + p_L
    f_momn_R = rho_R * un_R * un_R + p_R
    f_momt_L = rho_L * un_L * wt_L
    f_momt_R = rho_R * un_R * wt_R

    mass_L_branch = f_mass_L + s_minus * (rho_star_L - rho_L)
    mass_R_branch = f_mass_R + s_plus * (rho_star_R - rho_R)
    momn_L_branch = f_momn_L + s_minus * (momn_star_L - rho_L * un_L)
    momn_R_branch = f_momn_R + s_plus * (momn_star_R - rho_R * un_R)
    momt_L_branch = f_momt_L + s_minus * (momt_star_L - rho_L * wt_L)
    momt_R_branch = f_momt_R + s_plus * (momt_star_R - rho_R * wt_R)

    mass = np.where(choose_left, mass_L_branch, mass_R_branch)
    mom_n = np.where(choose_left, momn_L_branch, momn_R_branch)
    mom_t = np.where(choose_left, momt_L_branch, momt_R_branch)
    mom_x, mom_y = _map_momenta(mom_n, mom_t, direction)

    def energy_for_owner(gamma_o, e0_o):
        E_L = modified_total_energy_2d(V_L, gamma_o, e0_o)
        E_R = modified_total_energy_2d(V_R, gamma_o, e0_o)
        f_E_L = un_L * (E_L + p_L)
        f_E_R = un_R * (E_R + p_R)
        E_star_L = ((s_L - un_L) * E_L - p_L * un_L + p_M * s_M) / (s_L - s_M)
        E_star_R = ((s_R - un_R) * E_R - p_R * un_R + p_M * s_M) / (s_R - s_M)
        left_branch = f_E_L + s_minus * (E_star_L - E_L)
        right_branch = f_E_R + s_plus * (E_star_R - E_R)
        return np.where(choose_left, left_branch, right_branch)

    energy_left = energy_for_owner(gammaS_left, e0S_left)
    energy_right = energy_for_owner(gammaS_right, e0S_right)
    return FaceFluxes2D(mass, mom_x, mom_y, energy_left, energy_right, f"HLLC_DF_{direction.upper()}")


def flux_llf_df_2d(
    V_L: np.ndarray,
    V_R: np.ndarray,
    gammaS_left: np.ndarray,
    e0S_left: np.ndarray,
    gammaS_right: np.ndarray,
    e0S_right: np.ndarray,
    *,
    direction: Direction,
) -> FaceFluxes2D:
    """Flux LLF 2D terminal, rotationnel et owner-dependant en energie."""
    V_L = _validate(V_L, "V_L")
    V_R = _validate(V_R, "V_R")
    if V_L.shape != V_R.shape:
        raise ValueError("V_L et V_R doivent avoir la meme forme")

    rho_L, un_L, wt_L, p_L = _normal_transverse(V_L, direction)
    rho_R, un_R, wt_R, p_R = _normal_transverse(V_R, direction)
    a_L = get_sos(_thermo_triplet(V_L, direction))
    a_R = get_sos(_thermo_triplet(V_R, direction))
    alpha = np.maximum(np.abs(un_L) + a_L, np.abs(un_R) + a_R)

    mass = 0.5 * (rho_L * un_L + rho_R * un_R) - 0.5 * alpha * (rho_R - rho_L)
    mom_n = 0.5 * (
        rho_L * un_L * un_L + p_L + rho_R * un_R * un_R + p_R
    ) - 0.5 * alpha * (rho_R * un_R - rho_L * un_L)
    mom_t = 0.5 * (
        rho_L * un_L * wt_L + rho_R * un_R * wt_R
    ) - 0.5 * alpha * (rho_R * wt_R - rho_L * wt_L)
    mom_x, mom_y = _map_momenta(mom_n, mom_t, direction)

    def energy_for_owner(gamma_o, e0_o):
        E_L = modified_total_energy_2d(V_L, gamma_o, e0_o)
        E_R = modified_total_energy_2d(V_R, gamma_o, e0_o)
        f_E_L = un_L * (E_L + p_L)
        f_E_R = un_R * (E_R + p_R)
        return 0.5 * (f_E_L + f_E_R) - 0.5 * alpha * (E_R - E_L)

    energy_left = energy_for_owner(gammaS_left, e0S_left)
    energy_right = energy_for_owner(gammaS_right, e0S_right)
    return FaceFluxes2D(mass, mom_x, mom_y, energy_left, energy_right, f"LLF_DF_{direction.upper()}")
