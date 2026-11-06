"""Hybrid face fluxes of the high-order MOOD levels (central / ENO2-HLLC / THINC-BVD)."""

import numpy as np
from dataclasses import dataclass

from reconstruction import reconstruct_high_order, reconstruct_high_order_limited
from eno2 import reconstruct_eno2_left, reconstruct_eno2_right
from thinc import reconstruct_thinc
from bvd import eno_face_states_to_cell_boundaries, compare_bvd_candidates
from sensor import rs_face_mask_direct, rs_face_mask_from_cells, classify_contact_shock
from face_operators import flux_ho, flux_hllc_df, flux_ho_fc, flux_hllc_fc


@dataclass(frozen=True)
class ReconstructionCandidates:
    """High-order, ENO2, and THINC candidates of one state."""

    V_face_ho: np.ndarray
    V_L_eno: np.ndarray
    V_R_eno: np.ndarray
    rho_thinc_left_cell: np.ndarray
    rho_thinc_right_cell: np.ndarray
    limited_mask: object

    @property
    def rho_L_thinc_face(self):
        return self.rho_thinc_right_cell

    @property
    def rho_R_thinc_face(self):
        return np.roll(self.rho_thinc_left_cell, -1)

    @property
    def rho_eno_left_cell(self):
        q_left_cell, _ = eno_face_states_to_cell_boundaries(
            self.V_L_eno[0, :], self.V_R_eno[0, :]
        )
        return q_left_cell

    @property
    def rho_eno_right_cell(self):
        _, q_right_cell = eno_face_states_to_cell_boundaries(
            self.V_L_eno[0, :], self.V_R_eno[0, :]
        )
        return q_right_cell


def reconstruct_candidates(V, limit_ho=False, thinc_beta=1.6, thinc_eps=1.0e-14):
    """Compute all reconstruction candidates."""
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] < 3:
        raise ValueError(
            "V must be a 2-D array of shape (Nvars, Nx) with at least "
            "the rows (rho, u, p)"
        )

    if limit_ho:
        V_face_centered = reconstruct_high_order(V)
        V_face_ho = reconstruct_high_order_limited(V)
        limited_mask = np.any(np.abs(V_face_ho - V_face_centered) > 1e-14, axis=0)
    else:
        V_face_ho = reconstruct_high_order(V)
        limited_mask = None

    V_L_eno = reconstruct_eno2_left(V)
    V_R_eno = reconstruct_eno2_right(V)

    rho_thinc_left_cell, rho_thinc_right_cell = reconstruct_thinc(
        V[0, :], beta=thinc_beta, eps=thinc_eps
    )

    return ReconstructionCandidates(
        V_face_ho=V_face_ho,
        V_L_eno=V_L_eno,
        V_R_eno=V_R_eno,
        rho_thinc_left_cell=rho_thinc_left_cell,
        rho_thinc_right_cell=rho_thinc_right_cell,
        limited_mask=limited_mask,
    )


def diagnose_density_bvd(
    V,
    limit_ho=False,
    thinc_beta=1.6,
    thinc_eps=1.0e-14,
    bvd_atol=1.0e-14,
    bvd_rtol=1.0e-12,
):
    """BVD comparison of ENO2 and THINC for density."""
    candidates = reconstruct_candidates(
        V,
        limit_ho=limit_ho,
        thinc_beta=thinc_beta,
        thinc_eps=thinc_eps,
    )
    diagnostics = compare_bvd_candidates(
        candidates.rho_eno_left_cell,
        candidates.rho_eno_right_cell,
        candidates.rho_thinc_left_cell,
        candidates.rho_thinc_right_cell,
        atol=bvd_atol,
        rtol=bvd_rtol,
    )
    return candidates, diagnostics


def diagnose_density_bvd_with_wave_sensor(
    V,
    limit_ho=False,
    thinc_beta=1.6,
    thinc_eps=1.0e-14,
    bvd_atol=1.0e-14,
    bvd_rtol=1.0e-12,
    **sensor_options,
):
    """BVD selection restricted to contact cells."""
    candidates, bvd = diagnose_density_bvd(
        V,
        limit_ho=limit_ho,
        thinc_beta=thinc_beta,
        thinc_eps=thinc_eps,
        bvd_atol=bvd_atol,
        bvd_rtol=bvd_rtol,
    )
    waves = classify_contact_shock(V, **sensor_options)
    use_thinc_cell = bvd.use_thinc_cell & waves.allow_thinc_cell
    return candidates, bvd, waves, use_thinc_cell


def assemble_density_bvd_face_states(candidates, use_thinc_cell):
    """Riemann states with THINC density traces in selected cells."""
    use_thinc_cell = np.asarray(use_thinc_cell, dtype=bool)
    nx = candidates.V_L_eno.shape[1]
    if use_thinc_cell.shape != (nx,):
        raise ValueError("use_thinc_cell must have shape (Nx,)")

    rho_left_cell = np.where(
        use_thinc_cell,
        candidates.rho_thinc_left_cell,
        candidates.rho_eno_left_cell,
    )
    rho_right_cell = np.where(
        use_thinc_cell,
        candidates.rho_thinc_right_cell,
        candidates.rho_eno_right_cell,
    )

    V_L = candidates.V_L_eno.copy()
    V_R = candidates.V_R_eno.copy()
    V_L[0, :] = rho_right_cell
    V_R[0, :] = np.roll(rho_left_cell, -1)

    thinc_face = use_thinc_cell | np.roll(use_thinc_cell, -1)
    return V_L, V_R, thinc_face

def _reconstruct_all(V, limit_ho=False):
    candidates = reconstruct_candidates(V, limit_ho=limit_ho)
    return (
        candidates.V_face_ho,
        candidates.V_L_eno,
        candidates.V_R_eno,
        candidates.limited_mask,
    )


def _assemble_rhs_fc(mass, momentum, energy, dx):
    mass_R, mass_L = mass, np.roll(mass, 1)
    momentum_R, momentum_L = momentum, np.roll(momentum, 1)
    energy_R, energy_L = energy, np.roll(energy, 1)
    Nx = len(mass)
    RHS = np.zeros((3, Nx))
    RHS[0, :] = -(mass_R - mass_L) / dx
    RHS[1, :] = -(momentum_R - momentum_L) / dx
    RHS[2, :] = -(energy_R - energy_L) / dx
    return RHS


def _assemble_rhs_df(mass, momentum, energy_left, energy_right, dx):
    mass_R, mass_L = mass, np.roll(mass, 1)
    momentum_R, momentum_L = momentum, np.roll(momentum, 1)
    energy_R = energy_left
    energy_L = np.roll(energy_right, 1)
    Nx = len(mass)
    RHS = np.zeros((3, Nx))
    RHS[0, :] = -(mass_R - mass_L) / dx
    RHS[1, :] = -(momentum_R - momentum_L) / dx
    RHS[2, :] = -(energy_R - energy_L) / dx
    return RHS


def get_hybrid_face_fluxes_fc(V, dx, zeta=0.2, mask_variant='direct', limit_ho=False):
    """Hybrid face fluxes, fully conservative formulation."""
    rho = V[0, :]
    Nx = V.shape[1]
    V_face_ho, V_L_eno, V_R_eno, limited_mask = _reconstruct_all(V, limit_ho=limit_ho)

    if mask_variant == 'direct':
        mask, _ = rs_face_mask_direct(rho, zeta)
    else:
        mask = rs_face_mask_from_cells(rho, zeta)

    idx_ho = ~mask
    idx_eno = mask

    mass = np.empty(Nx)
    momentum = np.empty(Nx)
    energy = np.empty(Nx)

    if idx_ho.any():
        ff_ho = flux_ho_fc(V_face_ho[:, idx_ho])
        mass[idx_ho] = ff_ho.mass
        momentum[idx_ho] = ff_ho.momentum
        energy[idx_ho] = ff_ho.energy_left_owner

    if idx_eno.any():
        ff_eno = flux_hllc_fc(V_L_eno[:, idx_eno], V_R_eno[:, idx_eno])
        mass[idx_eno] = ff_eno.mass
        momentum[idx_eno] = ff_eno.momentum
        energy[idx_eno] = ff_eno.energy_left_owner

    return mass, momentum, energy, mask, V_L_eno, V_R_eno, limited_mask


def get_hybrid_face_fluxes_df(
    V,
    gammaS_frozen,
    e0S_frozen,
    dx,
    zeta=0.2,
    mask_variant='direct',
    limit_ho=False,
    use_thinc_bvd=False,
    thinc_beta=1.6,
    thinc_eps=1.0e-14,
    bvd_atol=1.0e-14,
    bvd_rtol=1.0e-12,
    sensor_options=None,
):
    """Hybrid face fluxes, double-flux formulation."""
    rho = V[0, :]
    Nx = V.shape[1]
    candidates = reconstruct_candidates(
        V,
        limit_ho=limit_ho,
        thinc_beta=thinc_beta,
        thinc_eps=thinc_eps,
    )
    V_face_ho = candidates.V_face_ho
    V_L_eno = candidates.V_L_eno
    V_R_eno = candidates.V_R_eno
    limited_mask = candidates.limited_mask

    if mask_variant == 'direct':
        mask_rs, _ = rs_face_mask_direct(rho, zeta)
    else:
        mask_rs = rs_face_mask_from_cells(rho, zeta)

    use_thinc_cell = np.zeros(Nx, dtype=bool)
    thinc_face_mask = np.zeros(Nx, dtype=bool)
    V_L_riemann = V_L_eno
    V_R_riemann = V_R_eno

    if use_thinc_bvd:
        bvd = compare_bvd_candidates(
            candidates.rho_eno_left_cell,
            candidates.rho_eno_right_cell,
            candidates.rho_thinc_left_cell,
            candidates.rho_thinc_right_cell,
            atol=bvd_atol,
            rtol=bvd_rtol,
        )
        waves = classify_contact_shock(V, **(sensor_options or {}))
        use_thinc_cell = bvd.use_thinc_cell & waves.allow_thinc_cell
        V_L_riemann, V_R_riemann, thinc_face_mask = assemble_density_bvd_face_states(
            candidates, use_thinc_cell
        )

    mask = mask_rs | thinc_face_mask

    gammaS_left = gammaS_frozen
    e0S_left = e0S_frozen
    gammaS_right = np.roll(gammaS_frozen, -1)
    e0S_right = np.roll(e0S_frozen, -1)

    idx_ho = ~mask
    idx_eno = mask

    mass = np.empty(Nx)
    momentum = np.empty(Nx)
    energy_left = np.empty(Nx)
    energy_right = np.empty(Nx)

    if idx_ho.any():
        ff_ho = flux_ho(V_face_ho[:, idx_ho], gammaS_left[idx_ho], e0S_left[idx_ho],
                         gammaS_right[idx_ho], e0S_right[idx_ho])
        mass[idx_ho] = ff_ho.mass
        momentum[idx_ho] = ff_ho.momentum
        energy_left[idx_ho] = ff_ho.energy_left_owner
        energy_right[idx_ho] = ff_ho.energy_right_owner

    if idx_eno.any():
        ff_eno = flux_hllc_df(V_L_riemann[:, idx_eno], V_R_riemann[:, idx_eno],
                                gammaS_left[idx_eno], e0S_left[idx_eno],
                                gammaS_right[idx_eno], e0S_right[idx_eno])
        mass[idx_eno] = ff_eno.mass
        momentum[idx_eno] = ff_eno.momentum
        energy_left[idx_eno] = ff_eno.energy_left_owner
        energy_right[idx_eno] = ff_eno.energy_right_owner

    return (
        mass,
        momentum,
        energy_left,
        energy_right,
        mask,
        V_L_riemann,
        V_R_riemann,
        gammaS_left,
        e0S_left,
        gammaS_right,
        e0S_right,
        limited_mask,
        thinc_face_mask,
        use_thinc_cell,
    )


def get_rhs_hybrid_fc(V, dx, zeta=0.2, mask_variant='direct', limit_ho=False):
    """Hybrid residual, fully conservative formulation."""
    mass, momentum, energy, mask, _, _, _ = get_hybrid_face_fluxes_fc(
        V, dx, zeta, mask_variant, limit_ho=limit_ho)
    RHS = _assemble_rhs_fc(mass, momentum, energy, dx)
    return RHS, mask


def get_rhs_hybrid_df(
    V,
    gammaS_frozen,
    e0S_frozen,
    dx,
    zeta=0.2,
    mask_variant='direct',
    limit_ho=False,
    use_thinc_bvd=False,
    **thinc_bvd_options,
):
    """Hybrid residual, double-flux formulation."""
    res = get_hybrid_face_fluxes_df(
        V,
        gammaS_frozen,
        e0S_frozen,
        dx,
        zeta,
        mask_variant,
        limit_ho=limit_ho,
        use_thinc_bvd=use_thinc_bvd,
        **thinc_bvd_options,
    )
    mass, momentum, energy_left, energy_right, mask = res[:5]
    RHS = _assemble_rhs_df(mass, momentum, energy_left, energy_right, dx)
    return RHS, mask
