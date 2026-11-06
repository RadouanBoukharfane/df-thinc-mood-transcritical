"""Oscillation sensor of Ma et al. and contact/shock classification."""

import numpy as np

from reconstruction import reconstruct_high_order


def rs_raw_indicator(rho, zeta=None):
    """Relative departure of the central density face value from both cells."""
    rho = np.asarray(rho, dtype=float)
    rho_face = reconstruct_high_order(rho)

    rho_ip1 = np.roll(rho, -1)

    indicator_left = np.abs(rho_face - rho) / rho
    indicator_right = np.abs(rho_face - rho_ip1) / rho_ip1
    return indicator_left, indicator_right


def rs_face_mask_direct(rho, zeta=0.2):
    """Faces flagged by the oscillation sensor."""
    indicator_left, indicator_right = rs_raw_indicator(rho)
    indicator_face = np.maximum(indicator_left, indicator_right)
    mask = indicator_face > zeta
    return mask, indicator_face


def rs_cell_flag(rho, zeta=0.2):
    """Cells with a flagged face."""
    indicator_left, indicator_right = rs_raw_indicator(rho)
    face_right_of_i = indicator_left
    face_left_of_i = np.roll(indicator_right, 1)

    M_i = np.maximum(face_right_of_i, face_left_of_i) > zeta
    return M_i


def rs_face_mask_from_cells(rho, zeta=0.2):
    """Faces adjacent to a flagged cell."""
    M = rs_cell_flag(rho, zeta)
    M_ip1 = np.roll(M, -1)
    mask = np.logical_or(M, M_ip1)
    return mask


def eno_fraction(mask):
    """Fraction of flagged faces."""
    return np.mean(mask)


from dataclasses import dataclass


@dataclass(frozen=True)
class ContactShockDiagnostics:
    """Face indicators and contact/shock/smooth masks."""

    density_jump_face: np.ndarray
    pressure_jump_face: np.ndarray
    velocity_jump_face: np.ndarray
    compression_face: np.ndarray
    contact_score_face: np.ndarray
    shock_score_face: np.ndarray
    contact_face: np.ndarray
    shock_face: np.ndarray
    smooth_face: np.ndarray
    contact_cell: np.ndarray
    shock_cell: np.ndarray
    smooth_cell: np.ndarray

    @property
    def contact_fraction(self):
        return float(np.mean(self.contact_cell))

    @property
    def shock_fraction(self):
        return float(np.mean(self.shock_cell))

    @property
    def smooth_fraction(self):
        return float(np.mean(self.smooth_cell))

    @property
    def allow_thinc_cell(self):
        return self.contact_cell.copy()


def _validate_primitive_state(V):
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] < 3:
        raise ValueError(
            "V must have shape (Nvars, Nx) with at least (rho, u, p)"
        )
    if V.shape[1] < 3:
        raise ValueError("at least three periodic cells are required")
    if not np.all(np.isfinite(V[:3])):
        raise ValueError("V contains a non-finite value in (rho, u, p)")
    if np.any(V[0] <= 0.0):
        raise ValueError("density must be positive")
    if np.any(V[2] <= 0.0):
        raise ValueError("pressure must be positive")
    return V


def contact_shock_face_indicators(
    V,
    *,
    eps=1.0e-14,
    velocity_floor=1.0,
    contact_pressure_weight=20.0,
    contact_velocity_weight=20.0,
    shock_compression_weight=1.0,
):
    """Density, pressure, velocity, and compression indicators."""
    V = _validate_primitive_state(V)

    eps = float(eps)
    velocity_floor = float(velocity_floor)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    if not np.isfinite(velocity_floor) or velocity_floor <= 0.0:
        raise ValueError("velocity_floor must be finite and positive")

    rho, u, p = V[0], V[1], V[2]
    rho_r = np.roll(rho, -1)
    u_r = np.roll(u, -1)
    p_r = np.roll(p, -1)

    density_jump = np.abs(rho_r - rho) / (np.abs(rho_r) + np.abs(rho) + eps)
    pressure_jump = np.abs(p_r - p) / (np.abs(p_r) + np.abs(p) + eps)

    velocity_scale = np.maximum(
        np.abs(u_r) + np.abs(u),
        velocity_floor,
    )
    velocity_jump = np.abs(u_r - u) / (velocity_scale + eps)
    compression = np.maximum(0.0, -(u_r - u)) / (velocity_scale + eps)

    contact_score = density_jump * np.exp(
        -float(contact_pressure_weight) * pressure_jump
        -float(contact_velocity_weight) * velocity_jump
    )
    shock_score = pressure_jump * (
        1.0 + float(shock_compression_weight) * compression
    )

    return {
        "density_jump_face": density_jump,
        "pressure_jump_face": pressure_jump,
        "velocity_jump_face": velocity_jump,
        "compression_face": compression,
        "contact_score_face": contact_score,
        "shock_score_face": shock_score,
    }


def classify_contact_shock(
    V,
    *,
    contact_threshold=5.0e-2,
    shock_pressure_threshold=5.0e-2,
    shock_compression_threshold=1.0e-3,
    shock_velocity_threshold=5.0e-2,
    eps=1.0e-14,
    velocity_floor=1.0,
    contact_pressure_weight=20.0,
    contact_velocity_weight=20.0,
    shock_compression_weight=1.0,
):
    """Classify faces and cells as contact, shock, or smooth."""
    thresholds = {
        "contact_threshold": contact_threshold,
        "shock_pressure_threshold": shock_pressure_threshold,
        "shock_compression_threshold": shock_compression_threshold,
        "shock_velocity_threshold": shock_velocity_threshold,
    }
    for name, value in thresholds.items():
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")

    ind = contact_shock_face_indicators(
        V,
        eps=eps,
        velocity_floor=velocity_floor,
        contact_pressure_weight=contact_pressure_weight,
        contact_velocity_weight=contact_velocity_weight,
        shock_compression_weight=shock_compression_weight,
    )

    pressure_jump = ind["pressure_jump_face"]
    velocity_jump = ind["velocity_jump_face"]
    compression = ind["compression_face"]
    contact_score = ind["contact_score_face"]

    shock_face = (
        pressure_jump > shock_pressure_threshold
    ) & (
        (compression > shock_compression_threshold)
        | (velocity_jump > shock_velocity_threshold)
    )
    contact_face = (~shock_face) & (contact_score > contact_threshold)
    smooth_face = ~(shock_face | contact_face)

    shock_cell = shock_face | np.roll(shock_face, 1)
    contact_cell = (~shock_cell) & (contact_face | np.roll(contact_face, 1))
    smooth_cell = ~(shock_cell | contact_cell)

    return ContactShockDiagnostics(
        density_jump_face=ind["density_jump_face"],
        pressure_jump_face=pressure_jump,
        velocity_jump_face=velocity_jump,
        compression_face=compression,
        contact_score_face=contact_score,
        shock_score_face=ind["shock_score_face"],
        contact_face=contact_face,
        shock_face=shock_face,
        smooth_face=smooth_face,
        contact_cell=contact_cell,
        shock_cell=shock_cell,
        smooth_cell=smooth_cell,
    )
