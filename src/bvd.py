"""Boundary-variation-diminishing (BVD) comparison of two reconstruction candidates."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BVDDiagnostics:
    """Face jumps, total boundary variations, and THINC selection."""

    jump_polynomial_face: np.ndarray
    jump_thinc_face: np.ndarray
    tbv_polynomial_cell: np.ndarray
    tbv_thinc_cell: np.ndarray
    use_thinc_cell: np.ndarray

    @property
    def thinc_fraction(self):
        return float(np.mean(self.use_thinc_cell))

    @property
    def tbv_gain(self):
        return self.tbv_polynomial_cell - self.tbv_thinc_cell


def _as_1d_finite(name, values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {array.shape}")
    if array.size < 3:
        raise ValueError(f"{name} must contain at least 3 periodic cells")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


def face_boundary_jump(q_left_cell, q_right_cell):
    """Absolute jump of the cell traces at each face."""
    q_left_cell = _as_1d_finite("q_left_cell", q_left_cell)
    q_right_cell = _as_1d_finite("q_right_cell", q_right_cell)
    if q_left_cell.shape != q_right_cell.shape:
        raise ValueError("q_left_cell and q_right_cell must have the same shape")

    q_right_state_at_face = np.roll(q_left_cell, -1)
    q_left_state_at_face = q_right_cell
    return np.abs(q_right_state_at_face - q_left_state_at_face)


def total_boundary_variation(q_left_cell, q_right_cell):
    """Total boundary variation of each cell and face jumps."""
    jump_face = face_boundary_jump(q_left_cell, q_right_cell)
    tbv_cell = np.roll(jump_face, 1) + jump_face
    return tbv_cell, jump_face


def eno_face_states_to_cell_boundaries(q_left_face, q_right_face):
    """Convert ENO face states to per-cell traces."""
    q_left_face = _as_1d_finite("q_left_face", q_left_face)
    q_right_face = _as_1d_finite("q_right_face", q_right_face)
    if q_left_face.shape != q_right_face.shape:
        raise ValueError("q_left_face and q_right_face must have the same shape")

    q_left_cell = np.roll(q_right_face, 1)
    q_right_cell = q_left_face.copy()
    return q_left_cell, q_right_cell


def compare_bvd_candidates(
    polynomial_left_cell,
    polynomial_right_cell,
    thinc_left_cell,
    thinc_right_cell,
    *,
    atol=1.0e-14,
    rtol=1.0e-12,
):
    """Select THINC where it strictly reduces the total boundary variation."""
    arrays = [
        _as_1d_finite("polynomial_left_cell", polynomial_left_cell),
        _as_1d_finite("polynomial_right_cell", polynomial_right_cell),
        _as_1d_finite("thinc_left_cell", thinc_left_cell),
        _as_1d_finite("thinc_right_cell", thinc_right_cell),
    ]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays[1:]):
        raise ValueError("all candidates must have the same shape")

    atol = float(atol)
    rtol = float(rtol)
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and non-negative")
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError("rtol must be finite and non-negative")

    tbv_p, jump_p = total_boundary_variation(arrays[0], arrays[1])
    tbv_t, jump_t = total_boundary_variation(arrays[2], arrays[3])

    scale = np.maximum(tbv_p, tbv_t)
    tolerance = atol + rtol * scale
    use_thinc = tbv_t + tolerance < tbv_p

    return BVDDiagnostics(
        jump_polynomial_face=jump_p,
        jump_thinc_face=jump_t,
        tbv_polynomial_cell=tbv_p,
        tbv_thinc_cell=tbv_t,
        use_thinc_cell=use_thinc,
    )
