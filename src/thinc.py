"""THINC reconstruction (hyperbolic-tangent subcell profile)."""

import numpy as np


def _validate_beta(beta):
    beta = float(beta)
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"beta must be finite and positive, got {beta!r}")
    return beta


def reconstruct_thinc(V, beta=1.6, eps=1.0e-14):
    """THINC traces at the left and right faces of each cell."""
    q = np.asarray(V, dtype=float)
    if q.ndim == 0:
        raise ValueError("V must have a spatial axis")
    if q.shape[-1] < 3:
        raise ValueError("THINC requires at least 3 periodic cells")

    beta = _validate_beta(beta)
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"eps must be finite and positive, got {eps!r}")

    axis = -1
    q_im1 = np.roll(q, shift=1, axis=axis)
    q_ip1 = np.roll(q, shift=-1, axis=axis)

    q_min = np.minimum(q_im1, q_ip1)
    q_max = np.maximum(q_im1, q_ip1)
    dq = q_max - q_min

    monotone = (q - q_im1) * (q_ip1 - q) > 0.0

    local_scale = np.maximum.reduce(
        [np.ones_like(q), np.abs(q_im1), np.abs(q), np.abs(q_ip1)]
    )
    nondegenerate = dq > eps * local_scale
    active = monotone & nondegenerate

    q_left = q.copy()
    q_right = q.copy()

    if not np.any(active):
        return q_left, q_right

    theta = np.sign(q_ip1 - q_im1)

    C = np.zeros_like(q)
    np.divide(q - q_min, dq, out=C, where=nondegenerate)
    tiny = np.finfo(float).eps
    C = np.clip(C, tiny, 1.0 - tiny)

    h_bar = theta * (2.0 * C - 1.0)
    h_bar = np.clip(h_bar, -1.0 + tiny, 1.0 - tiny)

    exp_beta = np.exp(beta)
    exp_minus_beta = np.exp(-beta)
    r = np.exp(beta * h_bar)

    numerator = exp_beta - r
    denominator = r - exp_minus_beta

    positive_floor = np.finfo(float).tiny
    y = np.maximum(numerator, positive_floor) / np.maximum(denominator, positive_floor)
    xi0 = 0.5 * np.log(y) / beta

    h_left = np.tanh(-beta * xi0)
    h_right = np.tanh(beta * (1.0 - xi0))

    q_left_thinc = q_min + 0.5 * dq * (1.0 + theta * h_left)
    q_right_thinc = q_min + 0.5 * dq * (1.0 + theta * h_right)

    q_left = np.where(active, q_left_thinc, q_left)
    q_right = np.where(active, q_right_thinc, q_right)

    q_left = np.where(active, np.clip(q_left, q_min, q_max), q_left)
    q_right = np.where(active, np.clip(q_right, q_min, q_max), q_right)

    return q_left, q_right


def reconstruct_thinc_left(V, beta=1.6, eps=1.0e-14):
    """THINC trace at the left face."""
    q_left, _ = reconstruct_thinc(V, beta=beta, eps=eps)
    return q_left


def reconstruct_thinc_right(V, beta=1.6, eps=1.0e-14):
    """THINC trace at the right face."""
    _, q_right = reconstruct_thinc(V, beta=beta, eps=eps)
    return q_right
