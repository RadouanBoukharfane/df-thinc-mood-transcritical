"""Centred four-point cell-average-to-face reconstruction."""

import numpy as np


def reconstruct_high_order(V):
    """Face value (-q[i-1] + 7q[i] + 7q[i+1] - q[i+2]) / 12."""
    V = np.asarray(V, dtype=float)
    axis = -1

    V_im1 = np.roll(V, shift=1, axis=axis)
    V_ip1 = np.roll(V, shift=-1, axis=axis)
    V_ip2 = np.roll(V, shift=-2, axis=axis)

    V_face_centered = (-V_im1 + 7.0 * V + 7.0 * V_ip1 - V_ip2) / 12.0
    return V_face_centered


def reconstruct_high_order_limited(V):
    """Same face value clipped to the range of its stencil."""
    V = np.asarray(V, dtype=float)
    axis = -1

    V_face_centered = reconstruct_high_order(V)

    V_im1 = np.roll(V, shift=1, axis=axis)
    V_i = V
    V_ip1 = np.roll(V, shift=-1, axis=axis)
    V_ip2 = np.roll(V, shift=-2, axis=axis)

    lo = np.minimum(np.minimum(V_im1, V_i), np.minimum(V_ip1, V_ip2))
    hi = np.maximum(np.maximum(V_im1, V_i), np.maximum(V_ip1, V_ip2))

    V_face_limited = np.clip(V_face_centered, lo, hi)
    return V_face_limited
