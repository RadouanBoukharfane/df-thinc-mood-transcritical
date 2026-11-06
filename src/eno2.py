"""Second-order ENO reconstruction."""

import numpy as np


def reconstruct_eno2_left(V):
    """Left state at face i+1/2 (from cell i)."""
    V = np.asarray(V, dtype=float)
    axis = -1

    V_im1 = np.roll(V, shift=1, axis=axis)
    V_ip1 = np.roll(V, shift=-1, axis=axis)

    diff_left = np.abs(V - V_im1)
    diff_right = np.abs(V_ip1 - V)

    cand_A = 1.5 * V - 0.5 * V_im1
    cand_B = 0.5 * V + 0.5 * V_ip1

    use_A = diff_left <= diff_right
    V_L = np.where(use_A, cand_A, cand_B)
    return V_L


def reconstruct_eno2_right(V):
    """Right state at face i+1/2 (from cell i+1)."""
    V = np.asarray(V, dtype=float)
    axis = -1

    V_ip1 = np.roll(V, shift=-1, axis=axis)
    V_ip2 = np.roll(V, shift=-2, axis=axis)

    diff_left = np.abs(V_ip1 - V)
    diff_right = np.abs(V_ip2 - V_ip1)

    cand_A = 1.5 * V_ip1 - 0.5 * V_ip2
    cand_B = 0.5 * V + 0.5 * V_ip1

    use_A = diff_right <= diff_left
    V_R = np.where(use_A, cand_A, cand_B)
    return V_R
