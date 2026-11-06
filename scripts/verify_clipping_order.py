"""Order of the centred four-point reconstruction with and without clipping (linear advection of a sine)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np

from reconstruction import reconstruct_high_order, reconstruct_high_order_limited


def run(n, clip, cfl=0.4, speed=1.0):
    """Errors after one period."""
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) * dx
    q0 = np.sin(2.0 * np.pi * x)
    q = q0.copy()

    def rhs(q):
        face = reconstruct_high_order_limited(q) if clip else reconstruct_high_order(q)
        f = speed * face
        return -(f - np.roll(f, 1)) / dx

    steps = int(np.ceil(1.0 / (cfl * dx / speed)))
    dt = 1.0 / steps
    for _ in range(steps):
        q1 = q + dt * rhs(q)
        q2 = 0.75 * q + 0.25 * (q1 + dt * rhs(q1))
        q = q / 3.0 + 2.0 / 3.0 * (q2 + dt * rhs(q2))
    e = q - q0
    return np.mean(np.abs(e)), np.sqrt(np.mean(e**2)), np.max(np.abs(e))


if __name__ == "__main__":
    for clip in (True, False):
        print("clipped" if clip else "unclipped")
        prev = None
        for n in (50, 100, 200, 400, 800):
            err = run(n, clip)
            rates = "" if prev is None else "  rates " + " ".join(f"{np.log2(a / b):.2f}" for a, b in zip(prev, err))
            print(f"  N={n:4d}  L1={err[0]:.3e}  L2={err[1]:.3e}  Linf={err[2]:.3e}{rates}")
            prev = err
