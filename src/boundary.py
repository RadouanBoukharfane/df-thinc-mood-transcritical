"""Boundary conditions and ghost cells for the non-periodic 1-D solver."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BoundaryCondition:
    """Boundary condition: 'transmissive', 'dirichlet', or 'wall'."""
    kind: str
    primitive: np.ndarray | None = None

    def __post_init__(self):
        kind = self.kind.lower()
        if kind not in {"transmissive", "dirichlet", "wall"}:
            raise ValueError("kind must be 'transmissive', 'dirichlet' or 'wall'")
        object.__setattr__(self, "kind", kind)
        if kind == "dirichlet":
            if self.primitive is None:
                raise ValueError("a Dirichlet condition requires primitive=(rho, u, p)")
            primitive = np.asarray(self.primitive, dtype=float)
            if primitive.shape != (3,) or not np.all(np.isfinite(primitive)):
                raise ValueError("primitive must be a finite array of shape (3,)")
            if primitive[0] <= 0.0 or primitive[2] <= 0.0:
                raise ValueError("boundary rho and p must be positive")
            object.__setattr__(self, "primitive", primitive)


def transmissive():
    """Zero-gradient boundary."""
    return BoundaryCondition("transmissive")


def dirichlet(primitive):
    """Fixed primitive state (rho, u, p)."""
    return BoundaryCondition("dirichlet", primitive=np.asarray(primitive, dtype=float))


def adiabatic_wall():
    """Inviscid adiabatic wall (mirror state, reversed normal velocity)."""
    return BoundaryCondition("wall")


def fill_primitive_ghosts(V, left, right, nghost=2):
    """Return the primitive state padded with ghost cells."""
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] != 3:
        raise ValueError("V must have shape (3, Nx)")
    if V.shape[1] < 4:
        raise ValueError("Nx must be at least 4")
    if int(nghost) != nghost or nghost < 2:
        raise ValueError("nghost must be an integer >= 2")
    nghost = int(nghost)

    Vg = np.empty((3, V.shape[1] + 2 * nghost), dtype=float)
    Vg[:, nghost:-nghost] = V

    if left.kind == "transmissive":
        Vg[:, :nghost] = V[:, [0]]
    elif left.kind == "dirichlet":
        Vg[:, :nghost] = left.primitive[:, None]
    else:
        Vg[:, :nghost] = V[:, :nghost][:, ::-1]
        Vg[1, :nghost] *= -1.0

    if right.kind == "transmissive":
        Vg[:, -nghost:] = V[:, [-1]]
    elif right.kind == "dirichlet":
        Vg[:, -nghost:] = right.primitive[:, None]
    else:
        Vg[:, -nghost:] = V[:, -nghost:][:, ::-1]
        Vg[1, -nghost:] *= -1.0
    return Vg


def fill_scalar_ghosts(q, left_value=None, right_value=None, nghost=2):
    """Pad a cell field with constant ghost values."""
    q = np.asarray(q, dtype=float)
    nghost = int(nghost)
    out = np.empty(q.shape[:-1] + (q.shape[-1] + 2 * nghost,), dtype=float)
    out[..., nghost:-nghost] = q
    lv = q[..., 0] if left_value is None else np.asarray(left_value, dtype=float)
    rv = q[..., -1] if right_value is None else np.asarray(right_value, dtype=float)
    out[..., :nghost] = np.expand_dims(lv, axis=-1)
    out[..., -nghost:] = np.expand_dims(rv, axis=-1)
    return out
