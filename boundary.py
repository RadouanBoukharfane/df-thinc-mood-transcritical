"""Conditions aux limites 1D et gestion explicite des cellules fantomes.

Etape 12 : infrastructure non periodique. Le solveur historique reste
strictement periodique; ce module fournit une branche separee pour les tubes
a choc et les problemes d'advection avec entree/sortie.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BoundaryCondition:
    kind: str
    primitive: np.ndarray | None = None

    def __post_init__(self):
        kind = self.kind.lower()
        if kind not in {"transmissive", "dirichlet", "wall"}:
            raise ValueError("kind doit valoir 'transmissive', 'dirichlet' ou 'wall'")
        object.__setattr__(self, "kind", kind)
        if kind == "dirichlet":
            if self.primitive is None:
                raise ValueError("une CL dirichlet exige primitive=(rho,u,p)")
            primitive = np.asarray(self.primitive, dtype=float)
            if primitive.shape != (3,) or not np.all(np.isfinite(primitive)):
                raise ValueError("primitive doit etre un tableau fini de forme (3,)")
            if primitive[0] <= 0.0 or primitive[2] <= 0.0:
                raise ValueError("rho et p de la CL doivent etre strictement positifs")
            object.__setattr__(self, "primitive", primitive)


def transmissive():
    return BoundaryCondition("transmissive")


def dirichlet(primitive):
    return BoundaryCondition("dirichlet", primitive=np.asarray(primitive, dtype=float))


def adiabatic_wall():
    """Paroi solide adiabatique/inviscide par reflexion des primitives.

    La densite et la pression sont prolongees symetriquement, tandis que la
    vitesse normale change de signe. Cette construction impose un flux
    massique et energetique nul a la paroi dans le probleme de Riemann.
    """
    return BoundaryCondition("wall")


def fill_primitive_ghosts(V, left, right, nghost=2):
    """Retourne ``Vg`` de forme ``(3, Nx+2*nghost)``.

    Les CL transmissives extrapolent la cellule de bord. Les CL Dirichlet
    imposent un etat primitif constant dans toutes les cellules fantomes.
    """
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] != 3:
        raise ValueError("V doit etre de forme (3,Nx)")
    if V.shape[1] < 4:
        raise ValueError("Nx doit etre au moins egal a 4")
    if int(nghost) != nghost or nghost < 2:
        raise ValueError("nghost doit etre un entier >= 2")
    nghost = int(nghost)

    Vg = np.empty((3, V.shape[1] + 2 * nghost), dtype=float)
    Vg[:, nghost:-nghost] = V

    if left.kind == "transmissive":
        Vg[:, :nghost] = V[:, [0]]
    elif left.kind == "dirichlet":
        Vg[:, :nghost] = left.primitive[:, None]
    else:
        # Miroir geometrique : le fantome le plus proche voit la premiere
        # cellule physique, le suivant voit la deuxieme, etc.
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
    """Remplit des fantomes pour un champ scalaire/cellulaire ``(...,Nx)``."""
    q = np.asarray(q, dtype=float)
    nghost = int(nghost)
    out = np.empty(q.shape[:-1] + (q.shape[-1] + 2 * nghost,), dtype=float)
    out[..., nghost:-nghost] = q
    lv = q[..., 0] if left_value is None else np.asarray(left_value, dtype=float)
    rv = q[..., -1] if right_value is None else np.asarray(right_value, dtype=float)
    out[..., :nghost] = np.expand_dims(lv, axis=-1)
    out[..., -nghost:] = np.expand_dims(rv, axis=-1)
    return out
