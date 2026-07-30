"""
cfl2d.py
Etape 2D-2 : pas de temps CFL pour le schema 2D splitte.

Formule (coherente avec le splitting directionnel) :

    dt = CFL / ( max(|u|+c)/dx + max(|v|+c)/dy )

C'est la generalisation naturelle de la formule 1D corrigee
(main_hybrid._cfl_dt, dt = CFL*dx/max(|u|+c)) au cas 2D avec deux
directions de sweep independantes -- chaque direction contribue sa propre
contrainte de stabilite, sommees (pas un simple minimum des deux dt 1D
independants, qui serait moins restrictif et donc potentiellement
instable).
"""

import numpy as np

from thermo import get_sos


def cfl_dt_2d(V2D, dx, dy, CFL):
    """V2D : (4, Ny, Nx) = [rho, u, v, p]. Retourne un scalaire dt."""
    V2D = np.asarray(V2D, dtype=float)
    if V2D.ndim != 3 or V2D.shape[0] != 4:
        raise ValueError("V2D doit etre de forme (4, Ny, Nx)")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError("dx et dy doivent etre strictement positifs")
    if CFL <= 0.0:
        raise ValueError("CFL doit etre strictement positif")

    rho = V2D[0].ravel()
    u = V2D[1].ravel()
    v = V2D[2].ravel()
    p = V2D[3].ravel()

    if np.any(~np.isfinite(rho)) or np.any(rho <= 0.0):
        raise ValueError("densite non finie ou non positive dans cfl_dt_2d")
    if np.any(~np.isfinite(p)) or np.any(p <= 0.0):
        raise ValueError("pression non finie ou non positive dans cfl_dt_2d")

    # get_sos n'utilise que V[0]=rho et V[2]=p ; la ligne du milieu est
    # ignoree, on peut y mettre n'importe quoi de fini (ici u par
    # commodite, sans consequence sur le resultat).
    sos = get_sos(np.vstack([rho, u, p]))

    speed_x = np.max(np.abs(u) + sos) / dx
    speed_y = np.max(np.abs(v) + sos) / dy

    return CFL / (speed_x + speed_y)
