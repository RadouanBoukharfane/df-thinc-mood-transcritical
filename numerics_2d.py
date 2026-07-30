"""Conversions primitives/conservatives pour l'Euler 2D double-flux.

Convention
----------
Primitives V = [rho, u, v, p], forme (4, ...)
Conservatives Q = [rho, rho*u, rho*v, rho*E], forme (4, ...)

L'energie totale contient les DEUX composantes cinetiques::

    rho E = rho e + 1/2 rho (u**2 + v**2)

Dans une etape double-flux, gammaS et e0S sont geles par cellule et la
pression est reconstruite par::

    p = (gammaS - 1) [rhoE - rho e0S
                       - 1/2 rho (u**2 + v**2)].
"""

from __future__ import annotations

import numpy as np

from thermo import get_T_from_p_rho, get_energy_from_T_rho, get_gammaS_e0S


def _validate_state(A: np.ndarray, name: str) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim < 2 or A.shape[0] != 4:
        raise ValueError(f"{name} doit avoir la forme (4, ...)")
    if not np.all(np.isfinite(A)):
        raise ValueError(f"{name} contient des valeurs non finies")
    return A


def primitives_to_conservatives_2d(V: np.ndarray) -> np.ndarray:
    """Convertit ``[rho,u,v,p]`` vers ``[rho,rho*u,rho*v,rho*E]``.

    L'energie interne reelle est obtenue avec l'EOS de Peng--Robinson.
    """
    V = _validate_state(V, "V")
    rho, u, v, p = V
    if np.any(rho <= 0.0) or np.any(p <= 0.0):
        raise ValueError("rho et p doivent etre strictement positifs")

    T = get_T_from_p_rho(p.reshape(-1), rho.reshape(-1)).reshape(rho.shape)
    e = get_energy_from_T_rho(T.reshape(-1), rho.reshape(-1)).reshape(rho.shape)

    Q = np.empty_like(V)
    Q[0] = rho
    Q[1] = rho * u
    Q[2] = rho * v
    Q[3] = rho * e + 0.5 * rho * (u * u + v * v)
    return Q


def update_primitives_df_2d(Q: np.ndarray, gammaS: np.ndarray, e0S: np.ndarray) -> np.ndarray:
    """Inversion double-flux 2D avec parametres thermodynamiques geles."""
    Q = _validate_state(Q, "Q")
    rho = Q[0]
    if np.any(rho <= 0.0):
        raise ValueError("rho doit etre strictement positif")

    gammaS = np.asarray(gammaS, dtype=float)
    e0S = np.asarray(e0S, dtype=float)
    try:
        gammaS = np.broadcast_to(gammaS, rho.shape)
        e0S = np.broadcast_to(e0S, rho.shape)
    except ValueError as exc:
        raise ValueError("gammaS/e0S non diffusables vers la forme spatiale") from exc

    if np.any(gammaS <= 1.0) or not np.all(np.isfinite(gammaS)) or not np.all(np.isfinite(e0S)):
        raise ValueError("parametres double-flux invalides")

    u = Q[1] / rho
    v = Q[2] / rho
    kinetic = 0.5 * rho * (u * u + v * v)
    p = (gammaS - 1.0) * (Q[3] - rho * e0S - kinetic)

    V = np.empty_like(Q)
    V[0] = rho
    V[1] = u
    V[2] = v
    V[3] = p
    return V


def get_gammaS_e0S_2d(V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcule les parametres DF depuis ``rho,p`` d'un etat 2D.

    La thermodynamique est independante des vitesses. On reutilise donc
    l'interface 1D avec une vitesse factice nulle.
    """
    V = _validate_state(V, "V")
    rho, _, _, p = V
    flat = np.vstack([rho.reshape(-1), np.zeros(rho.size), p.reshape(-1)])
    gammaS, e0S = get_gammaS_e0S(flat)
    return gammaS.reshape(rho.shape), e0S.reshape(rho.shape)


def modified_total_energy_2d(
    V: np.ndarray,
    gammaS_owner: np.ndarray,
    e0S_owner: np.ndarray,
) -> np.ndarray:
    """Energie volumique modifiee d'un proprietaire DF.

    ``gammaS_owner`` et ``e0S_owner`` sont appliques uniformement a l'etat
    de face, conformement au double-flux.
    """
    V = _validate_state(V, "V")
    rho, u, v, p = V
    gammaS_owner = np.broadcast_to(np.asarray(gammaS_owner, float), rho.shape)
    e0S_owner = np.broadcast_to(np.asarray(e0S_owner, float), rho.shape)
    return (
        p / (gammaS_owner - 1.0)
        + rho * e0S_owner
        + 0.5 * rho * (u * u + v * v)
    )
