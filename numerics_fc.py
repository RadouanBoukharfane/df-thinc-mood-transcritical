"""
numerics_fc.py
Schema "fully conservative" (FC) de comparaison, Section 4.1 du papier.

Contrairement au modele double-flux (numerics.py / HLLC_DF), il n'y a ICI
aucun gel de gammaS/e0S par cellule : le gamma effectif et l'energie de
reference sont recalcules independamment pour l'etat gauche et l'etat
droit a CHAQUE evaluation de flux, a partir de leur propre thermodynamique
reelle (Peng-Robinson). Ces deux quantites ne sont pas des parametres de
fermeture au sens strict : elles servent uniquement a reecrire exactement
l'energie interne locale sous la forme rho*e = p/(gammaS-1) + rho*e0S
(c'est une identite algebrique par construction, cf. Eq. 16 du papier).

Le flux est evalue avec l'energie thermodynamique reelle propre a chaque
etat. Un flux energetique UNIQUE est utilise a chaque face (contrairement
au double-flux qui evalue deux flux differents a la meme face, cf. Fig. 3),
ce qui garantit la conservation totale de l'energie -- mais ne garantit
PAS l'equilibre discret de pression lors de l'advection d'une interface
presentant de fortes variations thermodynamiques. C'est precisement cette
absence de garantie (formalisee par l'Eq. 19 du papier) qui cause les
oscillations de pression du schema fully-conservative, pas un desaccord
entre gammaS_L et gammaS_R en tant que tel.

Le flux est calcule UNE seule fois par face (contrairement au double-flux
qui evalue deux flux differents a la meme face, cf. Fig. 3) : le schema
est donc bien "fully conservative" au sens strict (masse, quantite de
mouvement ET energie conserves exactement par construction du schema en
volumes finis), mais souffre de l'erreur thermodynamique de l'Eq. (19).
"""

import numpy as np

from thermo import get_gammaS_e0S, get_T_from_e_rho, get_p_from_T_rho


def primitives_to_conservatives_fc(primitives):
    """Identique a primitives_to_conservatives (reutilise directement la
    meme fonction que dans numerics.py -- pas de redefinition necessaire,
    la conversion primitive->conservative ne depend pas du schema)."""
    from numerics import primitives_to_conservatives
    return primitives_to_conservatives(primitives)


def update_primitives_fc(W):
    """Mise a jour des primitives depuis les conservatives POUR LE SCHEMA FC.

    Contrairement a updatePrimitives_DF.m (qui utilise Eq. 22, directe,
    sans iteration, grace au gel de gammaS/e0S), le schema fully-
    conservative doit repasser par l'inversion iterative complete
    (rho, e) -> T -> p, exactement le chemin "iteratively solve for p or
    T, given rho, e" du flowchart Fig. 4(a) du papier.
    """
    rho = W[0, :]
    u = W[1, :] / W[0, :]
    E = W[2, :]
    e = E / rho - 0.5 * u**2

    T = get_T_from_e_rho(e, rho)     # <-- l'iteration couteuse evitee par le DF
    p = get_p_from_T_rho(T, rho)

    V = np.zeros_like(W)
    V[0, :] = rho
    V[1, :] = u
    V[2, :] = p
    return V


def _sign(s):
    return (s >= 0).astype(float) - (s <= 0).astype(float)


def hllc_fc(V_L, V_R):
    """Flux HLLC "fully conservative" : gammaS/e0S recalcules independamment
    pour l'etat gauche et l'etat droit a partir de LEUR PROPRE etat
    thermodynamique reel (pas de gel, pas de partage de valeur entre L et R).

    Un seul flux est retourne pour la face (partage entre les deux cellules
    voisines -> conservation exacte), contrairement a HLLC_DF qui produit
    deux flux differents pour la meme face (cf. numerics.py).
    """
    rho_L, rho_R = V_L[0, :], V_R[0, :]
    u_L, u_R = V_L[1, :], V_R[1, :]
    p_L, p_R = V_L[2, :], V_R[2, :]

    # gammaS, e0S propres a chaque etat -- NON partages entre L et R
    gammaS_L, e0S_L = get_gammaS_e0S(V_L)
    gammaS_R, e0S_R = get_gammaS_e0S(V_R)

    a_L = np.sqrt(gammaS_L * p_L / rho_L)
    a_R = np.sqrt(gammaS_R * p_R / rho_R)

    E_L = p_L / (gammaS_L - 1) + rho_L * e0S_L + 0.5 * rho_L * u_L**2
    E_R = p_R / (gammaS_R - 1) + rho_R * e0S_R + 0.5 * rho_R * u_R**2

    f_L = np.vstack([rho_L * u_L, rho_L * u_L**2 + p_L, u_L * (E_L + p_L)])
    f_R = np.vstack([rho_R * u_R, rho_R * u_R**2 + p_R, u_R * (E_R + p_R)])

    W_L = np.vstack([rho_L, rho_L * u_L, E_L])
    W_R = np.vstack([rho_R, rho_R * u_R, E_R])

    s_L = u_L - a_L
    s_R = u_R + a_R

    sM = (p_L - p_R - rho_L * u_L * (s_L - u_L) + rho_R * u_R * (s_R - u_R)) / (
        rho_R * (s_R - u_R) - rho_L * (s_L - u_L)
    )
    pM = rho_R * (u_R - s_R) * (u_R - sM) + p_R

    WML = np.vstack([
        (s_L - u_L) / (s_L - sM) * rho_L,
        ((s_L - u_L) * rho_L * u_L + (pM - p_L)) / (s_L - sM),
        ((s_L - u_L) * E_L - p_L * u_L + pM * sM) / (s_L - sM),
    ])
    WMR = np.vstack([
        (s_R - u_R) / (s_R - sM) * rho_R,
        ((s_R - u_R) * rho_R * u_R + (pM - p_R)) / (s_R - sM),
        ((s_R - u_R) * E_R - p_R * u_R + pM * sM) / (s_R - sM),
    ])

    s_minus = np.minimum(0, s_L)
    s_plus = np.maximum(0, s_R)
    sgn = _sign(sM)

    f_HLLC = (1 + sgn) / 2 * (f_L + s_minus * (WML - W_L)) + \
             (1 - sgn) / 2 * (f_R + s_plus * (WMR - W_R))

    return f_HLLC


def get_rhs_fc(V, dx):
    """RHS fully-conservative, reconstruction ordre 1, CL periodiques.

    Un seul flux HLLC est calcule par face (partage entre les deux
    cellules), d'ou l'appel unique a hllc_fc par face au lieu des deux
    appels flux_L/flux_R de get_rhs_df (qui, eux, utilisent chacun le
    meme gammaS/e0S du domaine entier -- cf. numerics.py).
    """
    Nx = V.shape[1]

    V_m = np.roll(V, shift=1, axis=1)   # V(i-1), periodique
    V_p = np.roll(V, shift=-1, axis=1)  # V(i+1), periodique

    flux_left_face = hllc_fc(V_m, V)    # flux a la face i-1/2
    flux_right_face = hllc_fc(V, V_p)   # flux a la face i+1/2

    RHS = -(flux_right_face - flux_left_face) / dx
    return RHS
