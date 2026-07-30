"""
eno2.py
Reconstruction ENO (Essentially Non-Oscillatory) d'ordre 2, schema de
secours dissipatif de la Section 4.3 du papier, destine a etre combine au
capteur RS avec la reconstruction centree d'ordre 4 (reconstruction.py).

DIFFERENCE FONDAMENTALE avec reconstruction.py : ENO2 produit DEUX etats
DISTINCTS a chaque face i+1/2, comme l'exige un solveur de Riemann (HLLC) :

    V_{i+1/2}^{L,ENO} -- approche depuis la gauche (biaise sur la cellule i)
    V_{i+1/2}^{R,ENO} -- approche depuis la droite (biaise sur la cellule i+1)

contrairement a la reconstruction centree, qui produit une seule valeur
V_{i+1/2}^{C} (cf. avertissement dans reconstruction.py).

Algorithme standard (cf. Shu, "Essentially Non-Oscillatory and Weighted
Essentially Non-Oscillatory Schemes for Hyperbolic Conservation Laws",
notes de cours ENO/WENO) : reconstruction lineaire (stencil a 2 points,
ordre 2) avec selection "essentiellement non-oscillatoire" du stencil --
a chaque face, on choisit entre deux stencils de 2 cellules celui dont la
difference non-divisee (indicateur de regularite locale) est la plus
petite en valeur absolue. Cela evite de construire un polynome a travers
une discontinuite quand un stencil plus lisse est disponible de l'autre
cote -- propriete cle : PAS DE NOUVEL EXTREMUM introduit par rapport aux
donnees du stencil (verifie par test_5 dans test_eno2_unit.py).

CL periodiques (via np.roll), coherent avec reconstruction.py.
"""

import numpy as np


def reconstruct_eno2_left(V):
    """Etat gauche V_{i+1/2}^{L,ENO}, biaise sur la cellule i.

    Deux stencils candidats :
        A = {i-1, i}  -> extrapolation vers la droite : 1.5*V[i] - 0.5*V[i-1]
        B = {i, i+1}  -> interpolation : 0.5*V[i] + 0.5*V[i+1]

    Selection : on prend le stencil dont la difference non-divisee locale
    est la plus petite en valeur absolue (le plus "lisse") :
        |V[i]-V[i-1]| <= |V[i+1]-V[i]|  =>  stencil A
        sinon                            =>  stencil B
    """
    V = np.asarray(V, dtype=float)
    axis = -1

    V_im1 = np.roll(V, shift=1, axis=axis)    # V[i-1]
    V_ip1 = np.roll(V, shift=-1, axis=axis)   # V[i+1]

    diff_left = np.abs(V - V_im1)      # |V[i]-V[i-1]|
    diff_right = np.abs(V_ip1 - V)     # |V[i+1]-V[i]|

    cand_A = 1.5 * V - 0.5 * V_im1     # stencil {i-1, i}
    cand_B = 0.5 * V + 0.5 * V_ip1     # stencil {i, i+1}

    use_A = diff_left <= diff_right
    V_L = np.where(use_A, cand_A, cand_B)
    return V_L


def reconstruct_eno2_right(V):
    """Etat droit V_{i+1/2}^{R,ENO}, biaise sur la cellule i+1 -- miroir
    exact de reconstruct_eno2_left, centre une cellule plus loin.

    Deux stencils candidats :
        A = {i+1, i+2} -> extrapolation vers la gauche : 1.5*V[i+1] - 0.5*V[i+2]
        B = {i, i+1}   -> interpolation : 0.5*V[i] + 0.5*V[i+1]

    Selection (miroir de reconstruct_eno2_left) :
        |V[i+2]-V[i+1]| <= |V[i+1]-V[i]|  =>  stencil A
        sinon                              =>  stencil B
    """
    V = np.asarray(V, dtype=float)
    axis = -1

    V_ip1 = np.roll(V, shift=-1, axis=axis)   # V[i+1]
    V_ip2 = np.roll(V, shift=-2, axis=axis)   # V[i+2]

    diff_left = np.abs(V_ip1 - V)       # |V[i+1]-V[i]|
    diff_right = np.abs(V_ip2 - V_ip1)  # |V[i+2]-V[i+1]|

    cand_A = 1.5 * V_ip1 - 0.5 * V_ip2  # stencil {i+1, i+2}
    cand_B = 0.5 * V + 0.5 * V_ip1      # stencil {i, i+1}

    use_A = diff_right <= diff_left
    V_R = np.where(use_A, cand_A, cand_B)
    return V_R
