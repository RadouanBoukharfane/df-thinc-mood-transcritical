"""
reconstruction.py
Reconstruction centree d'ordre 4 (flux "high-order non-dissipative",
Section 4.3 du papier). Premiere brique de l'etape 3 -- testee ICI en
isolation (fonctions constante, lineaire, lisse periodique, discontinue)
avant toute connexion au RK3 ou au capteur RS.

Le papier confirme seulement l'usage d'un schema central d'ordre 4 sur
maillage uniforme, combine a ENO2 (Section 4.3) ; il ne donne pas dans cet
article les coefficients/l'operateur exact (il renvoie a Khalighi et al.,
AIAA 2011-2886, sans les detailler ici). On implemente donc la
reconstruction volumes-finis centree uniforme standard "moyenne de
cellule -> valeur de face" :

    V_face[i] = (-V[i-1] + 7*V[i] + 7*V[i+1] - V[i+2]) / 12

associee a la face i+1/2 (interface entre cellule i et cellule i+1). Cette
formule est EXACTE pour tout polynome de degre <= 3, d'ou une precision
O(dx^4) sur donnees lisses (verifie ci-dessous par etude de convergence).

IMPORTANT -- ce que cette fonction produit reellement :
Cette reconstruction est SYMETRIQUE/CENTREE : elle ne produit PAS deux
etats distincts V_{i+1/2}^{L} et V_{i+1/2}^{R} comme le ferait un solveur
de Riemann. Elle produit UNE SEULE valeur centrale a la face,
V_{i+1/2}^{C} (d'ou le nom de la variable de sortie, V_face_centered).
Cette valeur est adaptee a la definition d'un FLUX CENTRAL non-dissipatif
(evalue directement a partir de cet etat unique), PAS a etre dupliquee
naivement comme VL=VR=V_face puis injectee dans HLLC : cela redonnerait
bien le flux physique de cet etat (HLLC se reduit au flux d'Euler exact
quand VL=VR), mais rien ne prouve que ce soit l'operateur exact utilise
par les auteurs, faute de details algorithmiques complets dans le papier.
L'integration au solveur (etape suivante) devra donc definir explicitement
comment ce flux central est combine au flux ENO2/HLLC dissipatif, plutot
que de supposer une equivalence non verifiee.

CL periodiques (via np.roll), coherent avec le reste du code 1D.
"""

import numpy as np


def reconstruct_high_order(V):
    """Reconstruction centree d'ordre 4 : moyenne de cellule -> valeur
    CENTRALE UNIQUE a la face (PAS deux etats gauche/droite -- voir
    l'avertissement du module ci-dessus).

    V : tableau (Nvars, Nx) ou (Nx,) de moyennes de cellule sur une grille
        periodique uniforme (l'axe spatial est toujours le DERNIER axe).
    Retourne V_face_centered, meme forme que V, ou V_face_centered[..., i]
    est la valeur reconstruite a la face i+1/2 (entre cellule i et
    cellule i+1). Pour i=Nx-1 (dernier indice), c'est la face entre la
    derniere et la premiere cellule (x=1 == x=0 en periodique).
    """
    V = np.asarray(V, dtype=float)
    axis = -1

    V_im1 = np.roll(V, shift=1, axis=axis)    # V[i-1]
    V_ip1 = np.roll(V, shift=-1, axis=axis)   # V[i+1]
    V_ip2 = np.roll(V, shift=-2, axis=axis)   # V[i+2]

    V_face_centered = (-V_im1 + 7.0 * V + 7.0 * V_ip1 - V_ip2) / 12.0
    return V_face_centered


def reconstruct_high_order_limited(V):
    """Variante LIMITEE de reconstruct_high_order (piste d'investigation :
    reduire/eliminer le depassement connu de 1/12 pres d'un saut net,
    cf. investigation de l'etalement d'interface).

    Principe : la valeur centree V_face_centered[i] est ECRETEE pour
    rester dans l'intervalle [min, max] du STENCIL COMPLET utilise pour la
    construire (V[i-1], V[i], V[i+1], V[i+2]) -- pas seulement les deux
    cellules immediatement adjacentes a la face (une version plus stricte,
    testee et rejetee : elle degradait l'ordre de 4 a ~2.5 meme en zone
    lisse, car une reconstruction d'ordre 4 peut legitimement depasser les
    deux valeurs immediatement adjacentes pres d'un extremum/d'une
    inflexion d'une fonction lisse courbee). Avec le stencil complet, le
    clip ne s'active QUE quand la reconstruction sort de la plage des 4
    valeurs qui l'ont produite -- ce qui n'arrive normalement qu'au
    voisinage d'un saut net, pas sur des donnees lisses courantes.
    """
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
