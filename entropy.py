"""
entropy.py
Fonction d'entropie de surrogate double-flux (Eq. 28) et principe du
minimum discret (Eq. 27) du papier Ma, Lv & Ihme (2017), JCP 340:330-357.

AVERTISSEMENT IMPORTANT (suite a relecture) :
Cette fonction n'est PAS l'entropie thermodynamique reelle de Peng-Robinson
(celle-ci est donnee par les Eq. 41-42 du papier, non implementee ici).
C'est la fonction d'entropie de gaz parfait adoptee par les auteurs pour
la representation "calorifiquement parfaite" propre au modele double-flux,
evaluee avec le gammaS effectif gele de la cellule cible. Pour une
comparaison Eq. (27), l etat candidat et les etats voisins servant a la
borne doivent etre evalues avec la MEME fonction d entropie locale, donc
avec le gammaS gele de la cellule cible. Son role est UNIQUEMENT de
servir de critere discret pour detecter les cellules "troubled" (Eq. 27) ;
sa valeur absolue n'a pas de sens thermodynamique dimensionne et ne doit
pas etre interpretee physiquement (voir la note en bas de fichier).

Pour un cas MONO-ESPECE (azote pur), le terme de melange de l'Eq. (28)
disparait : X_N2 = 1 partout, donc ln(X_N2) = 0. La formule se reduit a :

    s = ln(p) - gammaS * ln(rho * R)

(ecrite ainsi, plutot que ln(p / (rho*R)**gammaS), pour eviter le
debordement numerique de la puissance quand gammaS est grand, cf. Fig. 2
du papier ou gammaS ~ O(100) dans la region liquide-like).
"""

import numpy as np

# R specifique de l'azote (identique a thermo.get_thermo)
_MW_N2 = 28.0134e-3
R_N2 = 8.314 / _MW_N2


def entropy_double_flux(V, gamma_effective, R=R_N2):
    """Eq. (28) simplifiee au cas mono-espece, GELEE : s = ln(p) - gamma_effective*ln(rho*R).

    V : (3, N) = [rho, u, p]  (ou tout tableau dont V[0]=rho, V[2]=p)
    gamma_effective : le gammaS GELE au debut du pas RK courant (le meme
        tableau que celui utilise pour la mise a jour double-flux elle-meme,
        p.ex. via thermo.get_gammaS_e0S(V) appele UNE SEULE FOIS en debut
        de pas, PAS recalcule sur l'etat candidat).

    Ne PAS appeler get_gammaS_e0S(V_candidate) pour obtenir gamma_effective :
    cela changerait la fonction d'entropie elle-meme entre l'etat precedent
    et l'etat candidat, rendant la comparaison incoherente.
    """
    rho = V[0, :] if V.ndim == 2 else V[0]
    p = V[2, :] if V.ndim == 2 else V[2]

    s = np.log(p) - gamma_effective * np.log(rho * R)
    return s


def neighbor_entropy_bound(s_prev):
    """Construit s_min = min(s_{j-1}, s_j, s_{j+1}) pour chaque cellule j,
    a partir du champ d'entropie s_prev evalue a l'etat PRECEDENT (avant
    la mise a jour RK courante), comme l'exige l'Eq. (27) du papier :

        s(U_j^{k+1}) > min_{i in voisins(j)} s(U_i^k)

    C'est-a-dire s_prev doit toujours etre l'etat "k" (avant mise a jour),
    jamais l'etat candidat "k+1" qu'on cherche a tester -- se comparer a
    soi-meme rendrait le test trivial (toute cellule qui est un minimum
    local de son propre champ verifierait s_i == s_min,i, et serait donc
    marquee a tort par un test utilisant <=).
    """
    s_left = np.roll(s_prev, 1)
    s_right = np.roll(s_prev, -1)
    return np.minimum(np.minimum(s_left, s_prev), s_right)


def entropy_minimum_check(s_candidate, s_prev, atol=1.0e-12, rtol=1.0e-10):
    """Eq. (27) avec tolerance numerique : detecte les cellules "troubled".

    s_candidate : entropie (entropy_double_flux) de l'etat APRES l'etage RK
        courant (le "k+1").
    s_prev : entropie (entropy_double_flux, MEME gamma_effective) de l'etat
        AVANT l'etage RK courant (le "k") -- PAS l'etat candidat. La borne
        de voisinage est construite ICI, en interne, a partir de s_prev.

    Une cellule est "troubled" si :
        s_candidate_j < s_min_j - tol_j
    avec tol_j = atol + rtol * max(1, |s_min_j|) -- tolerance relative pour
    eviter les faux positifs causes par le bruit flottant (~1e-14) quand
    l'etat est stationnaire ou constant.
    """
    s_min = neighbor_entropy_bound(s_prev)
    tol = atol + rtol * np.maximum(1.0, np.abs(s_min))
    invalid = ~np.isfinite(s_candidate)
    troubled = invalid | (s_candidate < (s_min - tol))
    return troubled


# ---------------------------------------------------------------------------
# Note sur l'interpretation physique (suite a relecture) :
#
# La quantite s = ln(p) - gamma*ln(rho*R) est le logarithme d'une grandeur
# dimensionnee (p a des Pa, (rho*R) a des unites de pression/K^gamma...).
# Sa valeur absolue depend donc entierement du systeme d'unites choisi et
# NE CORRESPOND PAS a une entropie en J/(kg.K). Ce n'est pas un defaut :
# le principe du minimum discret (Eq. 27) n'utilise que l'ORDRE RELATIF
# entre valeurs evaluees avec la MEME convention et le MEME gamma gele --
# jamais la magnitude absolue. Ne pas chercher a interpreter physiquement
# une valeur de s isolee (p.ex. "s=-545" n'a pas de signification en soi).
# ---------------------------------------------------------------------------
