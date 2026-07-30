"""
face_operators.py
Etape 3A : definition et test EN ISOLATION (sans RK, sans capteur) des
TROIS operateurs de flux de face necessaires au schema hybride :

    F^HO        -- flux central non-dissipatif (etat unique reconstruit)
    F^ENO-HLLC  -- ENO2 (2 etats) -> solveur de Riemann HLLC
    F^LF        -- etats moyens voisins -> flux de Lax-Friedrichs (correcteur)

Architecture (suite a relecture) : chaque operateur retourne une structure
FaceFluxes explicite plutot qu'un tableau (3,Nx) opaque, pour rendre
visible le caractere QUASI-CONSERVATIF du modele double-flux :

    mass, momentum              -- PARTAGES entre les deux "proprietaires"
                                    de la face (formules independantes de
                                    gammaS/e0S)
    energy_left_owner            -- flux d'energie tel que VU/UTILISE par
                                    la cellule a GAUCHE de la face (avec
                                    SON PROPRE gammaS/e0S geles, Eq. 21)
    energy_right_owner           -- idem pour la cellule a DROITE

DECOUPLAGE VITESSES D'ONDE / ENERGIE (point de conception important) :
Pour F^ENO-HLLC et F^ENO-LF, les vitesses d'onde/le coefficient de
dissipation (a_L, a_R, d'ou s_L,s_R,sM,pM ou alpha) sont calcules a partir
de la VRAIE vitesse du son physique de chaque etat (get_sos, deja calculee
par ailleurs pour le CFL a chaque sous-pas RK -- pas de cout
supplementaire), PAS a partir d'un gammaS "gele" applique artificiellement
aux deux etats. Cela rend mass/momentum EXACTEMENT partages/conservatifs
(conforme a l'affirmation du papier : "mass, momentum ... are conserved
for finite-volume schemes"), et isole la dependance a gammaS/e0S gele a
la SEULE formule d'energie (E_L, E_R, Eq. 21), qui elle reste
owner-dependante (source de l'erreur de conservation d'energie du modele
double-flux, Eq. 23-24 du papier).

NOTE IMPORTANTE -- deviation assumee par rapport a numerics.py (HLLC_DF,
etape 1, deja validee qualitativement contre la Fig. 5 du papier) :
cette implementation anterieure utilisait un SEUL gammaS partage pour a_L
ET a_R au sein d'un meme appel (le meme gammaS gele applique aux DEUX
cotes de l'interface), rendant les vitesses d'onde -- et donc
potentiellement mass/momentum aussi -- owner-dependantes. C'etait une
approximation qui NE garantissait PAS la conservation exacte de
mass/momentum revendiquee par le papier. La conception ICI decouple
explicitement vitesses d'onde (physiques, partagees) et energie (gelee,
owner-dependante) -- plus fidele au texte du papier. Les deux
implementations restent disponibles ; test_face_operators_unit.py mesure
quantitativement l'ecart entre elles (diagnostic, pas une correction
retroactive de numerics.py).
"""

import numpy as np
from dataclasses import dataclass

from thermo import get_sos


@dataclass
class FaceFluxes:
    """Flux a un ensemble de faces, sous forme quasi-conservative explicite."""
    mass: np.ndarray                 # partage entre les deux proprietaires
    momentum: np.ndarray             # partage entre les deux proprietaires
    energy_left_owner: np.ndarray    # flux d'energie vu par la cellule a GAUCHE
    energy_right_owner: np.ndarray   # flux d'energie vu par la cellule a DROITE
    scheme_tag: str                  # 'HO', 'ENO_HLLC', ou 'ENO_LF'

    def energy_conservation_error(self):
        """Diagnostic : ecart entre les deux flux d'energie (nul pour un
        schema exactement conservatif, non nul par construction pour le
        double-flux -- cf. Eq. 23 du papier)."""
        return self.energy_right_owner - self.energy_left_owner


def flux_ho(V_face, gammaS_left, e0S_left, gammaS_right, e0S_right):
    """F^HO : flux central non-dissipatif, evalue directement sur l'etat
    unique reconstruit V_face = (rho, u, p) (sortie de
    reconstruction.reconstruct_high_order).

    mass/momentum sont PAR CONSTRUCTION owner-independants (aucune
    dependance a gammaS/e0S dans leurs formules) -- pas d'ambiguite ici.
    Seule l'energie differe selon le proprietaire (gammaS_left/right,
    e0S_left/right geles, un par cellule adjacente a la face).
    """
    rho, u, p = V_face[0, :], V_face[1, :], V_face[2, :]

    mass = rho * u
    momentum = rho * u**2 + p

    E_left = p / (gammaS_left - 1) + rho * e0S_left + 0.5 * rho * u**2
    E_right = p / (gammaS_right - 1) + rho * e0S_right + 0.5 * rho * u**2

    energy_left_owner = u * (E_left + p)
    energy_right_owner = u * (E_right + p)

    return FaceFluxes(mass, momentum, energy_left_owner, energy_right_owner, 'HO')


def _sign(s):
    return (s >= 0).astype(float) - (s <= 0).astype(float)


def flux_hllc_df(V_L, V_R, gammaS_left, e0S_left, gammaS_right, e0S_right):
    """F^ENO-HLLC : etats ENO2 (V_L, V_R) -> solveur de Riemann HLLC.

    Vitesses d'onde calculees a partir de la VRAIE vitesse du son physique
    de chaque etat (get_sos) -- rend mass/momentum exactement partages
    (voir note de module). L'energie (E_L, E_R) est calculee avec le
    gammaS/e0S GELE du proprietaire correspondant (gauche pour E_L, droit
    pour E_R), conformement a Eq. (21) : cette energie owner-dependante
    alimente a la fois le flux physique u*(E+p) ET la formule de l'etat
    etoile en energie -- d'ou deux resultats d'energie distincts
    (energy_left_owner, energy_right_owner) meme si mass/momentum restent
    uniques.
    """
    rho_L, rho_R = V_L[0, :], V_R[0, :]
    u_L, u_R = V_L[1, :], V_R[1, :]
    p_L, p_R = V_L[2, :], V_R[2, :]

    # vitesses d'onde a partir de la VRAIE vitesse du son (physique, non gelee)
    a_L = get_sos(V_L)
    a_R = get_sos(V_R)

    s_L = u_L - a_L
    s_R = u_R + a_R

    sM = (p_L - p_R - rho_L * u_L * (s_L - u_L) + rho_R * u_R * (s_R - u_R)) / (
        rho_R * (s_R - u_R) - rho_L * (s_L - u_L)
    )
    pM = rho_R * (u_R - s_R) * (u_R - sM) + p_R

    s_minus = np.minimum(0, s_L)
    s_plus = np.maximum(0, s_R)
    sgn = _sign(sM)

    # --- masse et quantite de mouvement : UNIQUES, partagees ---
    f_mass_L = rho_L * u_L
    f_mass_R = rho_R * u_R
    f_mom_L = rho_L * u_L**2 + p_L
    f_mom_R = rho_R * u_R**2 + p_R

    rho_starL = (s_L - u_L) / (s_L - sM) * rho_L
    rho_starR = (s_R - u_R) / (s_R - sM) * rho_R
    mom_starL = ((s_L - u_L) * rho_L * u_L + (pM - p_L)) / (s_L - sM)
    mom_starR = ((s_R - u_R) * rho_R * u_R + (pM - p_R)) / (s_R - sM)

    mass = (1 + sgn) / 2 * (f_mass_L + s_minus * (rho_starL - rho_L)) + \
           (1 - sgn) / 2 * (f_mass_R + s_plus * (rho_starR - rho_R))
    momentum = (1 + sgn) / 2 * (f_mom_L + s_minus * (mom_starL - rho_L * u_L)) + \
               (1 - sgn) / 2 * (f_mom_R + s_plus * (mom_starR - rho_R * u_R))

    # --- energie : DEUX versions, une par proprietaire (Eq. 21) ---
    def energy_flux_for_owner(gammaS_o, e0S_o):
        E_L = p_L / (gammaS_o - 1) + rho_L * e0S_o + 0.5 * rho_L * u_L**2
        E_R = p_R / (gammaS_o - 1) + rho_R * e0S_o + 0.5 * rho_R * u_R**2

        f_E_L = u_L * (E_L + p_L)
        f_E_R = u_R * (E_R + p_R)

        E_starL = ((s_L - u_L) * E_L - p_L * u_L + pM * sM) / (s_L - sM)
        E_starR = ((s_R - u_R) * E_R - p_R * u_R + pM * sM) / (s_R - sM)

        f_E = (1 + sgn) / 2 * (f_E_L + s_minus * (E_starL - E_L)) + \
              (1 - sgn) / 2 * (f_E_R + s_plus * (E_starR - E_R))
        return f_E

    energy_left_owner = energy_flux_for_owner(gammaS_left, e0S_left)
    energy_right_owner = energy_flux_for_owner(gammaS_right, e0S_right)

    return FaceFluxes(mass, momentum, energy_left_owner, energy_right_owner, 'ENO_HLLC')


def flux_llf_df(V_L, V_R, gammaS_left, e0S_left, gammaS_right, e0S_right):
    """F^LF : etats moyens des cellules adjacentes -> flux de Lax-Friedrichs local
    (Eq. 26 du papier), reserve au correcteur a posteriori.

    alpha (coefficient de dissipation) utilise la VRAIE vitesse du son
    (comme pour HLLC ci-dessus) -- choix standard, sans lien avec le gel
    double-flux (qui ne concerne que l'energie). mass/momentum partages ;
    energie owner-dependante (Eq. 21), meme logique que HLLC ci-dessus.
    """
    rho_L, rho_R = V_L[0, :], V_R[0, :]
    u_L, u_R = V_L[1, :], V_R[1, :]
    p_L, p_R = V_L[2, :], V_R[2, :]

    a_L = get_sos(V_L)
    a_R = get_sos(V_R)
    alpha = np.maximum(np.abs(u_L) + a_L, np.abs(u_R) + a_R)

    f_mass_L = rho_L * u_L
    f_mass_R = rho_R * u_R
    f_mom_L = rho_L * u_L**2 + p_L
    f_mom_R = rho_R * u_R**2 + p_R

    mass = 0.5 * (f_mass_L + f_mass_R) - 0.5 * alpha * (rho_R - rho_L)
    momentum = 0.5 * (f_mom_L + f_mom_R) - 0.5 * alpha * (rho_R * u_R - rho_L * u_L)

    def energy_flux_for_owner(gammaS_o, e0S_o):
        E_L = p_L / (gammaS_o - 1) + rho_L * e0S_o + 0.5 * rho_L * u_L**2
        E_R = p_R / (gammaS_o - 1) + rho_R * e0S_o + 0.5 * rho_R * u_R**2
        f_E_L = u_L * (E_L + p_L)
        f_E_R = u_R * (E_R + p_R)
        return 0.5 * (f_E_L + f_E_R) - 0.5 * alpha * (E_R - E_L)

    energy_left_owner = energy_flux_for_owner(gammaS_left, e0S_left)
    energy_right_owner = energy_flux_for_owner(gammaS_right, e0S_right)

    return FaceFluxes(mass, momentum, energy_left_owner, energy_right_owner, 'ENO_LF')


# ---------------------------------------------------------------------------
# Variantes FULLY-CONSERVATIVE (FC), etape 3C.
#
# IMPORTANT -- ce ne sont PAS les memes fonctions que ci-dessus avec juste
# des parametres differents : FC utilise une formule d'ENERGIE differente.
# En DF (Eq. 21), UN SEUL gammaS/e0S (celui du proprietaire) est applique
# UNIFORMEMENT aux DEUX etats (L et R) pour construire l'energie -- d'ou
# deux resultats distincts (energy_left_owner != energy_right_owner). En
# FC, chaque etat garde SA PROPRE identite thermodynamique REELLE
# (gammaS_L calcule a partir de V_L lui-meme, gammaS_R a partir de V_R
# lui-meme, PAS geles) -- il n'y a donc PAS de dualite proprietaire : un
# SEUL flux d'energie, exactement partage comme mass/momentum. C'est le
# meme principe que numerics_fc.py (etape 2), etendu ici a F^HO et
# F^ENO-HLLC.
# ---------------------------------------------------------------------------

def flux_ho_fc(V_face):
    """F^HO, variante FULLY-CONSERVATIVE : gammaS/e0S recalcules a partir
    de l'etat REEL V_face lui-meme (thermo.get_gammaS_e0S), pas geles, pas
    de dualite proprietaire -- un seul flux d'energie.

    energy_left_owner et energy_right_owner sont ICI TOUJOURS IDENTIQUES
    (retournes egaux par construction) : on les garde tous les deux dans
    FaceFluxes uniquement pour que l'appelant puisse traiter FC et DF de
    facon uniforme (meme interface), mais energy_conservation_error() est
    garanti nul pour ce flux -- teste explicitement.
    """
    from thermo import get_gammaS_e0S

    rho, u, p = V_face[0, :], V_face[1, :], V_face[2, :]

    mass = rho * u
    momentum = rho * u**2 + p

    gammaS, e0S = get_gammaS_e0S(V_face)
    E = p / (gammaS - 1) + rho * e0S + 0.5 * rho * u**2
    energy = u * (E + p)

    return FaceFluxes(mass, momentum, energy.copy(), energy.copy(), 'HO_FC')


def flux_hllc_fc(V_L, V_R):
    """F^ENO-HLLC, variante FULLY-CONSERVATIVE : gammaS_L/e0S_L calcules a
    partir de V_L lui-meme (thermo.get_gammaS_e0S), gammaS_R/e0S_R a
    partir de V_R lui-meme -- PAS geles, PAS de dualite proprietaire.
    Un SEUL flux d'energie (comme numerics_fc.py, etape 2), exactement
    partage comme mass/momentum.
    """
    from thermo import get_gammaS_e0S

    rho_L, rho_R = V_L[0, :], V_R[0, :]
    u_L, u_R = V_L[1, :], V_R[1, :]
    p_L, p_R = V_L[2, :], V_R[2, :]

    a_L = get_sos(V_L)
    a_R = get_sos(V_R)

    s_L = u_L - a_L
    s_R = u_R + a_R

    sM = (p_L - p_R - rho_L * u_L * (s_L - u_L) + rho_R * u_R * (s_R - u_R)) / (
        rho_R * (s_R - u_R) - rho_L * (s_L - u_L)
    )
    pM = rho_R * (u_R - s_R) * (u_R - sM) + p_R

    s_minus = np.minimum(0, s_L)
    s_plus = np.maximum(0, s_R)
    sgn = _sign(sM)

    f_mass_L = rho_L * u_L
    f_mass_R = rho_R * u_R
    f_mom_L = rho_L * u_L**2 + p_L
    f_mom_R = rho_R * u_R**2 + p_R

    rho_starL = (s_L - u_L) / (s_L - sM) * rho_L
    rho_starR = (s_R - u_R) / (s_R - sM) * rho_R
    mom_starL = ((s_L - u_L) * rho_L * u_L + (pM - p_L)) / (s_L - sM)
    mom_starR = ((s_R - u_R) * rho_R * u_R + (pM - p_R)) / (s_R - sM)

    mass = (1 + sgn) / 2 * (f_mass_L + s_minus * (rho_starL - rho_L)) + \
           (1 - sgn) / 2 * (f_mass_R + s_plus * (rho_starR - rho_R))
    momentum = (1 + sgn) / 2 * (f_mom_L + s_minus * (mom_starL - rho_L * u_L)) + \
               (1 - sgn) / 2 * (f_mom_R + s_plus * (mom_starR - rho_R * u_R))

    # --- energie : gammaS/e0S REELS, propres a chaque etat (PAS geles) ---
    gammaS_L, e0S_L = get_gammaS_e0S(V_L)
    gammaS_R, e0S_R = get_gammaS_e0S(V_R)

    E_L = p_L / (gammaS_L - 1) + rho_L * e0S_L + 0.5 * rho_L * u_L**2
    E_R = p_R / (gammaS_R - 1) + rho_R * e0S_R + 0.5 * rho_R * u_R**2

    f_E_L = u_L * (E_L + p_L)
    f_E_R = u_R * (E_R + p_R)

    E_starL = ((s_L - u_L) * E_L - p_L * u_L + pM * sM) / (s_L - sM)
    E_starR = ((s_R - u_R) * E_R - p_R * u_R + pM * sM) / (s_R - sM)

    energy = (1 + sgn) / 2 * (f_E_L + s_minus * (E_starL - E_L)) + \
             (1 - sgn) / 2 * (f_E_R + s_plus * (E_starR - E_R))

    return FaceFluxes(mass, momentum, energy.copy(), energy.copy(), 'ENO_HLLC_FC')


def flux_llf_fc(V_L, V_R):
    """F^LF, variante FULLY-CONSERVATIVE (meme principe que
    flux_hllc_fc) : gammaS_L/e0S_L depuis V_L lui-meme, gammaS_R/e0S_R
    depuis V_R lui-meme -- PAS geles, un SEUL flux d'energie, exactement
    partage comme mass/momentum."""
    from thermo import get_gammaS_e0S

    rho_L, rho_R = V_L[0, :], V_R[0, :]
    u_L, u_R = V_L[1, :], V_R[1, :]
    p_L, p_R = V_L[2, :], V_R[2, :]

    a_L = get_sos(V_L)
    a_R = get_sos(V_R)
    alpha = np.maximum(np.abs(u_L) + a_L, np.abs(u_R) + a_R)

    f_mass_L = rho_L * u_L
    f_mass_R = rho_R * u_R
    f_mom_L = rho_L * u_L**2 + p_L
    f_mom_R = rho_R * u_R**2 + p_R

    mass = 0.5 * (f_mass_L + f_mass_R) - 0.5 * alpha * (rho_R - rho_L)
    momentum = 0.5 * (f_mom_L + f_mom_R) - 0.5 * alpha * (rho_R * u_R - rho_L * u_L)

    gammaS_L, e0S_L = get_gammaS_e0S(V_L)
    gammaS_R, e0S_R = get_gammaS_e0S(V_R)
    E_L = p_L / (gammaS_L - 1) + rho_L * e0S_L + 0.5 * rho_L * u_L**2
    E_R = p_R / (gammaS_R - 1) + rho_R * e0S_R + 0.5 * rho_R * u_R**2
    f_E_L = u_L * (E_L + p_L)
    f_E_R = u_R * (E_R + p_R)
    energy = 0.5 * (f_E_L + f_E_R) - 0.5 * alpha * (E_R - E_L)

    return FaceFluxes(mass, momentum, energy.copy(), energy.copy(), 'ENO_LF_FC')


# ---------------------------------------------------------------------------
# Alias historiques de compatibilite.
#
# Les operateurs de Riemann sont independants de la reconstruction qui produit
# V_L et V_R. Les noms canoniques sont donc desormais flux_hllc_* et
# flux_llf_*. Les anciens noms sont conserves pour ne casser ni les scripts ni
# les tests existants. Les alias pointent vers les memes objets fonctions :
# aucune formule numerique n'est dupliquee ni modifiee.
# ---------------------------------------------------------------------------
flux_eno_hllc = flux_hllc_df
flux_eno_lf = flux_llf_df
flux_eno_hllc_fc = flux_hllc_fc
flux_eno_lf_fc = flux_llf_fc
