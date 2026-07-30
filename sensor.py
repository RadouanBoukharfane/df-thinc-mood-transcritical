"""
sensor.py
Etape 3B : capteur RS (Eq. 25 du papier), teste EN ISOLATION avant tout
branchement au RHS hybride ou au RK3.

Trois questions de conception tranchees explicitement ici (suite a
relecture) :

1. CELLULE OU FACE ?
   Eq. 25 est formulee "pour chaque CV" (cellule), mais une face separe
   deux cellules qui peuvent chacune porter un jugement different. On
   implemente ICI DEUX variantes, testees et comparees plutot que d'en
   choisir une sans preuve :
     - rs_face_mask_direct(rho) : verification DIRECTE par face, contre
       les DEUX cv adjacentes a CETTE face precise (lecture la plus
       litterale d'Eq. 25 : "pour chaque CV, le capteur determine si...").
     - rs_face_mask_from_cells(rho) : agregation a la Eq. 25 par cellule
       (M_i = pire des deux faces de la cellule i), PUIS regle de face
       conservatrice M_{i+1/2} = max(M_i, M_{i+1}) -- suggestion de
       relecture, plus permissive (une cellule "troubled" ailleurs
       "contamine" ses deux faces).
   Les deux sont fournies ; test_sensor_unit.py mesure leur difference.

2. QUELLE RECONSTRUCTION DE DENSITE ENTRE DANS LE CAPTEUR ?
   EXACTEMENT reconstruct_high_order(rho) -- la meme reconstruction que
   celle que le capteur doit juger, pas une approximation separee.

3. QUAND EST-IL RECALCULE ?
   A CHAQUE sous-etape RK, sur l'etat COURANT de cette sous-etape (pas
   gele). C'est deliberement DECOUPLE du gel de gammaS/e0S (qui repond a
   une autre necessite -- coherence thermodynamique du modele
   double-flux -- et reste gele sur tout le pas de temps, pas par
   sous-etape). Le capteur, lui, doit refleter la regularite REELLE de
   l'etat courant a chaque evaluation.
"""

import numpy as np

from reconstruction import reconstruct_high_order


def rs_raw_indicator(rho, zeta=None):
    """Indicateur brut d'Eq. 25 : |rho_face - rho_cv| / rho_cv, evalue
    face par face, separement pour chacune des deux cellules adjacentes.

    Retourne (indicator_left, indicator_right), chacun de forme (Nx,) :
        indicator_left[i]  = |rho_face[i] - rho[i]|   / rho[i]     (cellule i,
                              a GAUCHE de la face i+1/2)
        indicator_right[i] = |rho_face[i] - rho[i+1]| / rho[i+1]   (cellule i+1,
                              a DROITE de la face i+1/2)
    """
    rho = np.asarray(rho, dtype=float)
    rho_face = reconstruct_high_order(rho)  # rho_face[i] = face i+1/2

    rho_ip1 = np.roll(rho, -1)  # cellule i+1

    indicator_left = np.abs(rho_face - rho) / rho
    indicator_right = np.abs(rho_face - rho_ip1) / rho_ip1
    return indicator_left, indicator_right


def rs_face_mask_direct(rho, zeta=0.2):
    """Variante DIRECTE : la face i+1/2 est marquee ENO si sa propre valeur
    reconstruite differe de plus de zeta de L'UNE OU L'AUTRE des deux
    cellules qui la bordent DIRECTEMENT (i ou i+1) -- lecture la plus
    litterale d'Eq. 25, sans propagation via des faces non adjacentes."""
    indicator_left, indicator_right = rs_raw_indicator(rho)
    indicator_face = np.maximum(indicator_left, indicator_right)
    mask = indicator_face > zeta  # True = ENO2, False = HO
    return mask, indicator_face


def rs_cell_flag(rho, zeta=0.2):
    """M_i : la cellule i est "troubled" si L'UNE OU L'AUTRE de ses DEUX
    faces (i-1/2 et i+1/2) declenche Eq. 25 avec la densite de CETTE
    cellule. Etape intermediaire pour rs_face_mask_from_cells."""
    indicator_left, indicator_right = rs_raw_indicator(rho)
    # indicator_left[i]  = indicateur de la face i+1/2 vu par la cellule i (a sa DROITE)
    # indicator_right[i] = indicateur de la face i+1/2 vu par la cellule i+1 (a sa GAUCHE)
    # Pour la cellule i, sa face DROITE (i+1/2) utilise indicator_left[i].
    # Sa face GAUCHE (i-1/2) utilise indicator_right[i-1] (= vu depuis la
    # cellule i, cote droit de la face i-1/2).
    face_right_of_i = indicator_left               # indicator_left[i]
    face_left_of_i = np.roll(indicator_right, 1)    # indicator_right[i-1]

    M_i = np.maximum(face_right_of_i, face_left_of_i) > zeta
    return M_i


def rs_face_mask_from_cells(rho, zeta=0.2):
    """Variante par AGREGATION CELLULE PUIS FACE (suggestion de relecture) :
    M_{i+1/2} = max(M_i, M_{i+1}), avec M_i defini par rs_cell_flag.
    Plus PERMISSIVE que rs_face_mask_direct (une cellule troubled sur une
    face contamine aussi son AUTRE face, meme si celle-ci est individuellement
    lisse)."""
    M = rs_cell_flag(rho, zeta)
    M_ip1 = np.roll(M, -1)
    mask = np.logical_or(M, M_ip1)
    return mask


def eno_fraction(mask):
    """f_ENO = fraction de faces traitees en ENO2 (metrique de suivi,
    utile a chaque sous-etape pour verifier que le capteur se comporte
    comme attendu -- cf. tests plus bas)."""
    return np.mean(mask)


# ---------------------------------------------------------------------------
# Etape 4 : detecteur contact--choc, strictement diagnostique
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass(frozen=True)
class ContactShockDiagnostics:
    """Classification diagnostique des cellules et des faces.

    Les masques sont mutuellement exclusifs avec la priorite
    ``shock > contact > smooth``.  A cette etape, ils ne modifient ni la
    reconstruction, ni les flux, ni le RHS.
    """

    density_jump_face: np.ndarray
    pressure_jump_face: np.ndarray
    velocity_jump_face: np.ndarray
    compression_face: np.ndarray
    contact_score_face: np.ndarray
    shock_score_face: np.ndarray
    contact_face: np.ndarray
    shock_face: np.ndarray
    smooth_face: np.ndarray
    contact_cell: np.ndarray
    shock_cell: np.ndarray
    smooth_cell: np.ndarray

    @property
    def contact_fraction(self):
        return float(np.mean(self.contact_cell))

    @property
    def shock_fraction(self):
        return float(np.mean(self.shock_cell))

    @property
    def smooth_fraction(self):
        return float(np.mean(self.smooth_cell))

    @property
    def allow_thinc_cell(self):
        """THINC est autorise seulement dans les cellules classees contact."""
        return self.contact_cell.copy()


def _validate_primitive_state(V):
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] < 3:
        raise ValueError(
            "V doit etre de forme (Nvars, Nx), avec au moins (rho, u, p)"
        )
    if V.shape[1] < 3:
        raise ValueError("Au moins trois cellules periodiques sont requises")
    if not np.all(np.isfinite(V[:3])):
        raise ValueError("V contient une valeur non finie dans (rho, u, p)")
    if np.any(V[0] <= 0.0):
        raise ValueError("La densite doit etre strictement positive")
    if np.any(V[2] <= 0.0):
        raise ValueError("La pression doit etre strictement positive")
    return V


def contact_shock_face_indicators(
    V,
    *,
    eps=1.0e-14,
    velocity_floor=1.0,
    contact_pressure_weight=20.0,
    contact_velocity_weight=20.0,
    shock_compression_weight=1.0,
):
    """Calcule les indicateurs dimensionnels reduits a chaque face i+1/2.

    Definitions
    -----------
    ``S_rho`` mesure le saut relatif de densite.
    ``S_p`` mesure le saut relatif de pression.
    ``S_u`` mesure le saut relatif de vitesse, avec une vitesse plancher pour
    eviter qu'un ecoulement presque au repos ne produise un rapport singulier.
    ``C`` mesure la compression locale, nulle en expansion.

    Le score de contact est grand lorsque la densite saute alors que pression
    et vitesse restent presque constantes :

        chi_c = S_rho exp(-Cp S_p - Cu S_u).

    Le score de choc combine le saut de pression et la compression :

        chi_s = S_p (1 + Cc C).
    """
    V = _validate_primitive_state(V)

    eps = float(eps)
    velocity_floor = float(velocity_floor)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps doit etre strictement positif et fini")
    if not np.isfinite(velocity_floor) or velocity_floor <= 0.0:
        raise ValueError("velocity_floor doit etre strictement positif et fini")

    rho, u, p = V[0], V[1], V[2]
    rho_r = np.roll(rho, -1)
    u_r = np.roll(u, -1)
    p_r = np.roll(p, -1)

    density_jump = np.abs(rho_r - rho) / (np.abs(rho_r) + np.abs(rho) + eps)
    pressure_jump = np.abs(p_r - p) / (np.abs(p_r) + np.abs(p) + eps)

    velocity_scale = np.maximum(
        np.abs(u_r) + np.abs(u),
        velocity_floor,
    )
    velocity_jump = np.abs(u_r - u) / (velocity_scale + eps)
    compression = np.maximum(0.0, -(u_r - u)) / (velocity_scale + eps)

    contact_score = density_jump * np.exp(
        -float(contact_pressure_weight) * pressure_jump
        -float(contact_velocity_weight) * velocity_jump
    )
    shock_score = pressure_jump * (
        1.0 + float(shock_compression_weight) * compression
    )

    return {
        "density_jump_face": density_jump,
        "pressure_jump_face": pressure_jump,
        "velocity_jump_face": velocity_jump,
        "compression_face": compression,
        "contact_score_face": contact_score,
        "shock_score_face": shock_score,
    }


def classify_contact_shock(
    V,
    *,
    contact_threshold=5.0e-2,
    shock_pressure_threshold=5.0e-2,
    shock_compression_threshold=1.0e-3,
    shock_velocity_threshold=5.0e-2,
    eps=1.0e-14,
    velocity_floor=1.0,
    contact_pressure_weight=20.0,
    contact_velocity_weight=20.0,
    shock_compression_weight=1.0,
):
    """Classe faces et cellules en ``smooth``, ``contact`` ou ``shock``.

    Une face est classee choc si son saut de pression depasse le seuil et si
    elle est compressive ou porte un saut de vitesse significatif.  Une face
    non classee choc est classee contact si son score de contact depasse le
    seuil.  Les autres faces sont lisses.

    Une cellule herite de la classe de ses deux faces. La priorite est choc,
    puis contact, puis lisse. Cette priorite empeche THINC d'etre autorise dans
    une cellule adjacente a un choc.
    """
    thresholds = {
        "contact_threshold": contact_threshold,
        "shock_pressure_threshold": shock_pressure_threshold,
        "shock_compression_threshold": shock_compression_threshold,
        "shock_velocity_threshold": shock_velocity_threshold,
    }
    for name, value in thresholds.items():
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} doit etre fini et positif ou nul")

    ind = contact_shock_face_indicators(
        V,
        eps=eps,
        velocity_floor=velocity_floor,
        contact_pressure_weight=contact_pressure_weight,
        contact_velocity_weight=contact_velocity_weight,
        shock_compression_weight=shock_compression_weight,
    )

    pressure_jump = ind["pressure_jump_face"]
    velocity_jump = ind["velocity_jump_face"]
    compression = ind["compression_face"]
    contact_score = ind["contact_score_face"]

    shock_face = (
        pressure_jump > shock_pressure_threshold
    ) & (
        (compression > shock_compression_threshold)
        | (velocity_jump > shock_velocity_threshold)
    )
    contact_face = (~shock_face) & (contact_score > contact_threshold)
    smooth_face = ~(shock_face | contact_face)

    # Pour la cellule i, les faces sont i-1/2 -> index i-1 et i+1/2 -> i.
    shock_cell = shock_face | np.roll(shock_face, 1)
    contact_cell = (~shock_cell) & (contact_face | np.roll(contact_face, 1))
    smooth_cell = ~(shock_cell | contact_cell)

    return ContactShockDiagnostics(
        density_jump_face=ind["density_jump_face"],
        pressure_jump_face=pressure_jump,
        velocity_jump_face=velocity_jump,
        compression_face=compression,
        contact_score_face=contact_score,
        shock_score_face=ind["shock_score_face"],
        contact_face=contact_face,
        shock_face=shock_face,
        smooth_face=smooth_face,
        contact_cell=contact_cell,
        shock_cell=shock_cell,
        smooth_cell=smooth_cell,
    )
