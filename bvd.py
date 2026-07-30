"""
bvd.py

Etape 3 : critere BVD en mode strictement diagnostique.

Le module compare deux candidats CELLULAIRES disposant chacun d'une valeur
sur la face gauche et sur la face droite de chaque cellule. Il ne modifie ni
les reconstructions utilisees par le solveur, ni les flux, ni le RHS.

Pourquoi le candidat polynomial est ENO2 dans cette etape
----------------------------------------------------------
La reconstruction centree d'ordre 4 du solveur fournit UNE valeur unique par
face. Si cette meme valeur etait dupliquee comme etat gauche et droit, le saut
reconstruit a chaque face serait identiquement nul et le critere BVD serait
trivial. Le candidat polynomial diagnostique est donc ici ENO2, qui fournit
deux traces distinctes et permet une comparaison mathematiquement pertinente
avec THINC.

Convention
----------
Pour chaque cellule i :

    q_left_cell[i]  = trace issue de i a la face i-1/2
    q_right_cell[i] = trace issue de i a la face i+1/2

Le saut du candidat k a la face i+1/2 est

    J_k[i] = |q_left_cell[i+1] - q_right_cell[i]|.

La total boundary variation de la cellule i est

    TBV_k[i] = J_k[i-1] + J_k[i].

Il s'agit d'une comparaison "pure-candidate" : le TBV polynomial est calcule
quand toutes les cellules emploient le candidat polynomial, et le TBV THINC
quand toutes emploient THINC. La selection couplee finale, tenant compte des
combinaisons entre candidats voisins, sera introduite seulement lorsque BVD
sera effectivement injecte dans les etats de face.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BVDDiagnostics:
    """Resultats diagnostiques de la comparaison de deux candidats."""

    jump_polynomial_face: np.ndarray
    jump_thinc_face: np.ndarray
    tbv_polynomial_cell: np.ndarray
    tbv_thinc_cell: np.ndarray
    use_thinc_cell: np.ndarray

    @property
    def thinc_fraction(self):
        """Fraction des cellules ou THINC reduit strictement le TBV."""
        return float(np.mean(self.use_thinc_cell))

    @property
    def tbv_gain(self):
        """Gain positif lorsque THINC diminue le TBV."""
        return self.tbv_polynomial_cell - self.tbv_thinc_cell


def _as_1d_finite(name, values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} doit etre un tableau 1D, forme obtenue {array.shape}")
    if array.size < 3:
        raise ValueError(f"{name} doit contenir au moins 3 cellules periodiques")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contient une valeur non finie")
    return array


def face_boundary_jump(q_left_cell, q_right_cell):
    """Retourne le saut absolu a chaque face i+1/2.

    Parameters
    ----------
    q_left_cell, q_right_cell : ndarray, shape (Nx,)
        Traces issues de chaque cellule sur ses faces gauche et droite.
    """
    q_left_cell = _as_1d_finite("q_left_cell", q_left_cell)
    q_right_cell = _as_1d_finite("q_right_cell", q_right_cell)
    if q_left_cell.shape != q_right_cell.shape:
        raise ValueError("q_left_cell et q_right_cell doivent avoir la meme forme")

    q_right_state_at_face = np.roll(q_left_cell, -1)  # issu de la cellule i+1
    q_left_state_at_face = q_right_cell               # issu de la cellule i
    return np.abs(q_right_state_at_face - q_left_state_at_face)


def total_boundary_variation(q_left_cell, q_right_cell):
    """Retourne ``(tbv_cell, jump_face)`` pour un candidat periodique."""
    jump_face = face_boundary_jump(q_left_cell, q_right_cell)
    tbv_cell = np.roll(jump_face, 1) + jump_face
    return tbv_cell, jump_face


def eno_face_states_to_cell_boundaries(q_left_face, q_right_face):
    """Convertit les traces ENO historiques en traces par cellule.

    Les fonctions ENO du solveur utilisent :

        q_left_face[i]  = trace de la cellule i a la face i+1/2
        q_right_face[i] = trace de la cellule i+1 a la face i+1/2

    D'ou, pour la cellule i :

        q_right_cell[i] = q_left_face[i]
        q_left_cell[i]  = q_right_face[i-1].
    """
    q_left_face = _as_1d_finite("q_left_face", q_left_face)
    q_right_face = _as_1d_finite("q_right_face", q_right_face)
    if q_left_face.shape != q_right_face.shape:
        raise ValueError("q_left_face et q_right_face doivent avoir la meme forme")

    q_left_cell = np.roll(q_right_face, 1)
    q_right_cell = q_left_face.copy()
    return q_left_cell, q_right_cell


def compare_bvd_candidates(
    polynomial_left_cell,
    polynomial_right_cell,
    thinc_left_cell,
    thinc_right_cell,
    *,
    atol=1.0e-14,
    rtol=1.0e-12,
):
    """Compare le TBV polynomial et THINC sans modifier le solveur.

    Une egalite, y compris a la tolerance numerique, est tranchee en faveur
    du candidat polynomial. THINC est selectionne seulement si son TBV est
    strictement plus petit :

        TBV_T + atol + rtol*scale < TBV_P.
    """
    arrays = [
        _as_1d_finite("polynomial_left_cell", polynomial_left_cell),
        _as_1d_finite("polynomial_right_cell", polynomial_right_cell),
        _as_1d_finite("thinc_left_cell", thinc_left_cell),
        _as_1d_finite("thinc_right_cell", thinc_right_cell),
    ]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays[1:]):
        raise ValueError("Tous les candidats doivent avoir la meme forme")

    atol = float(atol)
    rtol = float(rtol)
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol doit etre fini et positif ou nul")
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError("rtol doit etre fini et positif ou nul")

    tbv_p, jump_p = total_boundary_variation(arrays[0], arrays[1])
    tbv_t, jump_t = total_boundary_variation(arrays[2], arrays[3])

    scale = np.maximum(tbv_p, tbv_t)
    tolerance = atol + rtol * scale
    use_thinc = tbv_t + tolerance < tbv_p

    return BVDDiagnostics(
        jump_polynomial_face=jump_p,
        jump_thinc_face=jump_t,
        tbv_polynomial_cell=tbv_p,
        tbv_thinc_cell=tbv_t,
        use_thinc_cell=use_thinc,
    )
