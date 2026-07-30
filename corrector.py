"""
corrector.py
Etape 3E (generalise pour l'etape 3F) : correcteur a posteriori
(Algorithme 2 du papier).

INTERFACE GENERALISEE (necessaire pour SSP-RK3, etape 3F) : le RHS
hybride est evalue sur un etat V_eval, mais le candidat mis a jour est
W_before + dt_eff*L(V_eval) -- V_eval et primitives(W_before) NE
COINCIDENT PAS en general pour les sous-etapes 2 et 3 du RK3 (combinaisons
convexes : k2 = 3/4 W + 1/4 k1 + 1/4 dt L(Vk1), le "W_before" pour cette
correction est 3/4 W + 1/4 k1, mais le RHS est evalue sur Vk1). D'ou la
signature (V_eval, W_before, s_prev, ..., dt_eff) plutot que le simple
"V0, dt" de la version precedente (qui correspondait au cas particulier
V_eval == primitives(W_before), dt_eff == dt -- toujours vrai pour un
Euler forward isole, etape 3D, mais pas pour RK2/RK3).

s_prev (entropie de reference pour le principe du minimum, Eq. 27) doit
etre l'entropie du stage PRECEDENT (pas forcement l'etat tout au debut du
pas de temps) -- fourni explicitement par l'appelant.

Logique de correction (inchangee depuis l'etape 3E) :
    1. calculer les flux hybrides de base sur V_eval ;
    2. construire le candidat W_after = W_before + dt_eff*L(V_eval) ;
    3. detecter cellules non admissibles + troubled (entropie) ;
    4. marquer leurs faces gauche/droite ;
    5. remplacer les flux de ces faces par F^ENO-LF ;
    6. reconstruire le candidat ;
    7. repeter jusqu'a stabilisation, avec limite d'iterations et echec
       explicite (CorrectionFailedError) sinon.
"""

import numpy as np

from thermo import get_gammaS_e0S
from numerics import update_primitives_df, primitives_to_conservatives
from numerics_fc import update_primitives_fc
from entropy import entropy_double_flux, entropy_minimum_check
from hybrid_rhs import (
    get_hybrid_face_fluxes_fc, get_hybrid_face_fluxes_df,
    _assemble_rhs_fc, _assemble_rhs_df,
)
from face_operators import flux_llf_df, flux_llf_fc


class CorrectionFailedError(Exception):
    """Levee quand le correcteur n'a pas stabilise le candidat apres
    max_iter iterations. Porte l'historique complet pour diagnostic."""

    def __init__(self, message, history, troubled_final):
        super().__init__(message)
        self.history = history
        self.troubled_final = troubled_final


def _faces_adjacent_to(troubled):
    """Faces GAUCHE et DROITE des cellules troubled : face[i] est
    adjacente si la cellule i (a gauche de face[i]) OU la cellule i+1
    (a droite de face[i]) est troubled."""
    return troubled | np.roll(troubled, -1)


def correct_hybrid_rhs_fc(V_eval, W_before, s_prev, gammaS_ref, dx, dt_eff,
                           zeta=0.2, mask_variant='direct', max_iter=10,
                           atol=1e-12, rtol=1e-9, limit_ho=False, verbose=False):
    """Un sous-pas W_after = W_before + dt_eff*L(V_eval), mode
    FULLY-CONSERVATIVE, avec correction a posteriori.

    gammaS_ref : reference FIXE pour le critere d'entropie (FC ne gele
    rien par ailleurs ; sert uniquement a donner une base de comparaison
    coherente a Eq. 27 entre s_prev et l'etat candidat).

    limit_ho : transmis a get_hybrid_face_fluxes_fc (defaut False =
        methode d'origine fidele au papier ; True = variante robuste,
        cf. hybrid_rhs.py et INVESTIGATION_SMEARING.md).
    """
    Nx = V_eval.shape[1]

    mass, momentum, energy, mask, V_L_eno, V_R_eno, limited_mask = get_hybrid_face_fluxes_fc(
        V_eval, dx, zeta, mask_variant, limit_ho=limit_ho)

    corrected_faces = np.zeros(Nx, dtype=bool)
    history = []

    for it in range(max_iter):
        RHS = _assemble_rhs_fc(mass, momentum, energy, dx)
        W_after = W_before + dt_eff * RHS

        rho_bad = (~np.isfinite(W_after[0, :])) | (W_after[0, :] <= 0)
        bad_finite = ~np.isfinite(W_after).all(axis=0)
        p_bad = np.zeros(Nx, dtype=bool)
        entropy_bad = np.zeros(Nx, dtype=bool)
        V_after = None

        if rho_bad.any() or bad_finite.any():
            troubled = rho_bad | bad_finite
        else:
            try:
                V_after = update_primitives_fc(W_after)
                bad_finite = ~np.isfinite(V_after).all(axis=0)
                p_bad = (~np.isfinite(V_after[2, :])) | (V_after[2, :] <= 0)
                with np.errstate(invalid='ignore', divide='ignore'):
                    s_candidate = entropy_double_flux(V_after, gammaS_ref)
                entropy_bad = entropy_minimum_check(s_candidate, s_prev, atol, rtol)
                troubled = bad_finite | p_bad | entropy_bad
            except (RuntimeError, ValueError):
                bad_finite = np.ones(Nx, dtype=bool)
                troubled = bad_finite.copy()

        min_rho = float(np.nanmin(W_after[0, :])) if np.isfinite(W_after[0, :]).any() else float('nan')
        min_p = (float(np.nanmin(V_after[2, :])) if V_after is not None and np.isfinite(V_after[2, :]).any()
                 else float('nan'))
        history.append(dict(iteration=it, n_troubled=int(troubled.sum()),
                             n_corrected_faces=int(corrected_faces.sum()),
                             n_bad_density=int(rho_bad.sum()), n_bad_pressure=int(p_bad.sum()),
                             n_bad_finite=int(bad_finite.sum()), n_bad_entropy=int(entropy_bad.sum()),
                             troubled_indices=np.flatnonzero(troubled).tolist(),
                             min_density=min_rho, min_pressure=min_p))
        if verbose:
            print(f'  [FC] iteration {it}: {troubled.sum()} cellules troubled, '
                  f'{corrected_faces.sum()} faces deja corrigees')

        if not troubled.any():
            return dict(W_after=W_after, V_after=V_after, mask=mask,
                        corrected_faces=corrected_faces, n_iterations=it,
                        converged=True, history=history, limited_mask=limited_mask)

        # Algorithme 2 : corriger les cellules troubled ET leurs voisines,
        # puis toutes les faces contigues a cet ensemble.
        cells_to_correct = troubled | np.roll(troubled, 1) | np.roll(troubled, -1)
        new_faces = _faces_adjacent_to(cells_to_correct)
        newly_added = new_faces & ~corrected_faces
        if not newly_added.any():
            break
        corrected_faces |= new_faces

        idx = np.where(corrected_faces)[0]
        # Eq. 26 : le flux LF de secours utilise les moyennes de cellules
        # adjacentes, et non les etats ENO2 reconstruits.
        V_L_lf = V_eval[:, idx]
        V_R_lf = np.roll(V_eval, -1, axis=1)[:, idx]
        ff_lf = flux_llf_fc(V_L_lf, V_R_lf)
        mass[idx] = ff_lf.mass
        momentum[idx] = ff_lf.momentum
        energy[idx] = ff_lf.energy_left_owner

    raise CorrectionFailedError(
        f"[FC] Correction non convergee apres {max_iter} iterations "
        f"({history[-1]['n_troubled']} cellules encore troubled).",
        history, troubled)


def correct_hybrid_rhs_df(
    V_eval,
    W_before,
    s_prev,
    gammaS_frozen,
    e0S_frozen,
    dx,
    dt_eff,
    zeta=0.2,
    mask_variant='direct',
    max_iter=10,
    atol=1e-12,
    rtol=1e-9,
    limit_ho=False,
    use_thinc_bvd=False,
    thinc_bvd_options=None,
    verbose=False,
):
    """Un sous-pas W_after = W_before + dt_eff*L(V_eval), mode
    DOUBLE-FLUX, avec correction a posteriori. gammaS_frozen/e0S_frozen
    geles sur tout le PAS DE TEMPS complet (memes valeurs aux 3 sous-etapes
    RK), fournis en argument.

    limit_ho : transmis a get_hybrid_face_fluxes_df (defaut False =
        methode d'origine fidele au papier).
    """
    Nx = V_eval.shape[1]

    flux_data = get_hybrid_face_fluxes_df(
        V_eval,
        gammaS_frozen,
        e0S_frozen,
        dx,
        zeta,
        mask_variant,
        limit_ho=limit_ho,
        use_thinc_bvd=use_thinc_bvd,
        **(thinc_bvd_options or {}),
    )
    (
        mass,
        momentum,
        energy_left,
        energy_right,
        mask,
        V_L_eno,
        V_R_eno,
        gammaS_left,
        e0S_left,
        gammaS_right,
        e0S_right,
        limited_mask,
        thinc_face_mask,
        use_thinc_cell,
    ) = flux_data

    corrected_faces = np.zeros(Nx, dtype=bool)
    history = []

    for it in range(max_iter):
        RHS = _assemble_rhs_df(mass, momentum, energy_left, energy_right, dx)
        W_after = W_before + dt_eff * RHS

        rho_bad = (~np.isfinite(W_after[0, :])) | (W_after[0, :] <= 0)
        bad_finite = ~np.isfinite(W_after).all(axis=0)
        p_bad = np.zeros(Nx, dtype=bool)
        entropy_bad = np.zeros(Nx, dtype=bool)
        V_after = None

        if rho_bad.any() or bad_finite.any():
            troubled = rho_bad | bad_finite
        else:
            V_after = update_primitives_df(W_after, gammaS_frozen, e0S_frozen)
            bad_finite = ~np.isfinite(V_after).all(axis=0)
            p_bad = (~np.isfinite(V_after[2, :])) | (V_after[2, :] <= 0)
            with np.errstate(invalid='ignore', divide='ignore'):
                s_candidate = entropy_double_flux(V_after, gammaS_frozen)
                # Pour chaque cellule j, evaluer aussi les etats voisins
                # avec la fonction d'entropie locale gelee gamma*_j.
                rho0 = V_eval[0, :]
                p0 = V_eval[2, :]
                from entropy import R_N2
                s_center_j = np.log(p0) - gammaS_frozen * np.log(rho0 * R_N2)
                s_left_j = np.log(np.roll(p0, 1)) - gammaS_frozen * np.log(np.roll(rho0, 1) * R_N2)
                s_right_j = np.log(np.roll(p0, -1)) - gammaS_frozen * np.log(np.roll(rho0, -1) * R_N2)
                _bound = np.minimum(np.minimum(s_left_j, s_center_j), s_right_j)
            _tol = atol + rtol * np.maximum(1.0, np.abs(_bound))
            entropy_bad = (~np.isfinite(s_candidate)) | (s_candidate < (_bound - _tol))
            _margin = s_candidate - _bound
            troubled = bad_finite | p_bad | entropy_bad

        min_rho = float(np.nanmin(W_after[0, :])) if np.isfinite(W_after[0, :]).any() else float('nan')
        min_p = (float(np.nanmin(V_after[2, :])) if V_after is not None and np.isfinite(V_after[2, :]).any()
                 else float('nan'))
        history.append(dict(iteration=it, n_troubled=int(troubled.sum()),
                             n_corrected_faces=int(corrected_faces.sum()),
                             n_bad_density=int(rho_bad.sum()), n_bad_pressure=int(p_bad.sum()),
                             n_bad_finite=int(bad_finite.sum()), n_bad_entropy=int(entropy_bad.sum()),
                             troubled_indices=np.flatnonzero(troubled).tolist(),
                             min_density=min_rho, min_pressure=min_p,
                             min_entropy_margin=(float(np.nanmin(_margin)) if V_after is not None else float('nan'))))
        if verbose:
            print(f'  [DF] iteration {it}: {troubled.sum()} cellules troubled, '
                  f'{corrected_faces.sum()} faces deja corrigees')

        if not troubled.any():
            return dict(
                W_after=W_after,
                V_after=V_after,
                mask=mask,
                corrected_faces=corrected_faces,
                n_iterations=it,
                converged=True,
                history=history,
                limited_mask=limited_mask,
                thinc_face_mask=thinc_face_mask,
                use_thinc_cell=use_thinc_cell,
                mass_flux=mass.copy(),
            )

        cells_to_correct = troubled | np.roll(troubled, 1) | np.roll(troubled, -1)
        new_faces = _faces_adjacent_to(cells_to_correct)
        newly_added = new_faces & ~corrected_faces
        if not newly_added.any():
            break
        corrected_faces |= new_faces

        idx = np.where(corrected_faces)[0]
        # Eq. 26 : etats moyens des cellules adjacentes.
        V_L_lf = V_eval[:, idx]
        V_R_lf = np.roll(V_eval, -1, axis=1)[:, idx]
        ff_lf = flux_llf_df(V_L_lf, V_R_lf,
                             gammaS_left[idx], e0S_left[idx],
                             gammaS_right[idx], e0S_right[idx])
        mass[idx] = ff_lf.mass
        momentum[idx] = ff_lf.momentum
        energy_left[idx] = ff_lf.energy_left_owner
        energy_right[idx] = ff_lf.energy_right_owner

    raise CorrectionFailedError(
        f"[DF] Correction non convergee apres {max_iter} iterations "
        f"({history[-1]['n_troubled']} cellules encore troubled).",
        history, troubled)


# ---------------------------------------------------------------------------
# Wrappers de compatibilite (interface de l'etape 3E, cas particulier d'un
# Euler forward isole : V_eval == primitives(W_before), dt_eff == dt).
# Conserves pour que test_corrector_unit.py (etape 3E) continue de passer
# sans modification.
# ---------------------------------------------------------------------------

def correct_hybrid_step_fc(V0, dx, dt, zeta=0.2, mask_variant='direct',
                            max_iter=10, atol=1e-12, rtol=1e-9, limit_ho=False, verbose=False):
    """Cas particulier de correct_hybrid_rhs_fc : un Euler forward isole
    depuis V0 (s_prev et gammaS_ref evalues sur V0 lui-meme)."""
    W0 = primitives_to_conservatives(V0)
    gammaS_ref, _ = get_gammaS_e0S(V0)
    s_prev = entropy_double_flux(V0, gammaS_ref)
    res = correct_hybrid_rhs_fc(V0, W0, s_prev, gammaS_ref, dx, dt, zeta=zeta,
                                 mask_variant=mask_variant, max_iter=max_iter,
                                 atol=atol, rtol=rtol, limit_ho=limit_ho, verbose=verbose)
    return dict(V1=res['V_after'], W1=res['W_after'], mask=res['mask'],
                corrected_faces=res['corrected_faces'], n_iterations=res['n_iterations'],
                converged=res['converged'], history=res['history'])


def correct_hybrid_step_df(
    V0,
    gammaS_frozen,
    e0S_frozen,
    dx,
    dt,
    zeta=0.2,
    mask_variant='direct',
    max_iter=10,
    atol=1e-12,
    rtol=1e-9,
    limit_ho=False,
    use_thinc_bvd=False,
    thinc_bvd_options=None,
    verbose=False,
):
    """Cas particulier de correct_hybrid_rhs_df : un Euler forward isole
    depuis V0."""
    W0 = primitives_to_conservatives(V0)
    s_prev = entropy_double_flux(V0, gammaS_frozen)
    res = correct_hybrid_rhs_df(
        V0,
        W0,
        s_prev,
        gammaS_frozen,
        e0S_frozen,
        dx,
        dt,
        zeta=zeta,
        mask_variant=mask_variant,
        max_iter=max_iter,
        atol=atol,
        rtol=rtol,
        limit_ho=limit_ho,
        use_thinc_bvd=use_thinc_bvd,
        thinc_bvd_options=thinc_bvd_options,
        verbose=verbose,
    )
    return dict(
        V1=res['V_after'],
        W1=res['W_after'],
        mask=res['mask'],
        corrected_faces=res['corrected_faces'],
        n_iterations=res['n_iterations'],
        converged=res['converged'],
        history=res['history'],
        thinc_face_mask=res['thinc_face_mask'],
        use_thinc_cell=res['use_thinc_cell'],
    )
