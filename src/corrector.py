"""Legacy a posteriori corrector (local replacement by the LLF flux)."""

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
    """Raised when no admissible candidate is obtained."""

    def __init__(self, message, history, troubled_final):
        super().__init__(message)
        self.history = history
        self.troubled_final = troubled_final


def _faces_adjacent_to(troubled):
    return troubled | np.roll(troubled, -1)


def correct_hybrid_rhs_fc(V_eval, W_before, s_prev, gammaS_ref, dx, dt_eff,
                           zeta=0.2, mask_variant='direct', max_iter=10,
                           atol=1e-12, rtol=1e-9, limit_ho=False, verbose=False):
    """Corrected forward-Euler stage, fully conservative formulation."""
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
            print(f'  [FC] iteration {it}: {troubled.sum()} troubled cells, '
                  f'{corrected_faces.sum()} corrected faces')

        if not troubled.any():
            return dict(W_after=W_after, V_after=V_after, mask=mask,
                        corrected_faces=corrected_faces, n_iterations=it,
                        converged=True, history=history, limited_mask=limited_mask)

        cells_to_correct = troubled | np.roll(troubled, 1) | np.roll(troubled, -1)
        new_faces = _faces_adjacent_to(cells_to_correct)
        newly_added = new_faces & ~corrected_faces
        if not newly_added.any():
            break
        corrected_faces |= new_faces

        idx = np.where(corrected_faces)[0]
        V_L_lf = V_eval[:, idx]
        V_R_lf = np.roll(V_eval, -1, axis=1)[:, idx]
        ff_lf = flux_llf_fc(V_L_lf, V_R_lf)
        mass[idx] = ff_lf.mass
        momentum[idx] = ff_lf.momentum
        energy[idx] = ff_lf.energy_left_owner

    raise CorrectionFailedError(
        f"[FC] Correction not converged after {max_iter} iterations "
        f"({history[-1]['n_troubled']} troubled cells left).",
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
    """Corrected forward-Euler stage, double-flux formulation."""
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
            print(f'  [DF] iteration {it}: {troubled.sum()} troubled cells, '
                  f'{corrected_faces.sum()} corrected faces')

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
        f"[DF] Correction not converged after {max_iter} iterations "
        f"({history[-1]['n_troubled']} troubled cells left).",
        history, troubled)


def correct_hybrid_step_fc(V0, dx, dt, zeta=0.2, mask_variant='direct',
                            max_iter=10, atol=1e-12, rtol=1e-9, limit_ho=False, verbose=False):
    """Corrected forward-Euler step, fully conservative formulation."""
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
    """Corrected forward-Euler step, double-flux formulation."""
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
