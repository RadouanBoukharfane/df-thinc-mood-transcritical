"""SSP-RK3 drivers of the periodic one-dimensional solver."""

import numpy as np

from thermo import get_rho_from_T_p, get_sos, get_gammaS_e0S
from numerics import primitives_to_conservatives, update_primitives_df
from numerics_fc import update_primitives_fc
from entropy import entropy_double_flux
from corrector import correct_hybrid_rhs_fc, correct_hybrid_rhs_df, CorrectionFailedError
from mood import correct_hybrid_rhs_df_mood_global, correct_hybrid_rhs_df_mood_local
from species import (
    species_mass_from_fractions, mass_fractions_from_species_mass,
    advance_species_euler, diagnose_species,
)


def _nitrogen_jump_ic(Nx, T_lo=100.0, T_hi=300.0, p0=5.0e6, u0=100.0, eta=1e-10):
    dx = 1.0 / Nx
    x = np.arange(dx / 2, 1.0, dx)
    T0 = T_hi + (T_lo - T_hi) / 2.0 * (np.tanh((x - 0.25) / eta) + np.tanh(-(x - 0.75) / eta))
    p0_arr = p0 * np.ones(Nx)
    u0_arr = u0 * np.ones(Nx)
    rho0 = get_rho_from_T_p(T0, p0_arr)
    V0 = np.vstack([rho0, u0_arr, p0_arr])
    return x, dx, V0


def _cfl_dt(V, dx, CFL):
    sos = get_sos(V)
    u = V[1, :]
    return CFL * dx / np.max(np.abs(u) + sos)


def _select_df_euler_corrector(
    mood_mode,
    *,
    zeta,
    mask_variant,
    max_iter,
    rtol,
    limit_ho,
    use_thinc_bvd,
    thinc_bvd_options,
    check_entropy,
):
    common = dict(
        zeta=zeta,
        mask_variant=mask_variant,
        rtol=rtol,
        limit_ho=limit_ho,
        use_thinc_bvd=use_thinc_bvd,
        thinc_bvd_options=thinc_bvd_options,
    )

    if mood_mode == 'legacy':
        def advance(V_eval, W_before, s_prev, gammaS_frozen, e0S_frozen, dx, dt):
            return correct_hybrid_rhs_df(
                V_eval, W_before, s_prev, gammaS_frozen, e0S_frozen,
                dx, dt, max_iter=max_iter, **common
            )
        return advance

    if mood_mode == 'global_multilevel':
        def advance(V_eval, W_before, s_prev, gammaS_frozen, e0S_frozen, dx, dt):
            return correct_hybrid_rhs_df_mood_global(
                V_eval, W_before, s_prev, gammaS_frozen, e0S_frozen,
                dx, dt, check_entropy=check_entropy, **common
            )
        return advance

    if mood_mode == 'local_multilevel':
        def advance(V_eval, W_before, s_prev, gammaS_frozen, e0S_frozen, dx, dt):
            return correct_hybrid_rhs_df_mood_local(
                V_eval, W_before, s_prev, gammaS_frozen, e0S_frozen,
                dx, dt, check_entropy=check_entropy, **common
            )
        return advance

    if mood_mode == 'none':
        def advance(V_eval, W_before, s_prev, gammaS_frozen, e0S_frozen, dx, dt):
            return correct_hybrid_rhs_df_mood_local(
                V_eval, W_before, s_prev, gammaS_frozen, e0S_frozen,
                dx, dt, check_entropy=check_entropy, cascade=False, **common
            )
        return advance

    raise ValueError(
        "mood_mode must be 'legacy', 'global_multilevel', "
        "'local_multilevel' or 'none'"
    )


def advance_ssprk3_df_step(
    V,
    W,
    dx,
    dt,
    *,
    zeta=0.2,
    mask_variant='direct',
    max_iter=10,
    rtol=1.0e-9,
    limit_ho=False,
    use_thinc_bvd=False,
    thinc_bvd_options=None,
    mood_mode='legacy',
    check_entropy=True,
    species_mass=None,
    species_atol=1.0e-11,
):
    """One SSP-RK3 step (double flux, parameters frozen over the step)."""
    gammaS_frozen, e0S_frozen = get_gammaS_e0S(V)
    advance_euler = _select_df_euler_corrector(
        mood_mode,
        zeta=zeta,
        mask_variant=mask_variant,
        max_iter=max_iter,
        rtol=rtol,
        limit_ho=limit_ho,
        use_thinc_bvd=use_thinc_bvd,
        thinc_bvd_options=thinc_bvd_options,
        check_entropy=check_entropy,
    )

    species_enabled = species_mass is not None
    if species_enabled:
        Z = np.asarray(species_mass, dtype=float).copy()
        if Z.ndim != 2 or Z.shape[1] != V.shape[1]:
            raise ValueError("species_mass must have shape (Ns, Nx)")
        mass_fractions_from_species_mass(Z, W[0], atol=species_atol)
    else:
        Z = None

    species_stage_fluxes = []

    s0 = entropy_double_flux(V, gammaS_frozen)
    res1 = advance_euler(V, W, s0, gammaS_frozen, e0S_frozen, dx, dt)
    W1 = res1['W_after']
    V1 = res1['V_after']
    if species_enabled:
        Z1, FZ1 = advance_species_euler(
            Z, res1['mass_flux'], W1[0], dx, dt, atol=species_atol
        )
        species_stage_fluxes.append(FZ1)
    else:
        Z1 = None

    s1 = entropy_double_flux(V1, gammaS_frozen)
    res2 = advance_euler(V1, W1, s1, gammaS_frozen, e0S_frozen, dx, dt)
    W2_euler = res2['W_after']
    W2 = 0.75 * W + 0.25 * W2_euler
    V2 = update_primitives_df(W2, gammaS_frozen, e0S_frozen)
    if species_enabled:
        Z2_euler, FZ2 = advance_species_euler(
            Z1, res2['mass_flux'], W2_euler[0], dx, dt, atol=species_atol
        )
        Z2 = 0.75 * Z + 0.25 * Z2_euler
        mass_fractions_from_species_mass(Z2, W2[0], atol=species_atol)
        species_stage_fluxes.append(FZ2)
    else:
        Z2_euler = Z2 = None

    s2 = entropy_double_flux(V2, gammaS_frozen)
    res3 = advance_euler(V2, W2, s2, gammaS_frozen, e0S_frozen, dx, dt)
    W3_euler = res3['W_after']
    W_new = (1.0 / 3.0) * W + (2.0 / 3.0) * W3_euler
    V_new = update_primitives_df(W_new, gammaS_frozen, e0S_frozen)
    if species_enabled:
        Z3_euler, FZ3 = advance_species_euler(
            Z2, res3['mass_flux'], W3_euler[0], dx, dt, atol=species_atol
        )
        Z_new = (1.0 / 3.0) * Z + (2.0 / 3.0) * Z3_euler
        Y_new = mass_fractions_from_species_mass(Z_new, W_new[0], atol=species_atol)
        species_diag = diagnose_species(Z_new, W_new[0])
        species_stage_fluxes.append(FZ3)
    else:
        Z3_euler = Z_new = Y_new = species_diag = None

    stage_results = (res1, res2, res3)
    stage_min_levels = tuple(
        result.get('accepted_level') for result in stage_results
    )
    stage_corrected_faces = tuple(
        int(np.count_nonzero(result['corrected_faces'])) for result in stage_results
    )

    return dict(
        W_after=W_new,
        V_after=V_new,
        gammaS_frozen=gammaS_frozen,
        e0S_frozen=e0S_frozen,
        stage_results=stage_results,
        stage_min_levels=stage_min_levels,
        stage_corrected_faces=stage_corrected_faces,
        species_mass_after=Z_new,
        mass_fractions_after=Y_new,
        species_diagnostics=species_diag,
        species_stage_fluxes=tuple(species_stage_fluxes),
        intermediates=dict(
            W1=W1, V1=V1,
            W2_euler=W2_euler, W2=W2, V2=V2,
            W3_euler=W3_euler,
            Z1=Z1, Z2_euler=Z2_euler, Z2=Z2, Z3_euler=Z3_euler,
        ),
    )

def run_hybrid_df(
    Nx=150,
    CFL=0.8,
    t_final=None,
    zeta=0.2,
    mask_variant='direct',
    max_iter=10,
    rtol=1e-9,
    limit_ho=False,
    use_thinc_bvd=False,
    thinc_bvd_options=None,
    mood_mode='legacy',
    check_entropy=True,
    Y0=None,
    species_atol=1.0e-11,
    check_interval=100,
    verbose=True,
):
    """Nitrogen-interface advection, double-flux formulation."""
    x, dx, V0 = _nitrogen_jump_ic(Nx)
    W = primitives_to_conservatives(V0)
    V = V0.copy()
    if Y0 is None:
        species_mass = None
        Y = None
    else:
        Y_arr = np.asarray(Y0, dtype=float)
        if Y_arr.ndim == 1:
            if Y_arr.size != 2:
                raise ValueError("a 1-D Y0 must contain two interface values")
            Y = np.empty((2, Nx), dtype=float)
            inside = (x >= 0.25) & (x <= 0.75)
            Y[0] = np.where(inside, Y_arr[0], Y_arr[1])
            Y[1] = 1.0 - Y[0]
        else:
            Y = Y_arr
        species_mass = species_mass_from_fractions(W[0], Y)
        Y = mass_fractions_from_species_mass(species_mass, W[0])

    t_final = t_final if t_final is not None else 1.0 / V0[1, 0]
    t_sim = 0.0
    step = 0
    total_corrections = 0
    max_eno_fraction = 0.0
    max_thinc_face_fraction = 0.0
    max_thinc_cell_fraction = 0.0
    min_accepted_mood_level = 3

    if mood_mode not in {'legacy', 'global_multilevel', 'local_multilevel'}:
        raise ValueError(
            "mood_mode must be 'legacy', 'global_multilevel' or "
            "'local_multilevel'"
        )

    stage_min_level_counts = [dict(), dict(), dict()]
    stage_corrected_face_updates = [0, 0, 0]

    while t_sim < t_final:
        dt = _cfl_dt(V, dx, CFL)
        if t_sim + dt > t_final:
            dt = t_final - t_sim

        try:
            rk = advance_ssprk3_df_step(
                V, W, dx, dt,
                zeta=zeta,
                mask_variant=mask_variant,
                max_iter=max_iter,
                rtol=rtol,
                limit_ho=limit_ho,
                use_thinc_bvd=use_thinc_bvd,
                thinc_bvd_options=thinc_bvd_options,
                mood_mode=mood_mode,
                species_mass=species_mass,
                species_atol=species_atol,
            )
            V_new = rk['V_after']
            species_mass_new = rk['species_mass_after']
            Y_new = rk['mass_fractions_after']
            res1, res2, res3 = rk['stage_results']
        except CorrectionFailedError as exc:
            raise CorrectionFailedError(
                f"[DF] Failure at step {step} (t={t_sim:.6g}s) : {exc}",
                exc.history, exc.troubled_final
            )

        n_corr = (res1['corrected_faces'].sum() + res2['corrected_faces'].sum()
                  + res3['corrected_faces'].sum())
        total_corrections += int(n_corr)
        eno_frac = max(res1['mask'].mean(), res2['mask'].mean(), res3['mask'].mean())
        max_eno_fraction = max(max_eno_fraction, eno_frac)
        thinc_face_frac = max(
            res1['thinc_face_mask'].mean(),
            res2['thinc_face_mask'].mean(),
            res3['thinc_face_mask'].mean(),
        )
        thinc_cell_frac = max(
            res1['use_thinc_cell'].mean(),
            res2['use_thinc_cell'].mean(),
            res3['use_thinc_cell'].mean(),
        )
        max_thinc_face_fraction = max(max_thinc_face_fraction, thinc_face_frac)
        max_thinc_cell_fraction = max(max_thinc_cell_fraction, thinc_cell_frac)
        if mood_mode in {'global_multilevel', 'local_multilevel'}:
            min_accepted_mood_level = min(
                min_accepted_mood_level,
                res1['accepted_level'], res2['accepted_level'], res3['accepted_level'],
            )
            for stage_idx, result in enumerate((res1, res2, res3)):
                level = int(result['accepted_level'])
                counts = stage_min_level_counts[stage_idx]
                counts[level] = counts.get(level, 0) + 1
        for stage_idx, result in enumerate((res1, res2, res3)):
            stage_corrected_face_updates[stage_idx] += int(
                np.count_nonzero(result['corrected_faces'])
            )

        V = V_new
        W = primitives_to_conservatives(V)
        if species_mass is not None:
            species_mass = species_mass_new
            Y = Y_new
        t_sim += dt
        step += 1

        if verbose and (step % check_interval == 0 or t_sim >= t_final):
            print(f'  [DF] step {step}, t={t_sim:.6g}s/{t_final:.6g}s, dt={dt:.4g}, '
                  f'rho=[{V[0].min():.3g},{V[0].max():.3g}], p=[{V[2].min():.4g},{V[2].max():.4g}], '
                  f'cumulated corrections={total_corrections}')

    return dict(
        x=x,
        V=V,
        W=W,
        t_final=t_final,
        step=step,
        total_corrections=total_corrections,
        max_eno_fraction=max_eno_fraction,
        max_thinc_face_fraction=max_thinc_face_fraction,
        max_thinc_cell_fraction=max_thinc_cell_fraction,
        mood_mode=mood_mode,
        min_accepted_mood_level=(
            min_accepted_mood_level if mood_mode in {'global_multilevel', 'local_multilevel'} else None
        ),
        stage_min_level_counts=stage_min_level_counts,
        stage_corrected_face_updates=stage_corrected_face_updates,
        species_mass=species_mass,
        Y=Y,
        species_diagnostics=(
            diagnose_species(species_mass, W[0]) if species_mass is not None else None
        ),
    )


def run_hybrid_fc(Nx=150, CFL=0.8, t_final=None, zeta=0.2, mask_variant='direct',
                   max_iter=10, rtol=1e-9, limit_ho=False, check_interval=100, verbose=True):
    """Nitrogen-interface advection, fully conservative formulation."""
    x, dx, V0 = _nitrogen_jump_ic(Nx)
    W = primitives_to_conservatives(V0)
    V = V0.copy()

    t_final = t_final if t_final is not None else 1.0 / V0[1, 0]
    t_sim = 0.0
    step = 0
    total_corrections = 0

    while t_sim < t_final:
        dt = _cfl_dt(V, dx, CFL)
        if t_sim + dt > t_final:
            dt = t_final - t_sim

        gammaS_ref, _ = get_gammaS_e0S(V)

        try:
            s0 = entropy_double_flux(V, gammaS_ref)
            res1 = correct_hybrid_rhs_fc(V, W, s0, gammaS_ref, dx, dt,
                                          zeta=zeta, mask_variant=mask_variant, max_iter=max_iter,
                                          rtol=rtol, limit_ho=limit_ho)
            k1, Vk1 = res1['W_after'], res1['V_after']

            s1 = entropy_double_flux(Vk1, gammaS_ref)
            res2 = correct_hybrid_rhs_fc(Vk1, k1, s1, gammaS_ref, dx, dt,
                                          zeta=zeta, mask_variant=mask_variant, max_iter=max_iter,
                                          rtol=rtol, limit_ho=limit_ho)
            q2 = res2['W_after']
            k2 = 3 / 4 * W + 1 / 4 * q2
            try:
                Vk2 = update_primitives_fc(k2)
            except (RuntimeError, ValueError) as exc_newton:
                raise CorrectionFailedError(
                    f"[FC] Thermodynamic inversion (e,rho)->T failed when forming Vk2 "
                    f"at step {step} (t={t_sim:.6g}s) : {exc_newton}", [], None)

            s2 = entropy_double_flux(Vk2, gammaS_ref)
            res3 = correct_hybrid_rhs_fc(Vk2, k2, s2, gammaS_ref, dx, dt,
                                          zeta=zeta, mask_variant=mask_variant, max_iter=max_iter,
                                          rtol=rtol, limit_ho=limit_ho)
            q3 = res3['W_after']
            W_new = 1 / 3 * W + 2 / 3 * q3
            V_new = update_primitives_fc(W_new)
        except CorrectionFailedError as exc:
            raise CorrectionFailedError(
                f"[FC] Failure at step {step} (t={t_sim:.6g}s) : {exc}", exc.history, exc.troubled_final)

        n_corr = (res1['corrected_faces'].sum() + res2['corrected_faces'].sum()
                  + res3['corrected_faces'].sum())
        total_corrections += int(n_corr)

        W, V = W_new, V_new
        t_sim += dt
        step += 1

        if verbose and (step % check_interval == 0 or t_sim >= t_final):
            print(f'  [FC] step {step}, t={t_sim:.6g}s/{t_final:.6g}s, dt={dt:.4g}, '
                  f'rho=[{V[0].min():.3g},{V[0].max():.3g}], p=[{V[2].min():.4g},{V[2].max():.4g}], '
                  f'cumulated corrections={total_corrections}')

    return dict(x=x, V=V, W=W, t_final=t_final, step=step, total_corrections=total_corrections)


if __name__ == '__main__':
    print('=== hybrid SSP-RK3, DOUBLE-FLUX mode (N=150, CFL=0.8) ===')
    res_df = run_hybrid_df(check_interval=50)
    print(f'Finished: {res_df["step"]} steps, {res_df["total_corrections"]} cumulative corrections, '
          f'fraction ENO max = {res_df["max_eno_fraction"]:.3f}')

    print()
    print('=== hybrid SSP-RK3, FULLY-CONSERVATIVE mode (N=150, CFL=0.8) ===')
    try:
        res_fc = run_hybrid_fc(check_interval=1)
        print(f'Finished (unexpected): {res_fc["step"]} steps')
    except CorrectionFailedError as e:
        print(f'Explicit failure (expected): {e}')
