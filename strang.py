"""
strang.py
Etape 2D-1 (suite) : orchestrateur de splitting directionnel de Strang,
au-dessus de sweep_x_step/sweep_y_step (twod.py).

Formule (symetrique, ordre 2 en temps global -- meme si SSP-RK3, ordre 3,
est utilise a l'interieur de chaque sweep, cf. discussion de relecture) :

    W^{n+1} = S_x(dt/2) . S_y(dt) . S_x(dt/2) . W^n        (motif "xyx")

avec alternance a chaque pas complet :

    pas pair   : x(dt/2) - y(dt) - x(dt/2)
    pas impair : y(dt/2) - x(dt) - y(dt/2)

pour reduire le biais directionnel cumule (cf. relecture).

Gel double-flux : chaque sweep gele gammaS/e0S a son propre debut (deja le
comportement natif de sweep_x_step/sweep_y_step, rien a faire de plus
ici) -- PAS un gel unique pour tout le pas de Strang.
"""

import numpy as np

from twod import sweep_x_step, sweep_y_step


def strang_step(V2D, dx, dy, dt, wall_bc, step_parity, opts_x=None, opts_y=None):
    """Un pas de temps complet de Strang, alternant xyx/yxy selon la
    parite du pas (step_parity=0 -> xyx, step_parity=1 -> yxy).

    opts_x, opts_y : dicts d'options transmises respectivement a
        sweep_x_step et sweep_y_step (mood_mode, use_thinc_bvd, zeta,
        limit_ho, etc. -- peuvent differer entre les deux directions si
        besoin, mais coherence recommandee).

    Retourne (V2D_new, diagnostics) ou diagnostics est un dict avec les
    diagnostics bruts des 3 sous-sweeps, pour inspection/debug.
    """
    opts_x = {} if opts_x is None else opts_x
    opts_y = {} if opts_y is None else opts_y

    if step_parity % 2 == 0:
        # motif x(dt/2) - y(dt) - x(dt/2)
        V2D, diag_a = sweep_x_step(V2D, dx, dt / 2.0, **opts_x)
        V2D, diag_b = sweep_y_step(V2D, dy, dt, wall_bc, **opts_y)
        V2D, diag_c = sweep_x_step(V2D, dx, dt / 2.0, **opts_x)
        pattern = 'xyx'
    else:
        # motif y(dt/2) - x(dt) - y(dt/2)
        V2D, diag_a = sweep_y_step(V2D, dy, dt / 2.0, wall_bc, **opts_y)
        V2D, diag_b = sweep_x_step(V2D, dx, dt, **opts_x)
        V2D, diag_c = sweep_y_step(V2D, dy, dt / 2.0, wall_bc, **opts_y)
        pattern = 'yxy'

    diagnostics = dict(pattern=pattern, first=diag_a, middle=diag_b, last=diag_c)
    return V2D, diagnostics


def run_strang_2d(V2D0, dx, dy, t_final, cfl_dt_fn, wall_bc,
                   opts_x=None, opts_y=None, check_interval=50, verbose=True,
                   max_steps=1_000_000):
    """Boucle temporelle complete, alternant xyx/yxy a chaque pas.

    cfl_dt_fn(V2D) -> dt : fonction fournie par l'appelant pour calculer
        le pas de temps CFL a partir de l'etat courant (etape 2D-2/2D-3 :
        a definir proprement avec les vitesses caracteristiques 2D,
        cf. max(|u|+c)/dx + max(|v|+c)/dy comme en 2D "Option A" plus tot
        dans le projet -- non fourni ici, infrastructure seulement).
    """
    V2D = V2D0.copy()
    t_sim = 0.0
    step = 0

    while t_sim < t_final and step < max_steps:
        dt = cfl_dt_fn(V2D)
        if t_sim + dt > t_final:
            dt = t_final - t_sim

        V2D, diag = strang_step(V2D, dx, dy, dt, wall_bc, step_parity=step % 2,
                                 opts_x=opts_x, opts_y=opts_y)

        t_sim += dt
        step += 1

        if verbose and (step % check_interval == 0 or t_sim >= t_final):
            print(f'  [2D] pas {step}, t={t_sim:.6g}/{t_final:.6g}, dt={dt:.4g}, '
                  f'motif={diag["pattern"]}, rho=[{V2D[0].min():.4g},{V2D[0].max():.4g}], '
                  f'p=[{V2D[3].min():.4g},{V2D[3].max():.4g}]')

    return dict(V2D=V2D, t_final=t_sim, step=step)


def strang_step_periodic_2d(V2D, dx, dy, dt, step_parity,
                             opts_x=None, opts_y=None):
    """Pas de Strang periodique dans les deux directions.

    Cette variante est un banc de verification du splitting et de
    l'isotropie. Le solveur physique principal peut conserver des murs en y.
    """
    from twod import sweep_y_periodic_step

    opts_x = {} if opts_x is None else opts_x
    opts_y = {} if opts_y is None else opts_y
    if step_parity % 2 == 0:
        V2D, da = sweep_x_step(V2D, dx, dt / 2.0, **opts_x)
        V2D, db = sweep_y_periodic_step(V2D, dy, dt, **opts_y)
        V2D, dc = sweep_x_step(V2D, dx, dt / 2.0, **opts_x)
        pattern = "xyx"
    else:
        V2D, da = sweep_y_periodic_step(V2D, dy, dt / 2.0, **opts_y)
        V2D, db = sweep_x_step(V2D, dx, dt, **opts_x)
        V2D, dc = sweep_y_periodic_step(V2D, dy, dt / 2.0, **opts_y)
        pattern = "yxy"
    return V2D, dict(pattern=pattern, first=da, middle=db, last=dc)


def run_strang_periodic_2d(V2D0, dx, dy, t_final, cfl_dt_fn,
                            opts_x=None, opts_y=None, max_steps=1_000_000):
    """Boucle temporelle periodique-periodique pour tests de verification."""
    V2D = np.asarray(V2D0, dtype=float).copy()
    t = 0.0
    step = 0
    while t < t_final and step < max_steps:
        dt = min(float(cfl_dt_fn(V2D)), t_final - t)
        V2D, _ = strang_step_periodic_2d(
            V2D, dx, dy, dt, step, opts_x=opts_x, opts_y=opts_y
        )
        t += dt
        step += 1
    return dict(V2D=V2D, t_final=t, step=step)
