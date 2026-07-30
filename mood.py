"""Cascade MOOD multiniveau globale pour le solveur double-flux 1D.

Cette premiere generalisation valide la logique de cascade avant la future
recomputation locale par niveaux de cellules/faces. Chaque niveau est essaye
sur tout le domaine, du moins dissipatif au plus robuste :

    3 : hybride THINC-BVD / HO / HLLC
    2 : hybride historique HO / ENO2-HLLC
    1 : ENO2-HLLC sur toutes les faces
    0 : etats constants par morceaux + LLF sur toutes les faces

Le niveau 0 constitue le parachute. Les parametres thermodynamiques
double-flux restent geles pendant tout le sous-pas, comme dans le correcteur
historique.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from entropy import entropy_double_flux, R_N2
from eno2 import reconstruct_eno2_left, reconstruct_eno2_right
from face_operators import flux_hllc_df, flux_llf_df
from hybrid_rhs import get_hybrid_face_fluxes_df, _assemble_rhs_df
from numerics import update_primitives_df


class MoodLevel(IntEnum):
    FIRST_ORDER_LLF = 0
    ENO2_HLLC = 1
    HYBRID = 2
    THINC_BVD_HYBRID = 3


LEVEL_NAMES = {
    MoodLevel.FIRST_ORDER_LLF: "FIRST_ORDER_LLF",
    MoodLevel.ENO2_HLLC: "ENO2_HLLC",
    MoodLevel.HYBRID: "HYBRID",
    MoodLevel.THINC_BVD_HYBRID: "THINC_BVD_HYBRID",
}


@dataclass(frozen=True)
class MoodDetection:
    troubled: np.ndarray
    bad_density: np.ndarray
    bad_finite: np.ndarray
    bad_pressure: np.ndarray
    bad_entropy: np.ndarray
    V_after: object
    min_density: float
    min_pressure: float
    min_entropy_margin: float


def _owner_thermo(gammaS_frozen, e0S_frozen):
    return (
        gammaS_frozen,
        e0S_frozen,
        np.roll(gammaS_frozen, -1),
        np.roll(e0S_frozen, -1),
    )


def _fluxes_at_level_df(
    level,
    V_eval,
    gammaS_frozen,
    e0S_frozen,
    dx,
    zeta=0.2,
    mask_variant="direct",
    limit_ho=False,
    thinc_bvd_options=None,
):
    """Construit les flux globaux d'un niveau MOOD.

    Retourne un dictionnaire uniforme pour tous les niveaux afin de faciliter
    les diagnostics et la future extension locale.
    """
    level = MoodLevel(level)
    nx = V_eval.shape[1]
    gammaS_left, e0S_left, gammaS_right, e0S_right = _owner_thermo(
        gammaS_frozen, e0S_frozen
    )

    if level in (MoodLevel.THINC_BVD_HYBRID, MoodLevel.HYBRID):
        use_thinc = level == MoodLevel.THINC_BVD_HYBRID
        data = get_hybrid_face_fluxes_df(
            V_eval,
            gammaS_frozen,
            e0S_frozen,
            dx,
            zeta=zeta,
            mask_variant=mask_variant,
            limit_ho=limit_ho,
            use_thinc_bvd=use_thinc,
            **(thinc_bvd_options or {}),
        )
        (
            mass,
            momentum,
            energy_left,
            energy_right,
            mask,
            _V_L,
            _V_R,
            _gammaS_left,
            _e0S_left,
            _gammaS_right,
            _e0S_right,
            limited_mask,
            thinc_face_mask,
            use_thinc_cell,
        ) = data
        return dict(
            mass=mass,
            momentum=momentum,
            energy_left=energy_left,
            energy_right=energy_right,
            riemann_face_mask=mask,
            thinc_face_mask=thinc_face_mask,
            use_thinc_cell=use_thinc_cell,
            limited_mask=limited_mask,
        )

    if level == MoodLevel.ENO2_HLLC:
        V_L = reconstruct_eno2_left(V_eval)
        V_R = reconstruct_eno2_right(V_eval)
        ff = flux_hllc_df(
            V_L,
            V_R,
            gammaS_left,
            e0S_left,
            gammaS_right,
            e0S_right,
        )
        return dict(
            mass=ff.mass,
            momentum=ff.momentum,
            energy_left=ff.energy_left_owner,
            energy_right=ff.energy_right_owner,
            riemann_face_mask=np.ones(nx, dtype=bool),
            thinc_face_mask=np.zeros(nx, dtype=bool),
            use_thinc_cell=np.zeros(nx, dtype=bool),
            limited_mask=None,
        )

    # Niveau terminal : etats constants par morceaux. A la face i+1/2,
    # l'etat gauche est la moyenne de i et l'etat droit celle de i+1.
    V_L = V_eval
    V_R = np.roll(V_eval, -1, axis=1)
    ff = flux_llf_df(
        V_L,
        V_R,
        gammaS_left,
        e0S_left,
        gammaS_right,
        e0S_right,
    )
    return dict(
        mass=ff.mass,
        momentum=ff.momentum,
        energy_left=ff.energy_left_owner,
        energy_right=ff.energy_right_owner,
        riemann_face_mask=np.ones(nx, dtype=bool),
        thinc_face_mask=np.zeros(nx, dtype=bool),
        use_thinc_cell=np.zeros(nx, dtype=bool),
        limited_mask=None,
    )


def _detect_candidate_df(
    W_after,
    V_eval,
    gammaS_frozen,
    e0S_frozen,
    atol=1.0e-12,
    rtol=1.0e-9,
    check_entropy=True,
):
    """Applique les memes criteres physiques/entropiques que le correcteur.

    Toute exception d'inversion thermodynamique transforme le candidat en
    echec MOOD au lieu de provoquer un plantage silencieux.
    """
    nx = W_after.shape[1]
    bad_density = (~np.isfinite(W_after[0])) | (W_after[0] <= 0.0)
    bad_finite = ~np.isfinite(W_after).all(axis=0)
    bad_pressure = np.zeros(nx, dtype=bool)
    bad_entropy = np.zeros(nx, dtype=bool)
    V_after = None
    min_entropy_margin = float("nan")

    if not (bad_density.any() or bad_finite.any()):
        try:
            V_after = update_primitives_df(W_after, gammaS_frozen, e0S_frozen)
            bad_finite = ~np.isfinite(V_after).all(axis=0)
            bad_pressure = (~np.isfinite(V_after[2])) | (V_after[2] <= 0.0)

            if check_entropy:
                with np.errstate(invalid="ignore", divide="ignore"):
                    s_candidate = entropy_double_flux(V_after, gammaS_frozen)
                    rho0 = V_eval[0]
                    p0 = V_eval[2]
                    s_center = np.log(p0) - gammaS_frozen * np.log(rho0 * R_N2)
                    s_left = np.log(np.roll(p0, 1)) - gammaS_frozen * np.log(
                        np.roll(rho0, 1) * R_N2
                    )
                    s_right = np.log(np.roll(p0, -1)) - gammaS_frozen * np.log(
                        np.roll(rho0, -1) * R_N2
                    )
                    bound = np.minimum(np.minimum(s_left, s_center), s_right)
                    tol = atol + rtol * np.maximum(1.0, np.abs(bound))
                    margin = s_candidate - bound

                bad_entropy = (~np.isfinite(s_candidate)) | (s_candidate < bound - tol)
                if np.isfinite(margin).any():
                    min_entropy_margin = float(np.nanmin(margin))
        except (RuntimeError, ValueError, FloatingPointError):
            bad_finite = np.ones(nx, dtype=bool)
            V_after = None

    troubled = bad_density | bad_finite | bad_pressure | bad_entropy
    min_density = (
        float(np.nanmin(W_after[0])) if np.isfinite(W_after[0]).any() else float("nan")
    )
    min_pressure = (
        float(np.nanmin(V_after[2]))
        if V_after is not None and np.isfinite(V_after[2]).any()
        else float("nan")
    )

    return MoodDetection(
        troubled=troubled,
        bad_density=bad_density,
        bad_finite=bad_finite,
        bad_pressure=bad_pressure,
        bad_entropy=bad_entropy,
        V_after=V_after,
        min_density=min_density,
        min_pressure=min_pressure,
        min_entropy_margin=min_entropy_margin,
    )


def correct_hybrid_rhs_df_mood_global(
    V_eval,
    W_before,
    s_prev,
    gammaS_frozen,
    e0S_frozen,
    dx,
    dt_eff,
    zeta=0.2,
    mask_variant="direct",
    atol=1.0e-12,
    rtol=1.0e-9,
    limit_ho=False,
    use_thinc_bvd=True,
    thinc_bvd_options=None,
    levels=None,
    check_entropy=True,
    verbose=False,
):
    """Sous-pas Euler avec cascade MOOD globale multiniveau.

    ``s_prev`` est conserve dans la signature pour compatibilite conceptuelle
    avec le correcteur historique. Le critere local est reevalue avec la meme
    borne entropique gelee que celui-ci.
    """
    del s_prev  # la borne locale utilise explicitement V_eval, comme legacy DF

    if levels is None:
        levels = [
            MoodLevel.THINC_BVD_HYBRID,
            MoodLevel.HYBRID,
            MoodLevel.ENO2_HLLC,
            MoodLevel.FIRST_ORDER_LLF,
        ]
    levels = [MoodLevel(level) for level in levels]
    if not use_thinc_bvd:
        levels = [level for level in levels if level != MoodLevel.THINC_BVD_HYBRID]
    if not levels:
        raise ValueError("La cascade MOOD doit contenir au moins un niveau")

    history = []
    top_level = levels[0]
    last_detection = None

    for attempt, level in enumerate(levels):
        try:
            fluxes = _fluxes_at_level_df(
                level,
                V_eval,
                gammaS_frozen,
                e0S_frozen,
                dx,
                zeta=zeta,
                mask_variant=mask_variant,
                limit_ho=limit_ho,
                thinc_bvd_options=thinc_bvd_options,
            )
            rhs = _assemble_rhs_df(
                fluxes["mass"],
                fluxes["momentum"],
                fluxes["energy_left"],
                fluxes["energy_right"],
                dx,
            )
            W_after = W_before + dt_eff * rhs
            detection = _detect_candidate_df(
                W_after,
                V_eval,
                gammaS_frozen,
                e0S_frozen,
                atol=atol,
                rtol=rtol,
                check_entropy=check_entropy,
            )
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            # Une reconstruction/thermo invalide au niveau courant est un rejet
            # normal de cascade. Le niveau suivant est essaye.
            nx = V_eval.shape[1]
            detection = MoodDetection(
                troubled=np.ones(nx, dtype=bool),
                bad_density=np.zeros(nx, dtype=bool),
                bad_finite=np.ones(nx, dtype=bool),
                bad_pressure=np.zeros(nx, dtype=bool),
                bad_entropy=np.zeros(nx, dtype=bool),
                V_after=None,
                min_density=float("nan"),
                min_pressure=float("nan"),
                min_entropy_margin=float("nan"),
            )
            fluxes = None
            W_after = None
            failure_message = str(exc)
        else:
            failure_message = None

        last_detection = detection
        history.append(
            dict(
                attempt=attempt,
                level=int(level),
                level_name=LEVEL_NAMES[level],
                n_troubled=int(detection.troubled.sum()),
                n_bad_density=int(detection.bad_density.sum()),
                n_bad_pressure=int(detection.bad_pressure.sum()),
                n_bad_finite=int(detection.bad_finite.sum()),
                n_bad_entropy=int(detection.bad_entropy.sum()),
                troubled_indices=np.flatnonzero(detection.troubled).tolist(),
                min_density=detection.min_density,
                min_pressure=detection.min_pressure,
                min_entropy_margin=detection.min_entropy_margin,
                failure_message=failure_message,
            )
        )

        if verbose:
            print(
                f"  [MOOD-DF] niveau {int(level)} {LEVEL_NAMES[level]} : "
                f"{detection.troubled.sum()} cellule(s) troubled"
            )

        if not detection.troubled.any():
            nx = V_eval.shape[1]
            corrected_faces = np.full(nx, level != top_level, dtype=bool)
            return dict(
                W_after=W_after,
                V_after=detection.V_after,
                accepted_level=int(level),
                accepted_level_name=LEVEL_NAMES[level],
                attempted_levels=[int(item["level"]) for item in history],
                n_iterations=attempt,
                converged=True,
                history=history,
                corrected_faces=corrected_faces,
                mass_flux=fluxes["mass"].copy(),
                mask=fluxes["riemann_face_mask"],
                limited_mask=fluxes["limited_mask"],
                thinc_face_mask=fluxes["thinc_face_mask"],
                use_thinc_cell=fluxes["use_thinc_cell"],
            )

    # Import local pour eviter une dependance circulaire au chargement.
    from corrector import CorrectionFailedError

    raise CorrectionFailedError(
        "[MOOD-DF] Aucun niveau de la cascade globale n'a produit un "
        "candidat admissible.",
        history,
        last_detection.troubled,
    )


def face_levels_from_cells(cell_levels):
    """Retourne le niveau commun de chaque face periodique.

    La face ``i`` represente ``i+1/2`` entre les cellules ``i`` et ``i+1``.
    Le niveau le plus robuste est impose aux deux traces de la face.
    """
    cell_levels = np.asarray(cell_levels, dtype=np.int8)
    if cell_levels.ndim != 1:
        raise ValueError("cell_levels doit etre un tableau 1D")
    return np.minimum(cell_levels, np.roll(cell_levels, -1))


def _mix_fluxes_by_face_level(fluxes_by_level, face_levels):
    """Assemble un jeu de flux unique a partir des niveaux de face.

    Les flux de tous les niveaux sont calcules sur le meme etat ``V_eval``.
    Cette fonction ne fait ensuite qu'une selection locale, face par face.
    """
    face_levels = np.asarray(face_levels, dtype=np.int8)
    nx = face_levels.size

    mass = np.empty(nx)
    momentum = np.empty(nx)
    energy_left = np.empty(nx)
    energy_right = np.empty(nx)
    riemann_mask = np.empty(nx, dtype=bool)
    thinc_face_mask = np.zeros(nx, dtype=bool)
    use_thinc_cell = np.zeros(nx, dtype=bool)

    limited_mask = None
    limited_accumulator = np.zeros(nx, dtype=bool)
    has_limited_mask = False

    available = {int(level) for level in fluxes_by_level}
    requested = set(np.unique(face_levels).tolist())
    missing = requested - available
    if missing:
        raise ValueError(f"Niveaux de face indisponibles : {sorted(missing)}")

    for level, fluxes in fluxes_by_level.items():
        idx = face_levels == int(level)
        if not idx.any():
            continue
        mass[idx] = fluxes["mass"][idx]
        momentum[idx] = fluxes["momentum"][idx]
        energy_left[idx] = fluxes["energy_left"][idx]
        energy_right[idx] = fluxes["energy_right"][idx]
        riemann_mask[idx] = fluxes["riemann_face_mask"][idx]
        thinc_face_mask[idx] = fluxes["thinc_face_mask"][idx]

        lm = fluxes.get("limited_mask")
        if lm is not None:
            limited_accumulator[idx] = np.asarray(lm, dtype=bool)[idx]
            has_limited_mask = True

    # Le masque THINC cellulaire est diagnostique. Une cellule est marquee
    # uniquement si le niveau 3 reste actif sur au moins une de ses faces.
    top_fluxes = fluxes_by_level.get(MoodLevel.THINC_BVD_HYBRID)
    if top_fluxes is not None:
        raw = np.asarray(top_fluxes["use_thinc_cell"], dtype=bool)
        adjacent_top_face = (
            (face_levels == int(MoodLevel.THINC_BVD_HYBRID))
            | np.roll(face_levels == int(MoodLevel.THINC_BVD_HYBRID), 1)
        )
        use_thinc_cell = raw & adjacent_top_face

    if has_limited_mask:
        limited_mask = limited_accumulator

    return dict(
        mass=mass,
        momentum=momentum,
        energy_left=energy_left,
        energy_right=energy_right,
        riemann_face_mask=riemann_mask,
        thinc_face_mask=thinc_face_mask,
        use_thinc_cell=use_thinc_cell,
        limited_mask=limited_mask,
    )


def correct_hybrid_rhs_df_mood_local(
    V_eval,
    W_before,
    s_prev,
    gammaS_frozen,
    e0S_frozen,
    dx,
    dt_eff,
    zeta=0.2,
    mask_variant="direct",
    atol=1.0e-12,
    rtol=1.0e-9,
    limit_ho=False,
    use_thinc_bvd=True,
    thinc_bvd_options=None,
    initial_cell_levels=None,
    max_iterations=None,
    check_entropy=True,
    verbose=False,
):
    """Sous-pas Euler avec cascade MOOD locale par niveaux de cellules.

    Algorithme
    ----------
    1. Chaque cellule commence au niveau maximal disponible.
    2. Le niveau de face est ``min(level_left, level_right)``.
    3. Les flux uniques de masse/momentum sont selectionnes face par face.
       Les deux flux energetiques owner-dependent suivent le meme niveau.
    4. Le candidat est teste. Seules les cellules rejetees descendent d'un
       niveau, puis les faces incidentes changent automatiquement de niveau.
    5. La procedure s'arrete lorsqu'aucune cellule n'est troubled.

    Pour cette premiere implementation Python, les flux de chaque niveau sont
    pre-calcules une fois par sous-pas sur tout le domaine. La recomputation
    est donc locale du point de vue mathematique, mais pas encore optimisee en
    cout CPU. Cela evite un code de cache complexe avant validation.
    """
    del s_prev

    nx = V_eval.shape[1]
    available_levels = [
        MoodLevel.THINC_BVD_HYBRID,
        MoodLevel.HYBRID,
        MoodLevel.ENO2_HLLC,
        MoodLevel.FIRST_ORDER_LLF,
    ]
    if not use_thinc_bvd:
        available_levels.remove(MoodLevel.THINC_BVD_HYBRID)
    top_level = max(available_levels)

    fluxes_by_level = {}
    flux_failures = {}
    for level in available_levels:
        try:
            fluxes_by_level[level] = _fluxes_at_level_df(
                level,
                V_eval,
                gammaS_frozen,
                e0S_frozen,
                dx,
                zeta=zeta,
                mask_variant=mask_variant,
                limit_ho=limit_ho,
                thinc_bvd_options=thinc_bvd_options,
            )
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            flux_failures[level] = str(exc)

    # Le parachute doit toujours etre disponible. Sinon le sous-pas ne peut
    # pas etre securise et l'on echoue explicitement.
    if MoodLevel.FIRST_ORDER_LLF not in fluxes_by_level:
        from corrector import CorrectionFailedError
        raise CorrectionFailedError(
            "[MOOD-DF local] Le niveau terminal FIRST_ORDER_LLF n'a pas pu "
            "etre construit.",
            [dict(level=0, failure_message=flux_failures.get(MoodLevel.FIRST_ORDER_LLF))],
            np.ones(nx, dtype=bool),
        )

    if initial_cell_levels is None:
        cell_levels = np.full(nx, int(top_level), dtype=np.int8)
    else:
        cell_levels = np.asarray(initial_cell_levels, dtype=np.int8).copy()
        if cell_levels.shape != (nx,):
            raise ValueError("initial_cell_levels doit etre de forme (Nx,)")
        if np.any(cell_levels < 0) or np.any(cell_levels > int(top_level)):
            raise ValueError("initial_cell_levels contient un niveau invalide")

    # Si un niveau n'a pas pu etre construit, les cellules qui le demandent
    # sont rabattues vers le prochain niveau disponible plus robuste.
    available_int = sorted((int(level) for level in fluxes_by_level), reverse=True)
    for i in range(nx):
        requested = int(cell_levels[i])
        choices = [level for level in available_int if level <= requested]
        if not choices:
            choices = [int(MoodLevel.FIRST_ORDER_LLF)]
        cell_levels[i] = max(choices)

    if max_iterations is None:
        # Les cellules ne deviennent pas necessairement troubled toutes au meme
        # iteration : la baisse d'un niveau modifie les faces voisines et peut
        # propager la detection. Une borne L+1 est donc insuffisante. Comme
        # chaque reduction diminue strictement la somme des niveaux, le nombre
        # total de reductions est borne par Nx*top_level.
        max_iterations = nx * int(top_level) + 1
    if max_iterations < 1:
        raise ValueError("max_iterations doit etre >= 1")

    history = []
    last_detection = None
    last_fluxes = None
    last_W_after = None

    for iteration in range(max_iterations):
        face_levels = face_levels_from_cells(cell_levels)
        mixed = _mix_fluxes_by_face_level(fluxes_by_level, face_levels)
        rhs = _assemble_rhs_df(
            mixed["mass"],
            mixed["momentum"],
            mixed["energy_left"],
            mixed["energy_right"],
            dx,
        )
        W_after = W_before + dt_eff * rhs
        detection = _detect_candidate_df(
            W_after,
            V_eval,
            gammaS_frozen,
            e0S_frozen,
            atol=atol,
            rtol=rtol,
            check_entropy=check_entropy,
        )

        last_detection = detection
        last_fluxes = mixed
        last_W_after = W_after
        history.append(
            dict(
                iteration=iteration,
                n_troubled=int(detection.troubled.sum()),
                troubled_indices=np.flatnonzero(detection.troubled).tolist(),
                n_bad_density=int(detection.bad_density.sum()),
                n_bad_pressure=int(detection.bad_pressure.sum()),
                n_bad_finite=int(detection.bad_finite.sum()),
                n_bad_entropy=int(detection.bad_entropy.sum()),
                min_density=detection.min_density,
                min_pressure=detection.min_pressure,
                min_entropy_margin=detection.min_entropy_margin,
                min_cell_level=int(cell_levels.min()),
                max_cell_level=int(cell_levels.max()),
                level_counts={
                    int(level): int(np.count_nonzero(cell_levels == int(level)))
                    for level in available_levels
                },
                n_corrected_faces=int(np.count_nonzero(face_levels < int(top_level))),
            )
        )

        if verbose:
            print(
                f"  [MOOD-DF local] iteration {iteration}: "
                f"{detection.troubled.sum()} troubled, "
                f"niveaux cellules [{cell_levels.min()},{cell_levels.max()}]"
            )

        if not detection.troubled.any():
            corrected_faces = face_levels < int(top_level)
            return dict(
                W_after=W_after,
                V_after=detection.V_after,
                accepted_level=int(cell_levels.min()),
                accepted_level_name=LEVEL_NAMES[MoodLevel(int(cell_levels.min()))],
                cell_levels=cell_levels.copy(),
                face_levels=face_levels.copy(),
                n_iterations=iteration,
                converged=True,
                history=history,
                corrected_faces=corrected_faces,
                mass_flux=mixed["mass"].copy(),
                mask=mixed["riemann_face_mask"],
                limited_mask=mixed["limited_mask"],
                thinc_face_mask=mixed["thinc_face_mask"],
                use_thinc_cell=mixed["use_thinc_cell"],
                flux_failures={int(k): v for k, v in flux_failures.items()},
            )

        # Une cellule troubled descend exactement d'un niveau. Si elle est
        # deja au parachute, le pas doit etre rejete/repris avec un dt plus
        # petit : continuer ne changerait plus aucun flux local.
        terminal_troubled = detection.troubled & (
            cell_levels == int(MoodLevel.FIRST_ORDER_LLF)
        )
        if terminal_troubled.any():
            from corrector import CorrectionFailedError
            raise CorrectionFailedError(
                "[MOOD-DF local] Des cellules restent inadmissibles au niveau "
                "terminal FIRST_ORDER_LLF.",
                history,
                terminal_troubled,
            )

        cell_levels[detection.troubled] -= 1

        # Evite de demander un niveau dont le calcul global a echoue.
        for i in np.flatnonzero(detection.troubled):
            requested = int(cell_levels[i])
            choices = [level for level in available_int if level <= requested]
            cell_levels[i] = max(choices) if choices else int(MoodLevel.FIRST_ORDER_LLF)

    from corrector import CorrectionFailedError
    raise CorrectionFailedError(
        "[MOOD-DF local] Nombre maximal d'iterations atteint sans candidat "
        "admissible.",
        history,
        last_detection.troubled if last_detection is not None else np.ones(nx, dtype=bool),
    )
