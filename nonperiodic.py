"""Branche 1D non periodique du solveur double-flux azote.

Cette etape valide l'infrastructure de bord sur le cas deja maitrise :
advection d'une interface d'azote a pression et vitesse uniformes. Les flux
sont stockes sur les Nx+1 faces physiques, contrairement a la branche
periodique historique qui stocke Nx faces.
"""
from dataclasses import dataclass
import numpy as np

from boundary import fill_primitive_ghosts, transmissive, dirichlet, adiabatic_wall
from face_operators import flux_ho, flux_hllc_df, flux_llf_df
from numerics import primitives_to_conservatives, update_primitives_df
from thermo import get_gammaS_e0S, get_rho_from_T_p, get_sos, get_T_from_p_rho
from entropy import entropy_double_flux, R_N2


@dataclass(frozen=True)
class NonPeriodicDiagnostics:
    hllc_face_mask: np.ndarray
    thinc_cell_mask: np.ndarray
    thinc_face_mask: np.ndarray
    shock_face_mask: np.ndarray | None = None
    shock_cell_mask: np.ndarray | None = None


def nitrogen_interface_ic(Nx, x_left=0.0, x_right=1.0, T_lo=100.0,
                          T_hi=300.0, p0=5.0e6, u0=100.0,
                          interface_left=0.25, interface_right=0.75):
    dx = (x_right - x_left) / Nx
    x = x_left + (np.arange(Nx) + 0.5) * dx
    inside = (x >= interface_left) & (x <= interface_right)
    T = np.where(inside, T_lo, T_hi)
    p = np.full(Nx, p0)
    u = np.full(Nx, u0)
    rho = get_rho_from_T_p(T, p)
    return x, dx, np.vstack((rho, u, p))


def _face_stencils(Vg, nghost=2):
    """Extrait les stencils des Nx+1 faces physiques.

    Pour la face f, la cellule gauche a l'indice g+f-1 et la droite g+f.
    """
    nx = Vg.shape[1] - 2 * nghost
    f = np.arange(nx + 1)
    iL = nghost + f - 1
    iR = nghost + f
    return (
        Vg[:, iL - 1], Vg[:, iL], Vg[:, iR], Vg[:, iR + 1],
        iL, iR,
    )


def reconstruct_nonperiodic(Vg, nghost=2, limit_ho=False):
    """Reconstructions HO et ENO2 sur les Nx+1 faces physiques."""
    qim1, qi, qip1, qip2, _, _ = _face_stencils(Vg, nghost)

    q_ho = (-qim1 + 7.0 * qi + 7.0 * qip1 - qip2) / 12.0
    if limit_ho:
        lo = np.minimum(np.minimum(qim1, qi), np.minimum(qip1, qip2))
        hi = np.maximum(np.maximum(qim1, qi), np.maximum(qip1, qip2))
        q_ho = np.clip(q_ho, lo, hi)

    # ENO gauche, centre sur qi.
    dl = np.abs(qi - qim1)
    dr = np.abs(qip1 - qi)
    qL = np.where(dl <= dr, 1.5 * qi - 0.5 * qim1, 0.5 * (qi + qip1))

    # ENO droit, centre sur qip1.
    dlr = np.abs(qip1 - qi)
    drr = np.abs(qip2 - qip1)
    qR = np.where(drr <= dlr, 1.5 * qip1 - 0.5 * qip2, 0.5 * (qi + qip1))
    return q_ho, qL, qR


def _thinc_cell_traces(qg, beta=1.6, eps=1.0e-14, nghost=2):
    """THINC pour les Nx cellules physiques, sans enroulement periodique.

    Formulation strictement identique a ``thinc.reconstruct_thinc`` ; seules
    les valeurs voisines proviennent ici des cellules fantomes physiques.
    """
    qg = np.asarray(qg, dtype=float)
    nx = qg.size - 2 * nghost
    q_im1 = qg[nghost - 1:nghost - 1 + nx]
    q = qg[nghost:nghost + nx]
    q_ip1 = qg[nghost + 1:nghost + 1 + nx]

    q_min = np.minimum(q_im1, q_ip1)
    q_max = np.maximum(q_im1, q_ip1)
    dq = q_max - q_min
    monotone = (q - q_im1) * (q_ip1 - q) > 0.0
    local_scale = np.maximum.reduce(
        [np.ones_like(q), np.abs(q_im1), np.abs(q), np.abs(q_ip1)]
    )
    nondegenerate = dq > eps * local_scale
    active = monotone & nondegenerate

    q_left = q.copy()
    q_right = q.copy()
    if not np.any(active):
        return q_left, q_right, active

    theta = np.sign(q_ip1 - q_im1)
    C = np.zeros_like(q)
    np.divide(q - q_min, dq, out=C, where=nondegenerate)
    tiny = np.finfo(float).eps
    C = np.clip(C, tiny, 1.0 - tiny)
    h_bar = theta * (2.0 * C - 1.0)
    h_bar = np.clip(h_bar, -1.0 + tiny, 1.0 - tiny)

    exp_beta = np.exp(beta)
    exp_minus_beta = np.exp(-beta)
    r = np.exp(beta * h_bar)
    positive_floor = np.finfo(float).tiny
    y = np.maximum(exp_beta - r, positive_floor) / np.maximum(
        r - exp_minus_beta, positive_floor
    )
    xi0 = 0.5 * np.log(y) / beta

    h_left = np.tanh(-beta * xi0)
    h_right = np.tanh(beta * (1.0 - xi0))
    q_left_t = q_min + 0.5 * dq * (1.0 + theta * h_left)
    q_right_t = q_min + 0.5 * dq * (1.0 + theta * h_right)
    q_left = np.where(active, np.clip(q_left_t, q_min, q_max), q_left)
    q_right = np.where(active, np.clip(q_right_t, q_min, q_max), q_right)
    return q_left, q_right, active


def _contact_thinc_selection(
    Vg,
    V_L_eno,
    V_R_eno,
    beta=1.6,
    contact_threshold=5.0e-2,
    shock_pressure_threshold=5.0e-2,
    shock_compression_threshold=1.0e-3,
    shock_velocity_threshold=5.0e-2,
    eps=1.0e-14,
    velocity_floor=1.0,
    contact_pressure_weight=20.0,
    contact_velocity_weight=20.0,
    nghost=2,
):
    """Selection THINC-BVD non periodique equivalente a la branche periodique.

    Le BVD reste cellulaire, mais la classification contact/choc est d'abord
    construite sur les ``Nx+1`` faces physiques puis propagee aux cellules
    adjacentes, exactement comme dans ``sensor.classify_contact_shock``.
    """
    nx = Vg.shape[1] - 2 * nghost
    rho_g = Vg[0]
    rho_t_l, rho_t_r, active = _thinc_cell_traces(
        rho_g, beta=beta, nghost=nghost
    )

    # Traces ENO par cellule : gauche i-1/2 et droite i+1/2.
    rho_e_l = V_R_eno[0, :nx]
    rho_e_r = V_L_eno[0, 1:nx + 1]

    # Sauts internes i+1/2, i=0..Nx-2. Les faces de domaine ne doivent pas
    # introduire une selection BVD artificielle dans les cellules physiques.
    j_eno = np.abs(rho_e_l[1:] - rho_e_r[:-1])
    j_th = np.abs(rho_t_l[1:] - rho_t_r[:-1])
    tbv_e = np.zeros(nx)
    tbv_t = np.zeros(nx)
    tbv_e[:-1] += j_eno
    tbv_e[1:] += j_eno
    tbv_t[:-1] += j_th
    tbv_t[1:] += j_th
    scale_tbv = np.maximum(tbv_e, tbv_t)
    prefer = tbv_t + (1.0e-14 + 1.0e-12 * scale_tbv) < tbv_e

    # Classification sur les Nx+1 faces physiques, meme definition que la
    # branche periodique (rho/u/p gauche et droite de chaque face).
    f = np.arange(nx + 1)
    iL = nghost + f - 1
    iR = nghost + f
    rhoL, rhoR = Vg[0, iL], Vg[0, iR]
    uL, uR = Vg[1, iL], Vg[1, iR]
    pL, pR = Vg[2, iL], Vg[2, iR]

    density_jump = np.abs(rhoR - rhoL) / (
        np.abs(rhoR) + np.abs(rhoL) + eps
    )
    pressure_jump = np.abs(pR - pL) / (
        np.abs(pR) + np.abs(pL) + eps
    )
    velocity_scale = np.maximum(np.abs(uR) + np.abs(uL), velocity_floor)
    velocity_jump = np.abs(uR - uL) / (velocity_scale + eps)
    compression = np.maximum(0.0, -(uR - uL)) / (velocity_scale + eps)
    contact_score = density_jump * np.exp(
        -contact_pressure_weight * pressure_jump
        -contact_velocity_weight * velocity_jump
    )

    shock_face = (
        pressure_jump > shock_pressure_threshold
    ) & (
        (compression > shock_compression_threshold)
        | (velocity_jump > shock_velocity_threshold)
    )
    contact_face = (~shock_face) & (contact_score > contact_threshold)

    shock_cell = shock_face[:-1] | shock_face[1:]
    contact_cell = (~shock_cell) & (contact_face[:-1] | contact_face[1:])

    # Comme dans la branche periodique, une cellule THINC inactive fournit
    # un candidat constant qui peut etre retenu par BVD. Ne pas filtrer par
    # ``active`` ici, sinon les deux branches ne sont plus equivalentes.
    use = prefer & contact_cell
    return use, rho_t_l, rho_t_r


def _dilate_face_mask_nonperiodic(mask, radius):
    """Dilate un masque de faces sans raccord periodique."""
    out = np.asarray(mask, dtype=bool).copy()
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius doit etre >= 0")
    base = out.copy()
    for shift in range(1, radius + 1):
        out[shift:] |= base[:-shift]
        out[:-shift] |= base[shift:]
    return out


def _shock_face_mask_nonperiodic(Vg, nx, nghost=2,
                                  pressure_threshold=2.0e-2,
                                  compression_threshold=0.0,
                                  velocity_floor=1.0, buffer_faces=2):
    """Detecteur choc sur les Nx+1 faces physiques.

    Un saut de pression relatif est le signal principal. La compression peut
    etre imposee avec ``compression_threshold > 0``. Le masque est dilate pour
    proteger le stencil autour du choc et interdire THINC dans son voisinage.
    """
    f = np.arange(nx + 1)
    iL = nghost + f - 1
    iR = nghost + f
    pL, pR = Vg[2, iL], Vg[2, iR]
    uL, uR = Vg[1, iL], Vg[1, iR]
    sp = np.abs(pR - pL) / (np.abs(pR) + np.abs(pL) + 1.0e-14)
    us = np.maximum(np.abs(uL) + np.abs(uR), float(velocity_floor))
    comp = np.maximum(0.0, uL - uR) / (us + 1.0e-14)
    shock = sp >= float(pressure_threshold)
    if compression_threshold > 0.0:
        shock &= comp >= float(compression_threshold)
    shock[[0, -1]] = False
    return _dilate_face_mask_nonperiodic(shock, buffer_faces), sp, comp


def get_face_fluxes_nonperiodic(V, left_bc, right_bc, gammaS, e0S,
                                zeta=0.2, limit_ho=False,
                                use_thinc_bvd=True, thinc_beta=1.6,
                                shock_pressure_threshold=2.0e-2,
                                shock_compression_threshold=0.0,
                                shock_buffer_faces=2):
    """Construit les flux DF sur les Nx+1 faces physiques."""
    ng = 2
    Vg = fill_primitive_ghosts(V, left_bc, right_bc, ng)
    Vho, VL, VR = reconstruct_nonperiodic(Vg, ng, limit_ho=limit_ho)
    nx = V.shape[1]

    # Prolongement des parametres geles selon les proprietaires des faces.
    gL = np.empty(nx + 1); gR = np.empty(nx + 1)
    eL = np.empty(nx + 1); eR = np.empty(nx + 1)
    gL[1:] = gammaS; gL[0] = gammaS[0]
    gR[:-1] = gammaS; gR[-1] = gammaS[-1]
    eL[1:] = e0S; eL[0] = e0S[0]
    eR[:-1] = e0S; eR[-1] = e0S[-1]

    shock_faces, shock_pressure_jump, shock_compression = _shock_face_mask_nonperiodic(
        Vg, nx, nghost=ng, pressure_threshold=shock_pressure_threshold,
        compression_threshold=shock_compression_threshold,
        buffer_faces=shock_buffer_faces,
    )
    shock_cells = shock_faces[:-1] | shock_faces[1:]

    thinc_cells = np.zeros(nx, dtype=bool)
    thinc_faces = np.zeros(nx + 1, dtype=bool)
    if use_thinc_bvd:
        thinc_cells, rho_t_l, rho_t_r = _contact_thinc_selection(
            Vg, VL, VR, beta=thinc_beta, nghost=ng
        )
        thinc_cells &= ~shock_cells
        # Replace density traces contributed by each physical cell.
        for i in np.flatnonzero(thinc_cells):
            VL[0, i + 1] = rho_t_r[i]   # right face of cell i
            VR[0, i] = rho_t_l[i]       # left face of cell i
        thinc_faces[:-1] |= thinc_cells
        thinc_faces[1:] |= thinc_cells

    # Capteur RS equivalent a la branche periodique : on juge la valeur
    # HO reconstruite a la face par rapport aux deux cellules adjacentes.
    # Un simple saut cellule-cellule ne couvre pas les faces voisines ou la
    # reconstruction centree peut deja sur/sous-osciller.
    rhoL = Vg[0, ng - 1:ng - 1 + nx + 1]
    rhoR = Vg[0, ng:ng + nx + 1]
    rho_face_ho = Vho[0]
    indicator_left = np.abs(rho_face_ho - rhoL) / (np.abs(rhoL) + 1.0e-14)
    indicator_right = np.abs(rho_face_ho - rhoR) / (np.abs(rhoR) + 1.0e-14)
    indicator_rs = np.maximum(indicator_left, indicator_right)
    hllc = (indicator_rs > zeta) | shock_faces | thinc_faces
    hllc[[0, -1]] = True

    mass = np.empty(nx + 1); mom = np.empty(nx + 1)
    EL = np.empty(nx + 1); ER = np.empty(nx + 1)
    idx = ~hllc
    if np.any(idx):
        ff = flux_ho(Vho[:, idx], gL[idx], eL[idx], gR[idx], eR[idx])
        mass[idx], mom[idx] = ff.mass, ff.momentum
        EL[idx], ER[idx] = ff.energy_left_owner, ff.energy_right_owner
    idx = hllc
    if np.any(idx):
        ff = flux_hllc_df(VL[:, idx], VR[:, idx], gL[idx], eL[idx], gR[idx], eR[idx])
        mass[idx], mom[idx] = ff.mass, ff.momentum
        EL[idx], ER[idx] = ff.energy_left_owner, ff.energy_right_owner

    diag = NonPeriodicDiagnostics(hllc, thinc_cells, thinc_faces, shock_faces, shock_cells)
    return mass, mom, EL, ER, diag


def assemble_rhs_nonperiodic(mass, momentum, energy_left, energy_right, dx):
    """Divergence pour Nx cellules et Nx+1 faces."""
    nx = mass.size - 1
    rhs = np.empty((3, nx))
    rhs[0] = -(mass[1:] - mass[:-1]) / dx
    rhs[1] = -(momentum[1:] - momentum[:-1]) / dx
    # face droite vue par cellule gauche; face gauche vue par cellule droite.
    rhs[2] = -(energy_left[1:] - energy_right[:-1]) / dx
    return rhs




NONPERIODIC_LEVEL_NAMES = {
    0: "FIRST_ORDER_LLF",
    1: "ENO2_HLLC",
    2: "HYBRID",
    3: "THINC_BVD_HYBRID",
}


def face_levels_from_cells_nonperiodic(cell_levels):
    """Construit les Nx+1 niveaux de face sans raccord periodique.

    Les faces de bord prennent le niveau de la cellule physique adjacente.
    Les faces internes prennent le minimum des deux cellules.
    """
    levels = np.asarray(cell_levels, dtype=np.int8)
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError("cell_levels doit etre un vecteur non vide")
    face = np.empty(levels.size + 1, dtype=np.int8)
    face[0] = levels[0]
    face[-1] = levels[-1]
    if levels.size > 1:
        face[1:-1] = np.minimum(levels[:-1], levels[1:])
    return face


def _owner_thermo_nonperiodic(gammaS, e0S):
    nx = gammaS.size
    gL = np.empty(nx + 1); gR = np.empty(nx + 1)
    eL = np.empty(nx + 1); eR = np.empty(nx + 1)
    gL[1:] = gammaS; gL[0] = gammaS[0]
    gR[:-1] = gammaS; gR[-1] = gammaS[-1]
    eL[1:] = e0S; eL[0] = e0S[0]
    eR[:-1] = e0S; eR[-1] = e0S[-1]
    return gL, eL, gR, eR


def _fluxes_at_level_nonperiodic(level, V, left_bc, right_bc, gammaS, e0S,
                                  zeta=0.2, limit_ho=False,
                                  thinc_beta=1.6,
                                  shock_pressure_threshold=2.0e-2,
                                  shock_compression_threshold=0.0,
                                  shock_buffer_faces=2):
    """Calcule une famille complete de flux pour un niveau MOOD."""
    level = int(level)
    nx = V.shape[1]
    if level in (2, 3):
        mass, mom, EL, ER, diag = get_face_fluxes_nonperiodic(
            V, left_bc, right_bc, gammaS, e0S,
            zeta=zeta, limit_ho=limit_ho,
            use_thinc_bvd=(level == 3), thinc_beta=thinc_beta,
            shock_pressure_threshold=shock_pressure_threshold,
            shock_compression_threshold=shock_compression_threshold,
            shock_buffer_faces=shock_buffer_faces,
        )
        return dict(mass=mass, momentum=mom, energy_left=EL,
                    energy_right=ER, hllc_face_mask=diag.hllc_face_mask,
                    thinc_cell_mask=diag.thinc_cell_mask,
                    thinc_face_mask=diag.thinc_face_mask)

    ng = 2
    Vg = fill_primitive_ghosts(V, left_bc, right_bc, ng)
    gL, eL, gR, eR = _owner_thermo_nonperiodic(gammaS, e0S)
    if level == 1:
        _, VL, VR = reconstruct_nonperiodic(Vg, ng, limit_ho=limit_ho)
        ff = flux_hllc_df(VL, VR, gL, eL, gR, eR)
    elif level == 0:
        # Etats constants situes de part et d'autre de chaque face physique.
        f = np.arange(nx + 1)
        VL = Vg[:, ng + f - 1]
        VR = Vg[:, ng + f]
        ff = flux_llf_df(VL, VR, gL, eL, gR, eR)
    else:
        raise ValueError(f"niveau MOOD non periodique invalide: {level}")
    return dict(mass=ff.mass, momentum=ff.momentum,
                energy_left=ff.energy_left_owner,
                energy_right=ff.energy_right_owner,
                hllc_face_mask=np.ones(nx + 1, dtype=bool),
                thinc_cell_mask=np.zeros(nx, dtype=bool),
                thinc_face_mask=np.zeros(nx + 1, dtype=bool))


def _mix_nonperiodic_fluxes(fluxes_by_level, face_levels):
    nface = face_levels.size
    out = {
        'mass': np.empty(nface), 'momentum': np.empty(nface),
        'energy_left': np.empty(nface), 'energy_right': np.empty(nface),
        'hllc_face_mask': np.empty(nface, dtype=bool),
        'thinc_face_mask': np.empty(nface, dtype=bool),
    }
    for level, flux in fluxes_by_level.items():
        m = face_levels == int(level)
        if not np.any(m):
            continue
        for key in ('mass','momentum','energy_left','energy_right',
                    'hllc_face_mask','thinc_face_mask'):
            out[key][m] = flux[key][m]
    return out


def _detect_nonperiodic_candidate(W_after, V_eval, left_bc, right_bc,
                                  gammaS, e0S, atol=1e-12, rtol=1e-9,
                                  check_entropy=False):
    nx = W_after.shape[1]
    bad_finite = ~np.isfinite(W_after).all(axis=0)
    bad_density = (~np.isfinite(W_after[0])) | (W_after[0] <= 0.0)
    bad_pressure = np.zeros(nx, dtype=bool)
    bad_entropy = np.zeros(nx, dtype=bool)
    V_after = None
    min_margin = float('nan')
    if not (bad_finite.any() or bad_density.any()):
        try:
            V_after = update_primitives_df(W_after, gammaS, e0S)
            bad_finite |= ~np.isfinite(V_after).all(axis=0)
            bad_pressure = (~np.isfinite(V_after[2])) | (V_after[2] <= 0.0)
            if check_entropy:
                # boundary.fill_primitive_ghosts exige au moins deux couches.
                # L'ancien appel avec nghost=1 levait une ValueError, ensuite
                # convertie silencieusement en bad_finite=True pour TOUTES les
                # cellules. Cela faisait echouer meme dt=0 au niveau LLF.
                ng_entropy = 2
                Vg = fill_primitive_ghosts(
                    V_eval, left_bc, right_bc, ng_entropy
                )
                rho_g, p_g = Vg[0], Vg[2]
                s_cand = entropy_double_flux(V_after, gammaS)
                # Pour chaque cellule physique j, ses voisins gauche/centre/
                # droite sont evalues avec le gamma* gele de CETTE cellule.
                sl = slice(ng_entropy - 1, ng_entropy - 1 + nx)
                sc = slice(ng_entropy, ng_entropy + nx)
                sr = slice(ng_entropy + 1, ng_entropy + 1 + nx)
                s0 = np.log(p_g[sc]) - gammaS * np.log(rho_g[sc] * R_N2)
                sL = np.log(p_g[sl]) - gammaS * np.log(rho_g[sl] * R_N2)
                sR = np.log(p_g[sr]) - gammaS * np.log(rho_g[sr] * R_N2)
                bound = np.minimum(np.minimum(sL, s0), sR)
                tol = atol + rtol * np.maximum(1.0, np.abs(bound))
                margin = s_cand - bound
                bad_entropy = (~np.isfinite(s_cand)) | (margin < -tol)
                if np.isfinite(margin).any():
                    min_margin = float(np.nanmin(margin))
        except (RuntimeError, ValueError, FloatingPointError):
            bad_finite[:] = True
            V_after = None
    troubled = bad_finite | bad_density | bad_pressure | bad_entropy
    return dict(
        troubled=troubled, bad_finite=bad_finite,
        bad_density=bad_density, bad_pressure=bad_pressure,
        bad_entropy=bad_entropy, V_after=V_after,
        min_density=float(np.nanmin(W_after[0])) if np.isfinite(W_after[0]).any() else float('nan'),
        min_pressure=float(np.nanmin(V_after[2])) if V_after is not None else float('nan'),
        min_entropy_margin=min_margin,
    )


def correct_euler_nonperiodic_mood_local(
        V_eval, W_before, dx, dt, left_bc, right_bc, gammaS, e0S,
        zeta=0.2, limit_ho=False, use_thinc_bvd=True, thinc_beta=1.6,
        shock_pressure_threshold=2.0e-2, shock_compression_threshold=0.0,
        shock_buffer_faces=2, atol=1e-12, rtol=1e-9, check_entropy=False, initial_cell_levels=None,
        max_iterations=None, verbose=False):
    """Sous-pas Euler non periodique avec cascade MOOD locale 3->2->1->0."""
    nx = V_eval.shape[1]
    levels = [3, 2, 1, 0] if use_thinc_bvd else [2, 1, 0]
    top = levels[0]
    fluxes = {}
    failures = {}
    for level in levels:
        try:
            fluxes[level] = _fluxes_at_level_nonperiodic(
                level, V_eval, left_bc, right_bc, gammaS, e0S,
                zeta=zeta, limit_ho=limit_ho, thinc_beta=thinc_beta,
                shock_pressure_threshold=shock_pressure_threshold,
                shock_compression_threshold=shock_compression_threshold,
                shock_buffer_faces=shock_buffer_faces)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            failures[level] = str(exc)
    if 0 not in fluxes:
        raise RuntimeError("Le niveau terminal LLF non periodique est indisponible")

    if initial_cell_levels is None:
        cell_levels = np.full(nx, top, dtype=np.int8)
    else:
        cell_levels = np.asarray(initial_cell_levels, dtype=np.int8).copy()
        if cell_levels.shape != (nx,):
            raise ValueError("initial_cell_levels doit etre de forme (Nx,)")
    available = sorted(fluxes, reverse=True)
    for i, requested in enumerate(cell_levels):
        choices = [ell for ell in available if ell <= int(requested)]
        cell_levels[i] = max(choices) if choices else 0

    if max_iterations is None:
        max_iterations = nx * top + 1
    history = []
    for iteration in range(max_iterations):
        face_levels = face_levels_from_cells_nonperiodic(cell_levels)
        mixed = _mix_nonperiodic_fluxes(fluxes, face_levels)
        rhs = assemble_rhs_nonperiodic(mixed['mass'], mixed['momentum'],
                                       mixed['energy_left'], mixed['energy_right'], dx)
        W_after = W_before + dt * rhs
        det = _detect_nonperiodic_candidate(W_after, V_eval, left_bc, right_bc,
                                            gammaS, e0S, atol=atol, rtol=rtol, check_entropy=check_entropy)
        history.append(dict(
            iteration=iteration, n_troubled=int(det['troubled'].sum()),
            troubled_indices=np.flatnonzero(det['troubled']).tolist(),
            n_bad_density=int(det['bad_density'].sum()),
            n_bad_pressure=int(det['bad_pressure'].sum()),
            n_bad_finite=int(det['bad_finite'].sum()),
            n_bad_entropy=int(det['bad_entropy'].sum()),
            min_density=det['min_density'], min_pressure=det['min_pressure'],
            min_entropy_margin=det['min_entropy_margin'],
            min_cell_level=int(cell_levels.min()),
            level_counts={ell:int(np.count_nonzero(cell_levels == ell)) for ell in levels},
            n_corrected_faces=int(np.count_nonzero(face_levels < top)),
        ))
        if verbose:
            print(f"  [MOOD-NP] iteration {iteration}: {det['troubled'].sum()} troubled")
        if not det['troubled'].any():
            # Cell mask THINC provenant du niveau 3, filtre par cellules encore niveau 3.
            thinc_cells = fluxes.get(3, {}).get('thinc_cell_mask', np.zeros(nx, bool)).copy()
            thinc_cells &= cell_levels == 3
            return dict(
                W_after=W_after, V_after=det['V_after'],
                accepted_level=int(cell_levels.min()),
                accepted_level_name=NONPERIODIC_LEVEL_NAMES[int(cell_levels.min())],
                cell_levels=cell_levels.copy(), face_levels=face_levels.copy(),
                corrected_faces=(face_levels < top),
                mass_flux=mixed['mass'].copy(),
                hllc_face_mask=mixed['hllc_face_mask'].copy(),
                thinc_face_mask=mixed['thinc_face_mask'].copy(),
                thinc_cell_mask=thinc_cells,
                history=history, n_iterations=iteration,
                flux_failures=failures,
            )
        terminal = det['troubled'] & (cell_levels == 0)
        if terminal.any():
            raise RuntimeError(
                "Cellules non admissibles au niveau terminal LLF; reduire le pas de temps")
        cell_levels[det['troubled']] -= 1
        for i in np.flatnonzero(det['troubled']):
            choices = [ell for ell in available if ell <= int(cell_levels[i])]
            cell_levels[i] = max(choices) if choices else 0
    raise RuntimeError("MOOD non periodique: nombre maximal d'iterations atteint")

def _euler(V, W, dx, dt, left_bc, right_bc, gammaS, e0S, **options):
    mass, mom, EL, ER, diag = get_face_fluxes_nonperiodic(
        V, left_bc, right_bc, gammaS, e0S, **options
    )
    rhs = assemble_rhs_nonperiodic(mass, mom, EL, ER, dx)
    Wn = W + dt * rhs
    Vn = update_primitives_df(Wn, gammaS, e0S)
    if not np.all(np.isfinite(Vn)) or np.min(Vn[0]) <= 0 or np.min(Vn[2]) <= 0:
        raise RuntimeError("etat non admissible dans la branche non periodique")
    return Wn, Vn, diag


def advance_ssprk3_nonperiodic_legacy(V, W, dx, dt, left_bc, right_bc, **options):
    """Un pas SSP-RK3 avec un gel double-flux unique."""
    gammaS, e0S = get_gammaS_e0S(V)
    W1, V1, d1 = _euler(V, W, dx, dt, left_bc, right_bc, gammaS, e0S, **options)
    W2e, _, d2 = _euler(V1, W1, dx, dt, left_bc, right_bc, gammaS, e0S, **options)
    W2 = 0.75 * W + 0.25 * W2e
    V2 = update_primitives_df(W2, gammaS, e0S)
    W3e, _, d3 = _euler(V2, W2, dx, dt, left_bc, right_bc, gammaS, e0S, **options)
    Wn = (1.0/3.0) * W + (2.0/3.0) * W3e
    Vn = update_primitives_df(Wn, gammaS, e0S)
    return Wn, Vn, (d1, d2, d3)




def advance_ssprk3_nonperiodic(V, W, dx, dt, left_bc, right_bc,
                               mood_mode="local_multilevel", **options):
    """Pas SSP-RK3 non periodique, legacy ou MOOD local multiniveau."""
    if mood_mode == "legacy":
        return advance_ssprk3_nonperiodic_legacy(
            V, W, dx, dt, left_bc, right_bc, **options)
    if mood_mode != "local_multilevel":
        raise ValueError("mood_mode doit valoir 'legacy' ou 'local_multilevel'")
    gammaS, e0S = get_gammaS_e0S(V)
    r1 = correct_euler_nonperiodic_mood_local(
        V, W, dx, dt, left_bc, right_bc, gammaS, e0S, **options)
    W1, V1 = r1['W_after'], r1['V_after']
    r2 = correct_euler_nonperiodic_mood_local(
        V1, W1, dx, dt, left_bc, right_bc, gammaS, e0S, **options)
    W2 = 0.75 * W + 0.25 * r2['W_after']
    V2 = update_primitives_df(W2, gammaS, e0S)
    r3 = correct_euler_nonperiodic_mood_local(
        V2, W2, dx, dt, left_bc, right_bc, gammaS, e0S, **options)
    Wn = (1.0/3.0) * W + (2.0/3.0) * r3['W_after']
    Vn = update_primitives_df(Wn, gammaS, e0S)
    if (not np.all(np.isfinite(Vn)) or np.min(Vn[0]) <= 0.0
            or np.min(Vn[2]) <= 0.0):
        raise RuntimeError("etat final SSP-RK3 non admissible")
    return Wn, Vn, (r1, r2, r3)

def run_nitrogen_interface_nonperiodic(Nx=150, CFL=0.5, t_final=0.002,
                                       use_thinc_bvd=True, limit_ho=False,
                                       zeta=0.2, verbose=False,
                                       mood_mode="local_multilevel"):
    """Advection de l'interface N2 avec entree Dirichlet et sortie transmissive."""
    x, dx, V = nitrogen_interface_ic(Nx)
    V_initial = V.copy()
    W = primitives_to_conservatives(V)
    inlet = V_initial[:, 0].copy()
    left_bc = dirichlet(inlet)
    right_bc = transmissive()

    t = 0.0; step = 0
    max_hllc = 0.0; max_thinc = 0.0
    min_mood_level = 3 if mood_mode != "legacy" else None
    max_corrected_face_fraction = 0.0
    max_terminal_face_fraction = 0.0
    while t < t_final:
        dt = CFL * dx / np.max(np.abs(V[1]) + get_sos(V))
        dt = min(dt, t_final - t)
        Wn, Vn, diags = advance_ssprk3_nonperiodic(
            V, W, dx, dt, left_bc, right_bc,
            zeta=zeta, limit_ho=limit_ho, use_thinc_bvd=use_thinc_bvd,
            mood_mode=mood_mode,
        )
        # Reprojection sur l'EOS reelle.
        V = Vn
        W = primitives_to_conservatives(V)
        t += dt; step += 1
        if mood_mode == "legacy":
            max_hllc = max(max_hllc, *(d.hllc_face_mask.mean() for d in diags))
            max_thinc = max(max_thinc, *(d.thinc_cell_mask.mean() for d in diags))
        else:
            max_hllc = max(max_hllc, *(d["hllc_face_mask"].mean() for d in diags))
            max_thinc = max(max_thinc, *(d["thinc_cell_mask"].mean() for d in diags))
            min_mood_level = min(min_mood_level, *(d["accepted_level"] for d in diags))
            max_corrected_face_fraction = max(
                max_corrected_face_fraction,
                *(d["corrected_faces"].mean() for d in diags),
            )
            max_terminal_face_fraction = max(
                max_terminal_face_fraction,
                *(np.mean(d["face_levels"] == 0) for d in diags),
            )

    shift = V_initial[1, 0] * t_final
    # Pour u>0, l'etat exact est l'IC translatee, completee a gauche par l'inlet.
    x_source = x - shift
    inside = (x_source >= 0.25) & (x_source <= 0.75)
    T_exact = np.where(inside, 100.0, 300.0)
    p_exact = np.full(Nx, 5.0e6)
    rho_exact = get_rho_from_T_p(T_exact, p_exact)
    V_exact = np.vstack((rho_exact, np.full(Nx, 100.0), p_exact))

    return dict(x=x, dx=dx, V=V, W=W, V_initial=V_initial, V_exact=V_exact,
                t_final=t_final, step=step, max_hllc_face_fraction=max_hllc,
                max_thinc_cell_fraction=max_thinc,
                rho_L1=float(np.mean(np.abs(V[0]-rho_exact))),
                pressure_Linf=float(np.max(np.abs(V[2]-5.0e6))),
                velocity_Linf=float(np.max(np.abs(V[1]-100.0))),
                boundary_flux_layout="Nx+1", mood_mode=mood_mode,
                min_accepted_mood_level=min_mood_level,
                max_corrected_face_fraction=max_corrected_face_fraction,
                max_terminal_face_fraction=max_terminal_face_fraction)


# ---------------------------------------------------------------------------
# Etape 14 : tube a choc transcritique de Ma, Lv & Ihme (2017), Sec. 5.2.
# ---------------------------------------------------------------------------

def transcritical_shock_tube_ic(
        Nx, x_left=0.0, x_right=1.0, diaphragm=0.5,
        rho_left=800.0, p_left=60.0e6, u_left=0.0,
        rho_right=80.0, p_right=6.0e6, u_right=0.0):
    """Condition initiale du tube a choc transcritique d'azote.

    Les etats par defaut reproduisent l'Eq. (33) de Ma, Lv & Ihme (2017).
    La discontinuite est placee entre cellules suivant ``x < diaphragm``.
    """
    if int(Nx) != Nx or Nx < 8:
        raise ValueError("Nx doit etre un entier >= 8")
    Nx = int(Nx)
    if not x_left < diaphragm < x_right:
        raise ValueError("diaphragm doit appartenir strictement au domaine")
    states = np.array([
        rho_left, p_left, rho_right, p_right
    ], dtype=float)
    if (not np.all(np.isfinite(states)) or rho_left <= 0.0
            or rho_right <= 0.0 or p_left <= 0.0 or p_right <= 0.0):
        raise ValueError("les densites et pressions initiales doivent etre finies et positives")

    dx = (x_right - x_left) / Nx
    x = x_left + (np.arange(Nx) + 0.5) * dx
    left = x < diaphragm
    rho = np.where(left, rho_left, rho_right)
    u = np.where(left, u_left, u_right)
    p = np.where(left, p_left, p_right)
    V = np.vstack((rho, u, p))
    # Validation EOS immediate : le test doit echouer tot si l'un des deux
    # etats n'appartient pas au domaine thermodynamique exploitable.
    T = get_T_from_p_rho(p, rho)
    c = get_sos(V)
    if (not np.all(np.isfinite(T)) or np.any(T <= 0.0)
            or not np.all(np.isfinite(c)) or np.any(c <= 0.0)):
        raise ValueError("etat initial hors domaine thermodynamique admissible")
    return x, dx, V


def run_transcritical_shock_tube(
        Nx=512, CFL=0.8, t_final=5.0e-4,
        use_thinc_bvd=True, limit_ho=False, zeta=0.2,
        mood_mode="local_multilevel", check_entropy=False,
        shock_pressure_threshold=2.0e-2, shock_compression_threshold=0.0,
        shock_buffer_faces=2, verbose=False, max_steps=200000, return_history=False):
    """Execute le tube a choc transcritique d'azote avec parois adiabatiques.

    Le niveau terminal LLF est disponible dans chaque stage SSP-RK3. Les
    diagnostics quantifient les activations des niveaux, les extrema et
    l'erreur quasi-conservative d'energie totale.
    """
    x, dx, V = transcritical_shock_tube_ic(Nx)
    V_initial = V.copy()
    W = primitives_to_conservatives(V)
    W_initial = W.copy()
    left_bc = adiabatic_wall()
    right_bc = adiabatic_wall()

    t = 0.0
    step = 0
    min_mood_level = 3 if mood_mode != "legacy" else None
    max_corrected_face_fraction = 0.0
    max_terminal_face_fraction = 0.0
    max_thinc_cell_fraction = 0.0
    max_hllc_face_fraction = 0.0
    level_face_updates = {0: 0, 1: 0, 2: 0, 3: 0}
    level_cell_updates = {0: 0, 1: 0, 2: 0, 3: 0}
    stage_min_level_counts = [{0: 0, 1: 0, 2: 0, 3: 0} for _ in range(3)]
    history = []

    while t < t_final:
        wavespeed = np.abs(V[1]) + get_sos(V)
        if not np.all(np.isfinite(wavespeed)) or np.max(wavespeed) <= 0.0:
            raise RuntimeError("vitesse caracteristique invalide")
        dt = CFL * dx / np.max(wavespeed)
        dt = min(dt, t_final - t)
        if dt <= 0.0 or not np.isfinite(dt):
            raise RuntimeError("pas de temps invalide")

        step_options = dict(
            zeta=zeta, limit_ho=limit_ho,
            use_thinc_bvd=use_thinc_bvd,
            shock_pressure_threshold=shock_pressure_threshold,
            shock_compression_threshold=shock_compression_threshold,
            shock_buffer_faces=shock_buffer_faces,
        )
        if mood_mode != "legacy":
            step_options["check_entropy"] = check_entropy
        Wn, Vn, stages = advance_ssprk3_nonperiodic(
            V, W, dx, dt, left_bc, right_bc,
            mood_mode=mood_mode, **step_options,
        )

        # Reprojection EOS reelle apres acceptation du pas SSP-RK3 complet.
        V = Vn
        W = primitives_to_conservatives(V)
        t += dt
        step += 1
        if step > max_steps:
            raise RuntimeError("max_steps atteint avant t_final")

        if mood_mode == "legacy":
            max_hllc_face_fraction = max(
                max_hllc_face_fraction,
                *(float(d.hllc_face_mask.mean()) for d in stages),
            )
            max_thinc_cell_fraction = max(
                max_thinc_cell_fraction,
                *(float(d.thinc_cell_mask.mean()) for d in stages),
            )
        else:
            for istage, d in enumerate(stages):
                ell = int(d["accepted_level"])
                stage_min_level_counts[istage][ell] += 1
                min_mood_level = min(min_mood_level, ell)
                max_corrected_face_fraction = max(
                    max_corrected_face_fraction,
                    float(d["corrected_faces"].mean()),
                )
                max_terminal_face_fraction = max(
                    max_terminal_face_fraction,
                    float(np.mean(d["face_levels"] == 0)),
                )
                max_hllc_face_fraction = max(
                    max_hllc_face_fraction,
                    float(d["hllc_face_mask"].mean()),
                )
                max_thinc_cell_fraction = max(
                    max_thinc_cell_fraction,
                    float(d["thinc_cell_mask"].mean()),
                )
                for level in range(4):
                    level_face_updates[level] += int(np.count_nonzero(d["face_levels"] == level))
                    level_cell_updates[level] += int(np.count_nonzero(d["cell_levels"] == level))

        if return_history:
            T_now = get_T_from_p_rho(V[2], V[0])
            history.append(dict(
                step=step, time=float(t), dt=float(dt),
                rho_min=float(V[0].min()), rho_max=float(V[0].max()),
                u_min=float(V[1].min()), u_max=float(V[1].max()),
                p_min=float(V[2].min()), p_max=float(V[2].max()),
                T_min=float(T_now.min()), T_max=float(T_now.max()),
                min_mood_level=(None if mood_mode == "legacy" else
                                min(int(d["accepted_level"]) for d in stages)),
            ))
        if verbose and (step == 1 or step % 100 == 0 or t >= t_final):
            print(f"[shock tube] step={step:5d}, t={t:.6e}, "
                  f"rho=[{V[0].min():.3f},{V[0].max():.3f}], "
                  f"p=[{V[2].min()/1e6:.3f},{V[2].max()/1e6:.3f}] MPa")

    T = get_T_from_p_rho(V[2], V[0])
    c = get_sos(V)
    if (not np.all(np.isfinite(T)) or np.any(T <= 0.0)
            or not np.all(np.isfinite(c)) or np.any(c <= 0.0)):
        raise RuntimeError("etat thermodynamique final non admissible")

    # Les parois imposent un flux massique et energetique nul. La masse est
    # donc conservative; l'energie DF n'est que quasi-conservative.
    mass_initial = float(dx * np.sum(W_initial[0]))
    mass_final = float(dx * np.sum(W[0]))
    energy_initial = float(dx * np.sum(W_initial[2]))
    energy_final = float(dx * np.sum(W[2]))
    mass_rel_error = abs(mass_final - mass_initial) / max(abs(mass_initial), 1.0)
    energy_rel_error = abs(energy_final - energy_initial) / max(abs(energy_initial), 1.0)

    total_face_updates = max(1, 3 * step * (Nx + 1))
    total_cell_updates = max(1, 3 * step * Nx)
    face_level_fractions = {
        level: level_face_updates[level] / total_face_updates for level in range(4)
    }
    cell_level_fractions = {
        level: level_cell_updates[level] / total_cell_updates for level in range(4)
    }

    return dict(
        x=x, dx=dx, V=V, W=W, T=T, speed_of_sound=c,
        V_initial=V_initial, W_initial=W_initial,
        t_final=float(t), step=step, CFL=float(CFL), Nx=int(Nx),
        mood_mode=mood_mode, use_thinc_bvd=bool(use_thinc_bvd),
        min_accepted_mood_level=min_mood_level,
        max_corrected_face_fraction=max_corrected_face_fraction,
        max_terminal_face_fraction=max_terminal_face_fraction,
        max_thinc_cell_fraction=max_thinc_cell_fraction,
        max_hllc_face_fraction=max_hllc_face_fraction,
        face_level_fractions=face_level_fractions,
        cell_level_fractions=cell_level_fractions,
        stage_min_level_counts=stage_min_level_counts,
        mass_initial=mass_initial, mass_final=mass_final,
        mass_rel_error=mass_rel_error,
        energy_initial=energy_initial, energy_final=energy_final,
        energy_rel_error=energy_rel_error,
        rho_min=float(V[0].min()), rho_max=float(V[0].max()),
        u_min=float(V[1].min()), u_max=float(V[1].max()),
        p_min=float(V[2].min()), p_max=float(V[2].max()),
        T_min=float(T.min()), T_max=float(T.max()),
        history=history if return_history else None,
        boundary_condition="solid_adiabatic_wall",
    )
