"""Sweeps directionnels 2D conservatives avec energie totale commune.

Convention V2D=(rho,u,v,p), forme (4,Ny,Nx). Chaque ligne/colonne est
avancee avec Q=(rho,rho*u,rho*v,rho*E), ou E contient u^2+v^2.
"""
import numpy as np

from numerics_2d import primitives_to_conservatives_2d, get_gammaS_e0S_2d
from directional_2d import advance_ssprk3_directional_2d


def _validate(V2D):
    V2D=np.asarray(V2D,dtype=float)
    if V2D.ndim!=3 or V2D.shape[0]!=4:
        raise ValueError("V2D doit etre de forme (4, Ny, Nx)")
    if not np.all(np.isfinite(V2D)) or np.any(V2D[0]<=0) or np.any(V2D[3]<=0):
        raise ValueError("etat 2D non admissible")
    return V2D


def _clean_options(options):
    options=dict(options)
    # Ces options appartenaient aux wrappers 1D. Le noyau 4D utilise
    # toujours la cascade locale et l'admissibilite rho/p.
    options.pop("mood_mode",None)
    options.pop("check_entropy",None)
    return options


def sweep_x_step(V2D, dx, dt, **options):
    """Sweep x periodique conservatif sur les quatre equations Euler 2D."""
    V2D=_validate(V2D); options=_clean_options(options)
    _,Ny,Nx=V2D.shape
    out=np.empty_like(V2D); diagnostics=[]
    for j in range(Ny):
        V=V2D[:,j,:]
        Q=primitives_to_conservatives_2d(V)
        gammaS,e0S=get_gammaS_e0S_2d(V)
        res=advance_ssprk3_directional_2d(
            V,Q,dx,dt,gammaS,e0S,"x",periodic=True,**options)
        out[:,j,:]=res["V_after"]
        diagnostics.append(res)
    return out,diagnostics


def sweep_y_step(V2D, dy, dt, wall_bc, **options):
    """Sweep y non periodique conservatif, avec vitesse v normale au mur."""
    V2D=_validate(V2D); options=_clean_options(options)
    _,Ny,Nx=V2D.shape
    out=np.empty_like(V2D); diagnostics=[]
    for i in range(Nx):
        V=V2D[:,:,i]
        Q=primitives_to_conservatives_2d(V)
        gammaS,e0S=get_gammaS_e0S_2d(V)
        res=advance_ssprk3_directional_2d(
            V,Q,dy,dt,gammaS,e0S,"y",periodic=False,
            left_bc=wall_bc,right_bc=wall_bc,**options)
        out[:,:,i]=res["V_after"]
        diagnostics.append(tuple(res["stage_results"]))
    return out,diagnostics


def sweep_y_periodic_step(V2D,dy,dt,**options):
    """Sweep y periodique conservatif, sans rotation approximative d'energie."""
    V2D=_validate(V2D); options=_clean_options(options)
    _,Ny,Nx=V2D.shape
    out=np.empty_like(V2D); diagnostics=[]
    for i in range(Nx):
        V=V2D[:,:,i]
        Q=primitives_to_conservatives_2d(V)
        gammaS,e0S=get_gammaS_e0S_2d(V)
        res=advance_ssprk3_directional_2d(
            V,Q,dy,dt,gammaS,e0S,"y",periodic=True,**options)
        out[:,:,i]=res["V_after"]
        diagnostics.append(res)
    return out,diagnostics


def extract_level_field_x(diagnostics, Ny, Nx):
    """Etape 2D-4/2D-5 : assemble le champ 2D ell^x_{i,j} = niveau MOOD
    minimal accepte sur les 3 sous-etapes SSP-RK3 du sweep x, PAR
    CELLULE (pas juste le pire cas agrege sur toute la ligne comme le
    fait 'accepted_level'). Vaut -1 la ou aucune information par cellule
    n'est disponible (mode 'legacy', qui n'a pas de cascade MOOD locale).

    diagnostics : liste retournee par sweep_x_step (longueur Ny).
    """
    level_x = np.full((Ny, Nx), -1, dtype=int)
    for j, res in enumerate(diagnostics):
        stage_levels = [s['cell_levels'] for s in res['stage_results'] if 'cell_levels' in s]
        if stage_levels:
            level_x[j, :] = np.min(np.stack(stage_levels), axis=0)
    return level_x


def extract_level_field_y(diagnostics, Ny, Nx):
    """Equivalent de extract_level_field_x pour sweep_y_step.

    diagnostics : liste de tuples (r1,r2,r3) retournee par sweep_y_step
    (longueur Nx).
    """
    level_y = np.full((Ny, Nx), -1, dtype=int)
    for i, item in enumerate(diagnostics):
        stages = item['stage_results'] if isinstance(item, dict) else item
        stage_levels = [s['cell_levels'] for s in stages if 'cell_levels' in s]
        if stage_levels:
            level_y[:, i] = np.min(np.stack(stage_levels), axis=0)
    return level_y


def extract_thinc_field_x(diagnostics, Ny, Nx):
    """Champ 2D booleen : cellule ayant utilise THINC-BVD au moins une
    fois sur les 3 sous-etapes du sweep x. (Note : la cle exacte differe
    entre les deux branches du code -- 'use_thinc_cell' en periodique,
    'thinc_cell_mask' en non periodique -- gere ici de maniere tolerante.)
    """
    thinc_x = np.zeros((Ny, Nx), dtype=bool)
    for j, res in enumerate(diagnostics):
        masks = [s[k] for s in res['stage_results']
                 for k in ('use_thinc_cell', 'thinc_cell_mask') if k in s]
        if masks:
            thinc_x[j, :] = np.any(np.stack(masks), axis=0)
    return thinc_x


def extract_thinc_field_y(diagnostics, Ny, Nx):
    """Equivalent de extract_thinc_field_x pour sweep_y_step."""
    thinc_y = np.zeros((Ny, Nx), dtype=bool)
    for i, item in enumerate(diagnostics):
        stages = item['stage_results'] if isinstance(item, dict) else item
        masks = [s[k] for s in stages
                 for k in ('use_thinc_cell', 'thinc_cell_mask') if k in s]
        if masks:
            thinc_y[:, i] = np.any(np.stack(masks), axis=0)
    return thinc_y
