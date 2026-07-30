"""Solveur directionnel 1D pour l'etat conservatif Euler 2D.

Etat primitif: V=(rho,u,v,p), etat conservatif commun:
Q=(rho,rho*u,rho*v,rho*E). Les sweeps x/y partagent donc exactement
la meme energie totale 2D, incluant les deux composantes cinetiques.
"""
from __future__ import annotations

import numpy as np

from reconstruction import reconstruct_high_order, reconstruct_high_order_limited
from eno2 import reconstruct_eno2_left, reconstruct_eno2_right
from hybrid_rhs import diagnose_density_bvd_with_wave_sensor
from sensor import rs_face_mask_direct
from face_operators_2d import flux_ho_df_2d, flux_hllc_df_2d, flux_llf_df_2d
from numerics_2d import update_primitives_df_2d
from nonperiodic import (
    reconstruct_nonperiodic,
    _contact_thinc_selection,
    face_levels_from_cells_nonperiodic,
)

LEVEL_NAMES = {0:"FIRST_ORDER_LLF",1:"ENO2_HLLC",2:"HYBRID",3:"THINC_BVD_HYBRID"}


def _normal_triplet(V, direction):
    if direction == "x":
        return np.vstack([V[0], V[1], V[3]])
    if direction == "y":
        return np.vstack([V[0], V[2], V[3]])
    raise ValueError("direction doit valoir 'x' ou 'y'")


def _periodic_owner_thermo(gammaS, e0S):
    return gammaS, e0S, np.roll(gammaS,-1), np.roll(e0S,-1)


def _periodic_face_levels(cell_levels):
    return np.minimum(cell_levels, np.roll(cell_levels,-1))


def _assemble_rhs_periodic(flux, dx):
    rhs = np.empty((4, flux['mass'].size))
    rhs[0] = -(flux['mass'] - np.roll(flux['mass'],1))/dx
    rhs[1] = -(flux['momentum_x'] - np.roll(flux['momentum_x'],1))/dx
    rhs[2] = -(flux['momentum_y'] - np.roll(flux['momentum_y'],1))/dx
    rhs[3] = -(flux['energy_left'] - np.roll(flux['energy_right'],1))/dx
    return rhs


def _assemble_rhs_nonperiodic(flux, dx):
    nx = flux['mass'].size-1
    rhs = np.empty((4,nx))
    rhs[0] = -(flux['mass'][1:]-flux['mass'][:-1])/dx
    rhs[1] = -(flux['momentum_x'][1:]-flux['momentum_x'][:-1])/dx
    rhs[2] = -(flux['momentum_y'][1:]-flux['momentum_y'][:-1])/dx
    rhs[3] = -(flux['energy_left'][1:]-flux['energy_right'][:-1])/dx
    return rhs


def _pack(ff, hllc, thinc_face, thinc_cell):
    return dict(mass=ff.mass, momentum_x=ff.momentum_x, momentum_y=ff.momentum_y,
                energy_left=ff.energy_left_owner, energy_right=ff.energy_right_owner,
                hllc_face_mask=np.asarray(hllc,bool), thinc_face_mask=np.asarray(thinc_face,bool),
                thinc_cell_mask=np.asarray(thinc_cell,bool))


def _periodic_flux_level(level,V,gammaS,e0S,direction, *, zeta=.2,limit_ho=False,
                         use_thinc_bvd=True,thinc_beta=1.6, sensor_options=None):
    n=V.shape[1]
    gL,eL,gR,eR=_periodic_owner_thermo(gammaS,e0S)
    if level==0:
        VL=V; VR=np.roll(V,-1,axis=1)
        ff=flux_llf_df_2d(VL,VR,gL,eL,gR,eR,direction=direction)
        return _pack(ff,np.ones(n,bool),np.zeros(n,bool),np.zeros(n,bool))
    if level==1:
        VL=reconstruct_eno2_left(V); VR=reconstruct_eno2_right(V)
        ff=flux_hllc_df_2d(VL,VR,gL,eL,gR,eR,direction=direction)
        return _pack(ff,np.ones(n,bool),np.zeros(n,bool),np.zeros(n,bool))

    Vho=(reconstruct_high_order_limited(V) if limit_ho else reconstruct_high_order(V))
    VL=reconstruct_eno2_left(V); VR=reconstruct_eno2_right(V)
    thinc_cell=np.zeros(n,bool); thinc_face=np.zeros(n,bool)
    if level==3 and use_thinc_bvd:
        opts={} if sensor_options is None else dict(sensor_options)
        cand,bvd,waves,use=diagnose_density_bvd_with_wave_sensor(
            _normal_triplet(V,direction), limit_ho=limit_ho, thinc_beta=thinc_beta, **opts)
        thinc_cell=use
        rho_left=np.where(use,cand.rho_thinc_left_cell,cand.rho_eno_left_cell)
        rho_right=np.where(use,cand.rho_thinc_right_cell,cand.rho_eno_right_cell)
        VL[0]=rho_right; VR[0]=np.roll(rho_left,-1)
        thinc_face=use|np.roll(use,-1)

    # Meme capteur RS historique que la branche 1D periodique.
    hllc,_ = rs_face_mask_direct(V[0], zeta)
    hllc = hllc | thinc_face

    out={k:np.empty(n,dtype=bool if 'mask' in k else float) for k in
         ('mass','momentum_x','momentum_y','energy_left','energy_right','hllc_face_mask','thinc_face_mask')}
    out['hllc_face_mask']=hllc.copy(); out['thinc_face_mask']=thinc_face.copy(); out['thinc_cell_mask']=thinc_cell.copy()
    m=~hllc
    if np.any(m):
        ff=flux_ho_df_2d(Vho[:,m],gL[m],eL[m],gR[m],eR[m],direction=direction)
        out['mass'][m]=ff.mass; out['momentum_x'][m]=ff.momentum_x; out['momentum_y'][m]=ff.momentum_y
        out['energy_left'][m]=ff.energy_left_owner; out['energy_right'][m]=ff.energy_right_owner
    m=hllc
    if np.any(m):
        ff=flux_hllc_df_2d(VL[:,m],VR[:,m],gL[m],eL[m],gR[m],eR[m],direction=direction)
        out['mass'][m]=ff.mass; out['momentum_x'][m]=ff.momentum_x; out['momentum_y'][m]=ff.momentum_y
        out['energy_left'][m]=ff.energy_left_owner; out['energy_right'][m]=ff.energy_right_owner
    return out


def _fill_ghosts_4(V,left_bc,right_bc,direction,ng=2):
    nx=V.shape[1]; out=np.empty((4,nx+2*ng)); out[:,ng:-ng]=V
    normal=1 if direction=='x' else 2
    transverse=2 if direction=='x' else 1
    def side(bc,is_left):
        src=V[:,:ng][:,::-1] if is_left else V[:,-ng:][:,::-1]
        if bc.kind=='transmissive': src=np.repeat(V[:,[0 if is_left else -1]],ng,axis=1)
        elif bc.kind=='dirichlet':
            # 1D BC stocke (rho,un,p); conserver la vitesse transverse de bord.
            prim=bc.primitive
            src=np.repeat(V[:,[0 if is_left else -1]],ng,axis=1)
            src[0]=prim[0]; src[normal]=prim[1]; src[3]=prim[2]
        elif bc.kind=='wall': src[normal]*=-1.0
        return src
    out[:,:ng]=side(left_bc,True); out[:,-ng:]=side(right_bc,False)
    return out


def _np_owner(gammaS,e0S):
    n=gammaS.size
    gL=np.r_[gammaS[0],gammaS]; gR=np.r_[gammaS,gammaS[-1]]
    eL=np.r_[e0S[0],e0S]; eR=np.r_[e0S,e0S[-1]]
    return gL,eL,gR,eR


def _nonperiodic_flux_level(level,V,gammaS,e0S,direction,left_bc,right_bc,*,zeta=.2,
                            limit_ho=False,use_thinc_bvd=True,thinc_beta=1.6):
    ng=2;n=V.shape[1];Vg=_fill_ghosts_4(V,left_bc,right_bc,direction,ng)
    gL,eL,gR,eR=_np_owner(gammaS,e0S)
    if level==0:
        f=np.arange(n+1); VL=Vg[:,ng+f-1]; VR=Vg[:,ng+f]
        ff=flux_llf_df_2d(VL,VR,gL,eL,gR,eR,direction=direction)
        return _pack(ff,np.ones(n+1,bool),np.zeros(n+1,bool),np.zeros(n,bool))
    Vho,VL,VR=reconstruct_nonperiodic(Vg,ng,limit_ho=limit_ho)
    if level==1:
        ff=flux_hllc_df_2d(VL,VR,gL,eL,gR,eR,direction=direction)
        return _pack(ff,np.ones(n+1,bool),np.zeros(n+1,bool),np.zeros(n,bool))
    thinc_cell=np.zeros(n,bool); thinc_face=np.zeros(n+1,bool)
    if level==3 and use_thinc_bvd:
        # helper attend triplet (rho,un,p)
        Vg3=np.vstack([Vg[0],Vg[1 if direction=='x' else 2],Vg[3]])
        VL3=np.vstack([VL[0],VL[1 if direction=='x' else 2],VL[3]])
        VR3=np.vstack([VR[0],VR[1 if direction=='x' else 2],VR[3]])
        use,rtl,rtr=_contact_thinc_selection(Vg3,VL3,VR3,beta=thinc_beta,nghost=ng)
        thinc_cell=use
        for i in np.flatnonzero(use):
            VL[0,i+1]=rtr[i]; VR[0,i]=rtl[i]
        thinc_face[:-1]|=use; thinc_face[1:]|=use
    rhoL=Vg[0,ng-1:ng+n]; rhoR=Vg[0,ng:ng+n+1]
    ind=np.maximum(np.abs(Vho[0]-rhoL)/(np.abs(rhoL)+1e-14),
                   np.abs(Vho[0]-rhoR)/(np.abs(rhoR)+1e-14))
    hllc=(ind>zeta)|thinc_face
    pL=Vg[3,ng-1:ng+n]; pR=Vg[3,ng:ng+n+1]
    hllc|=np.abs(pR-pL)/(np.abs(pR)+np.abs(pL)+1e-14)>=2e-2
    hllc[[0,-1]]=True
    out={k:np.empty(n+1,dtype=bool if 'mask' in k else float) for k in
         ('mass','momentum_x','momentum_y','energy_left','energy_right','hllc_face_mask','thinc_face_mask')}
    out['hllc_face_mask']=hllc.copy();out['thinc_face_mask']=thinc_face.copy();out['thinc_cell_mask']=thinc_cell.copy()
    m=~hllc
    if np.any(m):
        ff=flux_ho_df_2d(Vho[:,m],gL[m],eL[m],gR[m],eR[m],direction=direction)
        for key,val in [('mass',ff.mass),('momentum_x',ff.momentum_x),('momentum_y',ff.momentum_y),('energy_left',ff.energy_left_owner),('energy_right',ff.energy_right_owner)]: out[key][m]=val
    m=hllc
    if np.any(m):
        ff=flux_hllc_df_2d(VL[:,m],VR[:,m],gL[m],eL[m],gR[m],eR[m],direction=direction)
        for key,val in [('mass',ff.mass),('momentum_x',ff.momentum_x),('momentum_y',ff.momentum_y),('energy_left',ff.energy_left_owner),('energy_right',ff.energy_right_owner)]: out[key][m]=val
    return out


def _mix(fluxes,face_levels):
    n=face_levels.size
    out={k:np.empty(n,dtype=bool if 'mask' in k else float) for k in
         ('mass','momentum_x','momentum_y','energy_left','energy_right','hllc_face_mask','thinc_face_mask')}
    for ell,f in fluxes.items():
        m=face_levels==ell
        for k in out: out[k][m]=f[k][m]
    return out


def _detect(Q,gammaS,e0S):
    bad=~np.isfinite(Q).all(axis=0)|(Q[0]<=0)
    V=None
    if not bad.any():
        try:
            V=update_primitives_df_2d(Q,gammaS,e0S)
            bad|=~np.isfinite(V).all(axis=0)|(V[3]<=0)
        except Exception:
            bad[:]=True
    return bad,V


def correct_euler_2d_local(V,Q,ds,dt,gammaS,e0S,direction,*,periodic,left_bc=None,right_bc=None,
                           use_thinc_bvd=True,zeta=.2,limit_ho=False,thinc_beta=1.6,
                           initial_cell_levels=None,max_iterations=None,**unused):
    levels=[3,2,1,0] if use_thinc_bvd else [2,1,0];top=levels[0];n=V.shape[1]
    fluxes={}
    for ell in levels:
        if periodic:
            fluxes[ell]=_periodic_flux_level(ell,V,gammaS,e0S,direction,zeta=zeta,limit_ho=limit_ho,use_thinc_bvd=use_thinc_bvd,thinc_beta=thinc_beta)
        else:
            fluxes[ell]=_nonperiodic_flux_level(ell,V,gammaS,e0S,direction,left_bc,right_bc,zeta=zeta,limit_ho=limit_ho,use_thinc_bvd=use_thinc_bvd,thinc_beta=thinc_beta)
    cell=np.full(n,top,dtype=np.int8) if initial_cell_levels is None else np.asarray(initial_cell_levels,dtype=np.int8).copy()
    if max_iterations is None:max_iterations=n*top+1
    hist=[]
    for it in range(max_iterations):
        face=_periodic_face_levels(cell) if periodic else face_levels_from_cells_nonperiodic(cell)
        mixed=_mix(fluxes,face)
        rhs=_assemble_rhs_periodic(mixed,ds) if periodic else _assemble_rhs_nonperiodic(mixed,ds)
        Qa=Q+dt*rhs; bad,Va=_detect(Qa,gammaS,e0S)
        hist.append(dict(iteration=it,n_troubled=int(bad.sum()),troubled_indices=np.flatnonzero(bad).tolist(),min_cell_level=int(cell.min())))
        if not bad.any():
            tc=fluxes.get(3,{}).get('thinc_cell_mask',np.zeros(n,bool)).copy(); tc&=cell==3
            return dict(Q_after=Qa,W_after=Qa,V_after=Va,cell_levels=cell.copy(),face_levels=face.copy(),
                        accepted_level=int(cell.min()),accepted_level_name=LEVEL_NAMES[int(cell.min())],
                        corrected_faces=face<top,mass_flux=mixed['mass'].copy(),
                        hllc_face_mask=mixed['hllc_face_mask'].copy(),thinc_face_mask=mixed['thinc_face_mask'].copy(),
                        thinc_cell_mask=tc,use_thinc_cell=tc,history=hist)
        if np.any(bad&(cell==0)): raise RuntimeError('etat non admissible au niveau terminal LLF 2D')
        cell[bad]-=1
    raise RuntimeError('MOOD 2D: nombre maximal iterations atteint')


def advance_ssprk3_directional_2d(V,Q,ds,dt,gammaS,e0S,direction,*,periodic,left_bc=None,right_bc=None,**options):
    r1=correct_euler_2d_local(V,Q,ds,dt,gammaS,e0S,direction,periodic=periodic,left_bc=left_bc,right_bc=right_bc,**options)
    Q1,V1=r1['Q_after'],r1['V_after']
    r2=correct_euler_2d_local(V1,Q1,ds,dt,gammaS,e0S,direction,periodic=periodic,left_bc=left_bc,right_bc=right_bc,**options)
    Q2=.75*Q+.25*r2['Q_after']; V2=update_primitives_df_2d(Q2,gammaS,e0S)
    r3=correct_euler_2d_local(V2,Q2,ds,dt,gammaS,e0S,direction,periodic=periodic,left_bc=left_bc,right_bc=right_bc,**options)
    Qn=(1/3)*Q+(2/3)*r3['Q_after']; Vn=update_primitives_df_2d(Qn,gammaS,e0S)
    return dict(Q_after=Qn,W_after=Qn,V_after=Vn,stage_results=(r1,r2,r3),
                intermediates={'Q1':Q1,'V1':V1,'Q2':Q2,'V2':V2},
                stage_min_levels=tuple(r['accepted_level'] for r in (r1,r2,r3)))
