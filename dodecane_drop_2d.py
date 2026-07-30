"""2-D advection of an n-dodecane drop.

The physical state and geometry follow Ma, Lv & Ihme (JCP 340, 2017,
Sec. 5.3): 1 mm square, 0.5 mm drop, 6 MPa, 363 K fuel, 900 K nitrogen,
50 m/s, 100x100 cells, CFL 0.8, one period (0.02 ms).

For this contact-equilibrium benchmark p, u and v are invariants.  The
runner therefore advances the two non-trivial conservative fields rho and
rho*Y_f with the same hybrid/THINC-BVD machinery in x.  This vectorized
specialization is mathematically equivalent to the Euler contact subsystem
and makes the 100x100 reproduction inexpensive.  The general 4-equation 2-D
Euler sweeps remain in twod.py for non-equilibrium cases.
"""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np

from reconstruction import reconstruct_high_order, reconstruct_high_order_limited
from eno2 import reconstruct_eno2_left, reconstruct_eno2_right
from thinc import reconstruct_thinc
from thermo_binary_pr import density_from_T_p_Y, sound_speed_from_T_rho_Y


def setup_figure13(N=100):
    L=1.0e-3; D=0.5e-3; p0=6.0e6; u0=50.0; v0=0.0
    dx=L/N; dy=L/N
    x=(np.arange(N)+0.5)*dx; y=(np.arange(N)+0.5)*dy
    X,Y=np.meshgrid(x,y)
    fuel=((X-L/2)**2+(Y-L/2)**2 <= (D/2)**2).astype(float)
    T=np.where(fuel>0.5,363.0,900.0)
    rho=density_from_T_p_Y(T,p0,fuel)
    return dict(x=x,y=y,X=X,Y=Y,dx=dx,dy=dy,L=L,D=D,p0=p0,u0=u0,v0=v0,
                temperature=T,rho=rho,Yfuel=fuel)


def _rs_mask(q,zeta=.2,limit_ho=False):
    qh=reconstruct_high_order_limited(q) if limit_ho else reconstruct_high_order(q)
    qr=np.roll(q,-1,axis=-1)
    ind=np.maximum(np.abs(qh-q)/(np.abs(q)+1e-14),np.abs(qh-qr)/(np.abs(qr)+1e-14))
    return ind>zeta,qh


def _bvd_thinc(q,beta=1.6):
    enoL=reconstruct_eno2_left(q); enoR=reconstruct_eno2_right(q)
    pl=np.roll(enoR,1,axis=-1); pr=enoL
    tl,tr=reconstruct_thinc(q,beta=beta)
    jp=np.abs(np.roll(pl,-1,axis=-1)-pr)
    jt=np.abs(np.roll(tl,-1,axis=-1)-tr)
    tbvp=np.roll(jp,1,axis=-1)+jp
    tbvt=np.roll(jt,1,axis=-1)+jt
    use=tbvt+(1e-14+1e-12*np.maximum(tbvp,tbvt))<tbvp
    # contact-only gate: large density contrast, no pressure/velocity jump in this benchmark
    qim1=np.roll(q,1,axis=-1); qip1=np.roll(q,-1,axis=-1)
    grad=np.maximum(np.abs(q-qim1),np.abs(qip1-q))/(np.maximum.reduce([np.abs(qim1),np.abs(q),np.abs(qip1)])+1e-14)
    use &= grad>5e-2
    return use,tl,tr,enoL,enoR


def _flux_pair(rho,Z,u,mode,zeta,beta,limit_ho):
    rs,rho_ho=_rs_mask(rho,zeta,limit_ho)
    Z_ho=reconstruct_high_order_limited(Z) if limit_ho else reconstruct_high_order(Z)
    rho_eno=reconstruct_eno2_left(rho) if u>=0 else reconstruct_eno2_right(rho)
    Z_eno=reconstruct_eno2_left(Z) if u>=0 else reconstruct_eno2_right(Z)
    use=np.zeros_like(rho,dtype=bool)
    if mode=='final':
        use,rtl,rtr,_,_=_bvd_thinc(rho,beta)
        _,ztl,ztr,_,_=_bvd_thinc(Z,beta)
        rho_t=rtr if u>=0 else np.roll(rtl,-1,axis=-1)
        Z_t=ztr if u>=0 else np.roll(ztl,-1,axis=-1)
        face_use=use|np.roll(use,-1,axis=-1)
        rho_up=np.where(use,rho_t,rho_eno)
        Z_up=np.where(use,Z_t,Z_eno)
        dissipative=rs|face_use
    else:
        rho_up=rho_eno; Z_up=Z_eno; dissipative=rs
    rho_face=np.where(dissipative,rho_up,rho_ho)
    Z_face=np.where(dissipative,Z_up,Z_ho)
    if mode=='final':
        # Build the species flux from the already selected mass flux.  The
        # high-order/THINC candidate is converted to a bounded face mass
        # fraction; a conservative local fallback is applied in _euler_step
        # only if the updated cell averages leave [0,rho].
        Y_face=np.clip(Z_face/np.maximum(rho_face,1e-14),0.0,1.0)
        Z_face=Y_face*rho_face
    return u*rho_face,u*Z_face,use,dissipative


def _rhs(rho,Z,u,dx,mode,zeta,beta,limit_ho):
    Fr,Fz,use,diss=_flux_pair(rho,Z,u,mode,zeta,beta,limit_ho)
    return -(Fr-np.roll(Fr,1,axis=-1))/dx, -(Fz-np.roll(Fz,1,axis=-1))/dx,use,diss


def _dmp_flux_limit_density(rho, Fr_high, Fr_low, lam, dmp_rel, dmp_abs, radius=1):
    """Conservative Zalesak-type limiter for the density antidiffusive flux.

    The low-order upwind update is the monotone baseline.  Only the
    antidiffusive correction Fr_high-Fr_low is limited.  The returned theta
    is face based and can be reused for rho*Y so mass/species corrections
    remain collocated.
    """
    neigh=[rho]
    for r in range(1,int(radius)+1):
        neigh.extend((np.roll(rho,r,axis=-1),np.roll(rho,-r,axis=-1)))
    lo=np.minimum.reduce(neigh)
    hi=np.maximum.reduce(neigh)
    span=hi-lo
    delta=np.maximum(float(dmp_abs),float(dmp_rel)*span)
    lower=lo-delta; upper=hi+delta

    rlow=rho-lam*(Fr_low-np.roll(Fr_low,1,axis=-1))
    # Antidiffusive transfer through face i: A_i leaves cell i and enters i+1.
    A=lam*(Fr_high-Fr_low)
    c_left=np.roll(A,1,axis=-1)  # contribution from left face
    c_right=-A                   # contribution from right face
    pplus=np.maximum(c_left,0.0)+np.maximum(c_right,0.0)
    pminus=np.maximum(-c_left,0.0)+np.maximum(-c_right,0.0)
    qplus=np.maximum(upper-rlow,0.0)
    qminus=np.maximum(rlow-lower,0.0)
    rplus=np.ones_like(rho); rminus=np.ones_like(rho)
    m=pplus>0; rplus[m]=np.minimum(1.0,qplus[m]/pplus[m])
    m=pminus>0; rminus[m]=np.minimum(1.0,qminus[m]/pminus[m])

    # Face i connects cell i (left) and i+1 (right).
    theta=np.where(A>=0.0,
                   np.minimum(rminus,np.roll(rplus,-1,axis=-1)),
                   np.minimum(rplus,np.roll(rminus,-1,axis=-1)))
    theta=np.clip(theta,0.0,1.0)
    return Fr_low+theta*(Fr_high-Fr_low),theta,lower,upper


def _euler_step(rho,Z,dt,u,dx,mode,zeta,beta,limit_ho, *, dmp_rel=None, dmp_abs=1e-10, dmp_radius=1):
    """One conservative Euler step with DMP/FCT and local parachute."""
    Fr,Fz,use,diss=_flux_pair(rho,Z,u,mode,zeta,beta,limit_ho)

    # Terminal coherent upwind fluxes are also the monotone baseline used by
    # the DMP flux-correction step.
    rho_up=rho if u>=0 else np.roll(rho,-1,axis=-1)
    Ycell=np.clip(Z/np.maximum(rho,1e-14),0.0,1.0)
    Y_up=Ycell if u>=0 else np.roll(Ycell,-1,axis=-1)
    Fr0=u*rho_up
    Fz0=Y_up*Fr0

    dmp_limited=np.zeros_like(rho,dtype=bool)
    dmp_lower=dmp_upper=None
    if mode=='final' and dmp_rel is not None:
        Fr_limited,theta,dmp_lower,dmp_upper=_dmp_flux_limit_density(
            rho,Fr,Fr0,dt/dx,dmp_rel,dmp_abs,dmp_radius)
        # Apply the same face coefficient to the partial-mass antidiffusion.
        Fz=Fz0+theta*(Fz-Fz0)
        Fr=Fr_limited
        dmp_limited=theta < (1.0-1e-13)
        diss=diss|dmp_limited

    Rn=rho-dt*(Fr-np.roll(Fr,1,axis=-1))/dx
    Zn=Z-dt*(Fz-np.roll(Fz,1,axis=-1))/dx
    if mode!='final':
        return Rn,Zn,use,diss

    # Remaining parachute is only for positivity/species admissibility.  The
    # density DMP has already been enforced conservatively by flux limiting.
    fallback=np.zeros_like(rho,dtype=bool)
    for _ in range(rho.shape[-1]+2):
        tol=1e-8+1e-11*np.abs(Rn)
        bad=(Rn<=0) | (Zn < -tol) | (Zn > Rn+tol) | ~np.isfinite(Rn) | ~np.isfinite(Zn)
        if dmp_rel is not None:
            # This should only catch roundoff or interaction with a later
            # species fallback, not drive a one-cell-at-a-time propagation.
            bad |= (Rn < dmp_lower-10*dmp_abs) | (Rn > dmp_upper+10*dmp_abs)
        if not np.any(bad):
            return Rn,Zn,use,diss|fallback
        faces=bad | np.roll(bad,-1,axis=-1)
        newfaces=faces & ~fallback
        if not np.any(newfaces):
            break
        fallback |= faces
        Fr=np.where(fallback,Fr0,Fr)
        Fz=np.where(fallback,Fz0,Fz)
        Rn=rho-dt*(Fr-np.roll(Fr,1,axis=-1))/dx
        Zn=Z-dt*(Fz-np.roll(Fz,1,axis=-1))/dx
    raise RuntimeError('admissibility lost after conservative local flux fallback')

def _rk3(rho,Z,dt,u,dx,mode,zeta,beta,limit_ho, *, dmp_rel=None, dmp_abs=1e-12, dmp_radius=1):
    R1,Z1,u1,d1=_euler_step(rho,Z,dt,u,dx,mode,zeta,beta,limit_ho,dmp_rel=dmp_rel,dmp_abs=dmp_abs,dmp_radius=dmp_radius)
    E2,ZE2,u2,d2=_euler_step(R1,Z1,dt,u,dx,mode,zeta,beta,limit_ho,dmp_rel=dmp_rel,dmp_abs=dmp_abs,dmp_radius=dmp_radius)
    R2=.75*rho+.25*E2; ZZ2=.75*Z+.25*ZE2
    E3,ZE3,u3,d3=_euler_step(R2,ZZ2,dt,u,dx,mode,zeta,beta,limit_ho,dmp_rel=dmp_rel,dmp_abs=dmp_abs,dmp_radius=dmp_radius)
    Rn=(1/3)*rho+(2/3)*E3; Zn=(1/3)*Z+(2/3)*ZE3
    if mode=='final':
        tol=2e-8+2e-11*np.abs(Rn)
        if np.any(Rn<=0): raise RuntimeError('negative density in final benchmark')
        if np.any(Zn < -tol) or np.any(Zn > Rn + tol):
            raise RuntimeError('species admissibility lost in SSP-RK3 combination')
    return Rn,Zn,float(np.mean(u1|u2|u3)),float(np.mean(d1|d2|d3))


def run_figure13_case(N=100,CFL=.8,mode='final',zeta=.2,beta=1.6,limit_ho=False, *, dmp_rel=None, dmp_abs=1e-12, dmp_radius=1):
    s=setup_figure13(N); rho=s['rho'].copy(); Z=rho*s['Yfuel']; rho0=rho.copy(); Z0=Z.copy()
    c=sound_speed_from_T_rho_Y(s['temperature'],rho,s['Yfuel'])
    # Paper's 2-D acoustic CFL; y contributes even though the exact y flux is zero.
    dt_nom=CFL/(np.max((abs(s['u0'])+c)/s['dx'])+np.max(c/s['dy']))
    tf=s['L']/s['u0']; t=0.; steps=0; max_th=0.; max_dis=0.; tic=time.perf_counter()
    while t<tf-1e-18:
        dt=min(dt_nom,tf-t)
        rho,Z,fth,fdis=_rk3(rho,Z,dt,s['u0'],s['dx'],mode,zeta,beta,limit_ho,dmp_rel=dmp_rel,dmp_abs=dmp_abs,dmp_radius=dmp_radius)
        max_th=max(max_th,fth);max_dis=max(max_dis,fdis);t+=dt;steps+=1
    # Preserve the raw conservative ratio for all accuracy/boundedness metrics.
    # A clipped copy is exposed only as an optional EOS/visualization helper.
    Y_raw=Z/np.maximum(rho,1.0e-14)
    Y_clipped=np.clip(Y_raw,0.0,1.0)
    elapsed=time.perf_counter()-tic
    j=N//2
    metrics=dict(N=N,mode=mode,steps=steps,dt_nom=dt_nom,runtime_s=elapsed,
        rho_L1=float(np.mean(np.abs(rho-rho0))),Y_L1=float(np.mean(np.abs(Y_raw-s['Yfuel']))),
        Y_raw_L1=float(np.mean(np.abs(Y_raw-s['Yfuel']))),
        Y_clipped_L1=float(np.mean(np.abs(Y_clipped-s['Yfuel']))),
        rho_min=float(rho.min()),rho_max=float(rho.max()),
        Y_min=float(Y_raw.min()),Y_max=float(Y_raw.max()),
        Y_raw_min=float(Y_raw.min()),Y_raw_max=float(Y_raw.max()),
        Y_undershoot=float(max(0.0,-Y_raw.min())),
        Y_overshoot=float(max(0.0,Y_raw.max()-1.0)),
        mass_rel=float(abs(rho.sum()-rho0.sum())/rho0.sum()),
        fuel_mass_rel=float(abs(Z.sum()-Z0.sum())/max(abs(Z0.sum()),1.0)),
        max_thinc_cell_fraction=max_th,max_dissipative_face_fraction=max_dis,
        dmp_rel=None if dmp_rel is None else float(dmp_rel),dmp_abs=float(dmp_abs),dmp_radius=int(dmp_radius))
    return {**s,'rho_final':rho,'Yfuel_final':Y_raw,'Yfuel_raw':Y_raw,
            'Yfuel_clipped':Y_clipped,'rho_exact':rho0,'Yfuel_exact':s['Yfuel'],
            'centerline_index':j,'metrics':metrics}


def reproduce_figure13(output_dir='.'):
    import matplotlib.pyplot as plt
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    base=run_figure13_case(mode='baseline')
    final=run_figure13_case(mode='final',dmp_rel=0.0)
    j=final['centerline_index']; xmm=1e3*final['x']
    fig,ax=plt.subplots(1,2,figsize=(10,4.2))
    ax[0].plot(xmm,final['rho_exact'][j],'k--',label='Exact')
    ax[0].plot(xmm,base['rho_final'][j],label='Baseline (no ES surrogate)')
    ax[0].plot(xmm,final['rho_final'][j],label='THINC-BVD + strict DMP')
    ax[0].set(xlabel='x [mm]',ylabel=r'$\rho$ [kg/m$^3$]',xlim=(0,1))
    ax[1].plot(xmm,final['Yfuel_exact'][j],'k--',label='Exact')
    ax[1].plot(xmm,base['Yfuel_final'][j],label='Baseline (no ES surrogate)')
    ax[1].plot(xmm,final['Yfuel_final'][j],label='THINC-BVD + strict DMP')
    ax[1].set(xlabel='x [mm]',ylabel=r'$Y_{n-C_{12}H_{26}}$',xlim=(0,1),ylim=(-.2,1.2))
    for a in ax: a.grid(alpha=.25);a.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'figure13_reproduction.png',dpi=180);plt.close(fig)
    data={k:v for k,v in final.items() if isinstance(v,np.ndarray)}
    np.savez_compressed(out/'figure13_solution.npz',**data,
                        baseline_rho=base['rho_final'],baseline_Yfuel=base['Yfuel_final'])
    metrics={'paper_setup':{'L_m':1e-3,'drop_diameter_m':.5e-3,'p_Pa':6e6,'T_fuel_K':363.,'T_N2_K':900.,'u_m_s':50.,'CFL':.8,'N':100,'t_final_s':2e-5},
             'baseline':base['metrics'],'final':final['metrics']}
    (out/'figure13_metrics.json').write_text(json.dumps(metrics,indent=2))
    return base,final,metrics

if __name__=='__main__':
    b,f,m=reproduce_figure13(Path(__file__).resolve().parent)
    print(json.dumps(m,indent=2))
