"""Wall-compatible multidirectional verification for the Strang 2-D solver.

A smooth, circular, low-amplitude pressure pulse starts from rest in the
centre of a periodic-x / slip-wall-y domain.  The final time is chosen before
the wave can reach the walls.  Isotropy is measured from the pressure
perturbation, not from the full density field.
"""
from __future__ import annotations
import numpy as np
from boundary import adiabatic_wall
from cfl2d import cfl_dt_2d
from strang import run_strang_2d, run_strang_periodic_2d
from thermo import get_rho_from_T_p, get_gammaS_e0S


def setup_acoustic_pulse(N=32, p0=5.0e6, T0=300.0, amplitude=2.0e-3,
                         sigma=0.06, center=(0.5,0.5)):
    dx=dy=1.0/N
    x=(np.arange(N)+0.5)*dx; y=(np.arange(N)+0.5)*dy
    X,Y=np.meshgrid(x,y,indexing='xy')
    r2=(X-center[0])**2+(Y-center[1])**2
    p=p0*(1.0+amplitude*np.exp(-r2/(2.0*sigma**2)))
    T=np.full_like(p,T0)
    rho=get_rho_from_T_p(T.ravel(),p.ravel()).reshape(N,N)
    V=np.stack([rho,np.zeros_like(rho),np.zeros_like(rho),p])
    return V,x,y,X,Y,dx,dy,p0


def pressure_moments(V,X,Y,p0):
    w=np.abs(V[3]-p0)
    total=float(w.sum())
    xc=float((w*X).sum()/total); yc=float((w*Y).sum()/total)
    vx=float((w*(X-xc)**2).sum()/total); vy=float((w*(Y-yc)**2).sum()/total)
    anis=abs(vx-vy)/max(vx,vy)
    return dict(xc=xc,yc=yc,var_x=vx,var_y=vy,anisotropy=anis,
                pressure_amplitude=float(w.max()))


def run_wall_acoustic_benchmark(N=32,CFL=0.5,t_final=None):
    V0,x,y,X,Y,dx,dy,p0=setup_acoustic_pulse(N=N)
    gamma,_=get_gammaS_e0S(np.vstack([V0[0].ravel(),V0[1].ravel(),V0[3].ravel()]))
    cmax=float(np.max(np.sqrt(gamma*V0[3].ravel()/V0[0].ravel())))
    wall_distance=0.5
    if t_final is None:
        # Keep the physical wavefront well away from the walls.
        t_final=0.12*wall_distance/cmax
    opts=dict(zeta=0.2,use_thinc_bvd=False,check_entropy=False)
    dt_fn=lambda V:cfl_dt_2d(V,dx,dy,CFL=CFL)
    wall=run_strang_2d(V0,dx,dy,t_final,dt_fn,adiabatic_wall(),
                       opts_x=opts,opts_y=opts,verbose=False)
    periodic=run_strang_periodic_2d(V0,dx,dy,t_final,dt_fn,
                                     opts_x=opts,opts_y=opts)
    mw=pressure_moments(wall['V2D'],X,Y,p0)
    mp=pressure_moments(periodic['V2D'],X,Y,p0)
    diff=float(np.max(np.abs(wall['V2D']-periodic['V2D'])))
    return dict(N=N,CFL=CFL,t_final=float(t_final),cmax=cmax,
                estimated_wave_distance=float(cmax*t_final),
                wall=wall,periodic=periodic,wall_metrics=mw,
                periodic_metrics=mp,max_state_difference=diff)
