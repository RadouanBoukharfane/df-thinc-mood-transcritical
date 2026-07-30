"""Binary Peng--Robinson thermodynamics for N2 / n-dodecane.

This module implements the binary mixing rules reported in Appendix A of
Ma, Lv & Ihme (JCP 340, 2017).  It is intentionally scoped to the
Figure-13 advection benchmark.  Ideal-gas caloric properties of nitrogen use
the existing NASA fit; n-dodecane uses a smooth engineering cp correlation
because the current repository does not contain the paper's full species
thermochemistry database.  Density and pressure, which dominate this pure
advection benchmark, use the PR mixture relation and Harstad corresponding-
states mixing rules.
"""
from __future__ import annotations
import numpy as np

RU = 8.31446261815324

# Order: nitrogen, n-dodecane
MW = np.array([28.0134e-3, 170.33484e-3])
TC = np.array([126.19, 658.10])
PC = np.array([3.3958e6, 1.8170e6])
OMEGA = np.array([0.03720, 0.5760])
RHOC = np.array([313.3, 226.6])
VC = MW / RHOC
ZC = PC * VC / (RU * TC)
K_BINARY = np.zeros((2, 2))

# Existing low-temperature N2 NASA coefficients, Cp/R.
NASA_N2 = np.array([
    3.531005280e+00, -1.236609870e-04, -5.029994370e-07,
    2.435306120e-09, -1.408812350e-12, -1.046976280e+03,
    2.967474680e+00,
])
T_REF = 298.15


def mass_to_mole_fraction(Y_fuel):
    """Return X_N2, X_fuel and mixture molecular weight."""
    Yf = np.clip(np.asarray(Y_fuel, dtype=float), 0.0, 1.0)
    Y = np.stack([1.0 - Yf, Yf], axis=0)
    n = Y / MW.reshape((2,) + (1,) * Yf.ndim)
    denom = np.sum(n, axis=0)
    X = n / denom
    MWmix = 1.0 / denom
    return X[0], X[1], MWmix


def _pair_constants():
    Tcij = np.empty((2, 2)); pcij = np.empty((2, 2)); omegaij = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            Tcij[i,j] = np.sqrt(TC[i] * TC[j]) * (1.0 - K_BINARY[i,j])
            vcij = 0.125 * (VC[i] ** (1/3) + VC[j] ** (1/3)) ** 3
            zcij = 0.5 * (ZC[i] + ZC[j])
            pcij[i,j] = zcij * RU * Tcij[i,j] / vcij
            omegaij[i,j] = 0.5 * (OMEGA[i] + OMEGA[j])
    return Tcij, pcij, omegaij

TCIJ, PCIJ, OMEGAIJ = _pair_constants()
CIJ = 0.37464 + 1.54226 * OMEGAIJ - 0.26992 * OMEGAIJ**2
BI = 0.077796 * RU * TC / PC


def mixture_parameters(T, Y_fuel):
    T = np.asarray(T, dtype=float)
    Yf = np.asarray(Y_fuel, dtype=float)
    T, Yf = np.broadcast_arrays(T, Yf)
    Xn, Xf, MWmix = mass_to_mole_fraction(Yf)
    X = np.stack([Xn, Xf], axis=0)

    a = np.zeros_like(T); dadT = np.zeros_like(T); d2adT2 = np.zeros_like(T)
    for i in range(2):
        for j in range(2):
            tc = TCIJ[i,j]; pc = PCIJ[i,j]; c = CIJ[i,j]
            root = np.sqrt(T / tc)
            phi = 1.0 + c * (1.0 - root)
            aij = 0.457236 * (RU * tc)**2 / pc * phi**2
            G = c * root / phi
            daij = -(aij / T) * G
            d2aij = 0.457236 * RU**2 / (2.0*T) * c*(1.0+c)*tc/pc*np.sqrt(tc/T)
            w = X[i] * X[j]
            a += w*aij; dadT += w*daij; d2adT2 += w*d2aij
    b = Xn*BI[0] + Xf*BI[1]
    return a, b, dadT, d2adT2, MWmix, X


def pressure_from_T_rho_Y(T, rho, Y_fuel):
    T, rho, Yf = np.broadcast_arrays(np.asarray(T,float), np.asarray(rho,float), np.asarray(Y_fuel,float))
    a,b,_,_,MWmix,_ = mixture_parameters(T,Yf)
    Vm = MWmix / rho
    return RU*T/(Vm-b) - a/(Vm**2 + 2*b*Vm - b**2)


def _cp0_molar(T, X):
    T = np.asarray(T,float)
    an = NASA_N2
    cp_n2 = RU*(an[0]+an[1]*T+an[2]*T**2+an[3]*T**3+an[4]*T**4)
    # Smooth vapor-reference correlation; adequate for the inert advection
    # benchmark and kept positive over 250--1200 K.
    cp_c12 = np.clip(220.0 + 0.34*(T-300.0), 180.0, 500.0)
    return X[0]*cp_n2 + X[1]*cp_c12


def _u0_molar(T, X):
    T = np.asarray(T,float)
    an = NASA_N2
    h_n2 = RU*T*(an[0]+an[1]*T/2+an[2]*T**2/3+an[3]*T**3/4+an[4]*T**4/5+an[5]/T)
    u_n2 = h_n2 - RU*T
    dT = T-T_REF
    # Integral of cp_c12 - R with the unclipped correlation in the range used.
    u_c12 = (220.0-RU)*dT + 0.17*((T-300.0)**2-(T_REF-300.0)**2)
    return X[0]*u_n2 + X[1]*u_c12


def internal_energy_from_T_rho_Y(T,rho,Y_fuel):
    T,rho,Yf=np.broadcast_arrays(np.asarray(T,float),np.asarray(rho,float),np.asarray(Y_fuel,float))
    a,b,dadT,_,MWmix,X=mixture_parameters(T,Yf)
    Vm=MWmix/rho
    K1=np.log((Vm+(1-np.sqrt(2))*b)/(Vm+(1+np.sqrt(2))*b))/(np.sqrt(8)*b)
    return (_u0_molar(T,X)+K1*(a-T*dadT))/MWmix


def temperature_from_p_rho_Y(p,rho,Y_fuel, *, max_iter=80):
    p,rho,Yf=np.broadcast_arrays(np.asarray(p,float),np.asarray(rho,float),np.asarray(Y_fuel,float))
    T=np.where(Yf>0.5,363.0,900.0).astype(float)
    for _ in range(max_iter):
        a,b,dadT,_,MWmix,_=mixture_parameters(T,Yf)
        Vm=MWmix/rho
        pn=RU*T/(Vm-b)-a/(Vm**2+2*b*Vm-b**2)
        diff=pn-p
        if np.max(np.abs(diff)/np.maximum(p,1.0))<1e-11: return T
        dpdT=RU/(Vm-b)-dadT/(Vm**2+2*b*Vm-b**2)
        step=np.clip(diff/dpdT,-0.25*T,0.25*T)
        T=np.maximum(80.0,T-step)
    raise RuntimeError('temperature_from_p_rho_Y did not converge')


def sound_speed_from_T_rho_Y(T,rho,Y_fuel):
    T,rho,Yf=np.broadcast_arrays(np.asarray(T,float),np.asarray(rho,float),np.asarray(Y_fuel,float))
    a,b,dadT,d2adT2,MWmix,X=mixture_parameters(T,Yf)
    Vm=MWmix/rho
    den=Vm**2+2*b*Vm-b**2
    dpdT=RU/(Vm-b)-dadT/den
    dpdv=-RU*T/(Vm-b)**2 + 2*a*(Vm+b)/den**2
    K1=np.log((Vm+(1-np.sqrt(2))*b)/(Vm+(1+np.sqrt(2))*b))/(np.sqrt(8)*b)
    cv0=_cp0_molar(T,X)-RU
    cv=cv0-K1*T*d2adT2
    cp=cv0-K1*T*d2adT2-T*dpdT**2/dpdv
    kT=-1.0/(Vm*dpdv)
    alpha=-dpdT/(Vm*dpdv)
    ks=kT-Vm*T*alpha**2/cp
    c2=1.0/(rho*ks)
    if np.any(c2<=0) or np.any(~np.isfinite(c2)):
        raise RuntimeError('non-hyperbolic binary PR state')
    return np.sqrt(c2)


def gamma_e0_from_rho_p_Y(rho,p,Y_fuel):
    T=temperature_from_p_rho_Y(p,rho,Y_fuel)
    e=internal_energy_from_T_rho_Y(T,rho,Y_fuel)
    c=sound_speed_from_T_rho_Y(T,rho,Y_fuel)
    gamma=c*c*rho/p
    e0=e-p/rho/(gamma-1.0)
    return gamma,e0,T


def density_from_T_p_Y(T,p,Y_fuel):
    """Robust PR cubic roots, choosing the stable root nearest the phase."""
    T,p,Yf=np.broadcast_arrays(np.asarray(T,float),np.asarray(p,float),np.asarray(Y_fuel,float))
    out=np.empty_like(T)
    it=np.nditer([T,p,Yf,out],op_flags=[['readonly']]*3+[['writeonly']])
    for tv,pv,yv,ov in it:
        a,b,_,_,MWmix,_=mixture_parameters(float(tv),float(yv))
        a=float(a);b=float(b);mw=float(MWmix);tt=float(tv);pp=float(pv)
        coeff=[pp,pp*b-RU*tt,-3*pp*b*b-2*RU*tt*b+a,pp*b**3+RU*tt*b*b-a*b]
        roots=np.roots(coeff)
        real=np.real(roots[np.abs(np.imag(roots))<1e-8])
        real=real[real>b*(1+1e-10)]
        if real.size==0: raise RuntimeError('no physical PR root')
        Vm=np.min(real) if float(yv)>0.5 else np.max(real)
        ov[...] = mw/Vm
    return out
