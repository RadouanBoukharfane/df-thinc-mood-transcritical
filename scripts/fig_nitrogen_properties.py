"""Figure 1: density and gamma* = rho c^2 / p of nitrogen along supercritical isobars (CoolProp, PR)."""
import CoolProp.CoolProp as CP
import numpy as np

import paperstyle as ps

RATIOS = (1.1, 1.3, 1.5, 2.0)


def isobar(state, p, T):
    rho, gam = np.full_like(T, np.nan), np.full_like(T, np.nan)
    for i, t in enumerate(T):
        try:
            state.update(CP.PT_INPUTS, p, t)
            rho[i] = state.rhomass()
            gam[i] = rho[i] * state.speed_sound() ** 2 / p
        except ValueError:
            pass
    return rho, gam


if __name__ == "__main__":
    ps.apply()
    state = CP.AbstractState("PR", "Nitrogen")
    Tc, pc = state.T_critical(), state.p_critical()
    T = np.linspace(110.0, 260.0, 600)
    fig, (a, b) = ps.figure(1, 2, height=1.75)
    for r, col in zip(RATIOS, ps.ordinal(len(RATIOS))):
        rho, gam = isobar(state, r * pc, T)
        a.plot(T, rho, color=col, label=rf"${r:.1f}\,p_c$")
        b.plot(T, gam, color=col)
    for ax in (a, b):
        ax.axvline(Tc, color=ps.MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.set_xlim(T[0], T[-1])
        ax.set_xlabel(r"$T$ [K]")
    a.text(Tc + 2, 40, r"$T_c$", color=ps.MUTED, fontsize=7, va="bottom")
    a.set_ylim(bottom=0)
    a.set_ylabel(r"$\rho$ [kg\,m$^{-3}$]")
    b.set_ylabel(r"$\gamma^\star=\rho c^2/p$")
    ps.panel_label(a, "a")
    ps.panel_label(b, "b")
    a.legend(loc="upper right", title="Isobar", title_fontsize=7)
    ps.save(fig, "nitrogen_transcritical_properties")
