"""
transverse.py
Etape 2D-1 : advection de la composante de vitesse TRANSVERSE a un sweep
directionnel (rotation des variables, recommandee en relecture avant de
demarrer le 2D).

Contexte : les solveurs 1D existants (main_hybrid.advance_ssprk3_df_step,
nonperiodic.advance_ssprk3_nonperiodic) ne connaissent que 3 variables
primitives (rho, u_normal, p) -- pas de place pour une deuxieme
composante de vitesse. Pour un sweep en x, la vitesse NORMALE est u et il
faut transporter v (transverse) separement ; pour un sweep en y, la
vitesse normale est v et il faut transporter u.

Principe (identique a species.py, SANS les contraintes specifiques aux
fractions massiques -- positivite, renormalisation, fermeture sum=1, qui
n'ont pas de sens pour une quantite de mouvement signee) :

    1. la quantite advectee de maniere UPWIND est l'intensif w = (rho*w)/rho
       (vitesse transverse, pas la quantite de mouvement elle-meme) ;
    2. le sens de l'upwind est donne par le signe du flux massique HYDRO
       deja accepte par le correcteur (MOOD/legacy) a ce sous-pas -- on ne
       resout PAS un probleme de Riemann separe pour w, on le transporte
       passivement avec la meme masse qui bouge deja ;
    3. la mise a jour est un simple Euler explicite couple au dt du
       sous-pas, a combiner ensuite avec les MEMES poids convexes SSP-RK3
       que l'hydrodynamique (0.75/0.25 puis 1/3/2/3).

Deux variantes selon la convention de face du sweep :
    - PERIODIQUE (sweep x) : mass_flux de forme (Nx,), face[i] = interface
      i+1/2, periodique (meme convention que hybrid_rhs.py).
    - NON PERIODIQUE (sweep y, murs) : mass_flux de forme (Nx+1,),
      face[j] = face GAUCHE de la cellule j (j=0..Nx-1), face[Nx] = face
      droite du domaine (meme convention que nonperiodic.assemble_rhs_nonperiodic).
      A un mur adiabatique, le flux massique a la face limite est ~0 par
      construction (reflexion), donc le sens de l'upwind y est sans
      consequence : la formule degenere proprement, aucun cas particulier
      necessaire.
"""

import numpy as np


def advance_transverse_periodic_euler(rhow_before, rho_before, mass_flux, dx, dt):
    """Un pas Euler explicite pour rho*w, convention de face PERIODIQUE.

    rhow_before, rho_before : (Nx,)
    mass_flux : (Nx,) -- face[i] = interface i+1/2 (periodique)
    Retourne (rhow_after, flux) ou flux a la forme (Nx,).
    """
    rhow_before = np.asarray(rhow_before, dtype=float)
    rho_before = np.asarray(rho_before, dtype=float)
    mass_flux = np.asarray(mass_flux, dtype=float)
    if rhow_before.shape != rho_before.shape or rhow_before.shape != mass_flux.shape:
        raise ValueError("rhow_before, rho_before et mass_flux doivent avoir la meme forme (Nx,)")
    if np.any(rho_before <= 0.0) or not np.all(np.isfinite(rho_before)):
        raise ValueError("rho_before doit etre fini et strictement positif")

    w = rhow_before / rho_before
    w_right = np.roll(w, -1)
    w_adv = np.where(mass_flux >= 0.0, w, w_right)
    flux = w_adv * mass_flux

    rhs = -(flux - np.roll(flux, 1)) / dx
    rhow_after = rhow_before + dt * rhs

    if not np.all(np.isfinite(rhow_after)):
        raise ValueError("mise a jour transverse (periodique) non finie")
    return rhow_after, flux


def advance_transverse_nonperiodic_euler(rhow_before, rho_before, mass_flux, dx, dt):
    """Un pas Euler explicite pour rho*w, convention de face NON PERIODIQUE
    (murs). rhow_before, rho_before : (Nx,). mass_flux : (Nx+1,) --
    face[j] = face gauche de la cellule j, face[Nx] = face droite du
    domaine (meme convention que nonperiodic.assemble_rhs_nonperiodic).

    A un mur, mass_flux[0] et mass_flux[Nx] valent ~0 par construction
    (reflexion adiabatique) -- le sens de l'upwind y est donc sans
    consequence, aucun traitement special necessaire.
    """
    rhow_before = np.asarray(rhow_before, dtype=float)
    rho_before = np.asarray(rho_before, dtype=float)
    mass_flux = np.asarray(mass_flux, dtype=float)
    nx = rhow_before.shape[0]
    if rho_before.shape != (nx,) or mass_flux.shape != (nx + 1,):
        raise ValueError("formes incompatibles : rhow/rho (Nx,), mass_flux (Nx+1,)")
    if np.any(rho_before <= 0.0) or not np.all(np.isfinite(rho_before)):
        raise ValueError("rho_before doit etre fini et strictement positif")

    w = rhow_before / rho_before  # (Nx,)
    # etats gauche/droite a CHAQUE face (Nx+1 faces), extrapolation aux
    # deux bords du domaine (pas de contrainte de reflexion sur la
    # composante TRANSVERSE -- seule la composante normale est reflechie
    # par le mur, cf. boundary.adiabatic_wall)
    w_left = np.concatenate([[w[0]], w])     # (Nx+1,)
    w_right = np.concatenate([w, [w[-1]]])   # (Nx+1,)
    w_adv = np.where(mass_flux >= 0.0, w_left, w_right)
    flux = w_adv * mass_flux

    rhs = -(flux[1:] - flux[:-1]) / dx
    rhow_after = rhow_before + dt * rhs

    if not np.all(np.isfinite(rhow_after)):
        raise ValueError("mise a jour transverse (non periodique) non finie")
    return rhow_after, flux
