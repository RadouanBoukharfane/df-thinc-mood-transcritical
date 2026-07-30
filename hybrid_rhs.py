"""
hybrid_rhs.py
Etape 3C : assemblage du RHS hybride, combinant F^HO et F^ENO-HLLC selon
le capteur RS -- fully-conservative (FC) et double-flux (DF).

Etape 3E : refactorise pour exposer les FLUX DE FACE BRUTS (pas seulement
le RHS deja assemble), reutilises par corrector.py pour remplacer
localement certaines faces par F^ENO-LF sans recalculer toute la
reconstruction.

Investigation etalement d'interface : reconstruct_high_order_limited
(piste 1, validee -- reduit un vrai depassement physique residuel sans
degrader la largeur ni la precision en zone lisse) est disponible via le
parametre EXPLICITE `limit_ho` (par defaut False), plutot que substituee
silencieusement partout. Raison (suite a relecture) : l'ecretage n'est
PAS specifie dans l'algorithme du papier -- ce n'est pas juste un detail
d'implementation du schema central d'ordre 4 qu'ils decrivent, c'est une
modification additionnelle. Pour reproduire fidelement la methode
d'origine, laisser limit_ho=False (defaut) ; pour la variante plus
robuste (recommandee suite a l'investigation), passer limit_ho=True
explicitement.

Reconstruction : reconstruction.reconstruct_high_order(_limited) et
eno2.reconstruct_eno2_left/right operent NATIVEMENT sur des tableaux
(Nvars, Nx) (axis=-1), donc appeles directement sur V=(rho,u,p) entier,
sans boucle manuelle sur les lignes.

Convention d'indexation des faces (coherente avec get_rhs_df, etape 1) :
    face[i] = face i+1/2, entre cellule i et cellule i+1.
    Pour la cellule i :
        - sa face DROITE est face[i]     (elle est proprietaire GAUCHE)
        - sa face GAUCHE est face[i-1]   (elle est proprietaire DROIT)
"""

import numpy as np
from dataclasses import dataclass

from reconstruction import reconstruct_high_order, reconstruct_high_order_limited
from eno2 import reconstruct_eno2_left, reconstruct_eno2_right
from thinc import reconstruct_thinc
from bvd import eno_face_states_to_cell_boundaries, compare_bvd_candidates
from sensor import rs_face_mask_direct, rs_face_mask_from_cells, classify_contact_shock
from face_operators import flux_ho, flux_hllc_df, flux_ho_fc, flux_hllc_fc


@dataclass(frozen=True)
class ReconstructionCandidates:
    """Candidats de reconstruction calcules a partir du meme etat cellulaire.

    Cette structure est volontairement diagnostique a l'etape 2 :
    ``hybrid_rhs`` continue d'utiliser les champs historiques HO/ENO. Les
    champs THINC sont seulement exposes ; le critere BVD peut etre evalue en
    diagnostic, sans modifier les flux.

    Conventions
    -----------
    ``V_face_ho[..., i]`` est l'etat centre a la face i+1/2.
    ``V_L_eno[..., i]`` et ``V_R_eno[..., i]`` sont les etats gauche/droit
    ENO a la face i+1/2, selon la convention historique du solveur.
    ``rho_thinc_left_cell[i]`` est la valeur issue de la cellule i a sa face
    gauche i-1/2 ; ``rho_thinc_right_cell[i]`` celle issue de i a sa face
    droite i+1/2. Les etats THINC a la face i+1/2 sont donc obtenus via les
    proprietes ``rho_L_thinc_face`` et ``rho_R_thinc_face``.
    """

    V_face_ho: np.ndarray
    V_L_eno: np.ndarray
    V_R_eno: np.ndarray
    rho_thinc_left_cell: np.ndarray
    rho_thinc_right_cell: np.ndarray
    limited_mask: object

    @property
    def rho_L_thinc_face(self):
        """Etat gauche THINC a chaque face i+1/2, issu de la cellule i."""
        return self.rho_thinc_right_cell

    @property
    def rho_R_thinc_face(self):
        """Etat droit THINC a chaque face i+1/2, issu de la cellule i+1."""
        return np.roll(self.rho_thinc_left_cell, -1)

    @property
    def rho_eno_left_cell(self):
        """Trace ENO issue de chaque cellule a sa face gauche i-1/2."""
        q_left_cell, _ = eno_face_states_to_cell_boundaries(
            self.V_L_eno[0, :], self.V_R_eno[0, :]
        )
        return q_left_cell

    @property
    def rho_eno_right_cell(self):
        """Trace ENO issue de chaque cellule a sa face droite i+1/2."""
        _, q_right_cell = eno_face_states_to_cell_boundaries(
            self.V_L_eno[0, :], self.V_R_eno[0, :]
        )
        return q_right_cell


def reconstruct_candidates(V, limit_ho=False, thinc_beta=1.6, thinc_eps=1.0e-14):
    """Construit en parallele les candidats historiques et le candidat THINC.

    Parameters
    ----------
    V : ndarray, shape (3, Nx)
        Variables primitives ``(rho, u, p)``.
    limit_ho : bool
        Active la variante limitee de la reconstruction centree historique.
    thinc_beta, thinc_eps : float
        Parametres transmis a :func:`thinc.reconstruct_thinc`.

    Returns
    -------
    ReconstructionCandidates
        Les candidats HO, ENO2 et THINC-densite. Aucun choix BVD n'est fait
        ici et les flux du solveur ne sont pas modifies a cette etape.
    """
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] < 3:
        raise ValueError(
            "V doit etre un tableau 2D de forme (Nvars, Nx), avec au moins "
            "les lignes (rho, u, p)"
        )

    if limit_ho:
        V_face_centered = reconstruct_high_order(V)
        V_face_ho = reconstruct_high_order_limited(V)
        limited_mask = np.any(np.abs(V_face_ho - V_face_centered) > 1e-14, axis=0)
    else:
        V_face_ho = reconstruct_high_order(V)
        limited_mask = None

    V_L_eno = reconstruct_eno2_left(V)
    V_R_eno = reconstruct_eno2_right(V)

    rho_thinc_left_cell, rho_thinc_right_cell = reconstruct_thinc(
        V[0, :], beta=thinc_beta, eps=thinc_eps
    )

    return ReconstructionCandidates(
        V_face_ho=V_face_ho,
        V_L_eno=V_L_eno,
        V_R_eno=V_R_eno,
        rho_thinc_left_cell=rho_thinc_left_cell,
        rho_thinc_right_cell=rho_thinc_right_cell,
        limited_mask=limited_mask,
    )


def diagnose_density_bvd(
    V,
    limit_ho=False,
    thinc_beta=1.6,
    thinc_eps=1.0e-14,
    bvd_atol=1.0e-14,
    bvd_rtol=1.0e-12,
):
    """Compare ENO2 et THINC pour la densite, sans modifier les flux.

    Le candidat centre d'ordre 4 n'est pas utilise dans le TBV car il fournit
    une valeur unique par face et conduirait donc a un saut gauche/droit nul
    par construction. Cette fonction est exclusivement diagnostique.
    """
    candidates = reconstruct_candidates(
        V,
        limit_ho=limit_ho,
        thinc_beta=thinc_beta,
        thinc_eps=thinc_eps,
    )
    diagnostics = compare_bvd_candidates(
        candidates.rho_eno_left_cell,
        candidates.rho_eno_right_cell,
        candidates.rho_thinc_left_cell,
        candidates.rho_thinc_right_cell,
        atol=bvd_atol,
        rtol=bvd_rtol,
    )
    return candidates, diagnostics


def diagnose_density_bvd_with_wave_sensor(
    V,
    limit_ho=False,
    thinc_beta=1.6,
    thinc_eps=1.0e-14,
    bvd_atol=1.0e-14,
    bvd_rtol=1.0e-12,
    **sensor_options,
):
    """Diagnostic combine BVD et classification contact--choc.

    THINC est seulement *autorise* dans une cellule si :

        1. BVD prefere strictement THINC a ENO2 pour la densite ;
        2. le detecteur classe la cellule comme contact ;
        3. la cellule n'est pas adjacente a un choc.

    Aucun masque retourne ici n'est encore injecte dans les flux.
    """
    candidates, bvd = diagnose_density_bvd(
        V,
        limit_ho=limit_ho,
        thinc_beta=thinc_beta,
        thinc_eps=thinc_eps,
        bvd_atol=bvd_atol,
        bvd_rtol=bvd_rtol,
    )
    waves = classify_contact_shock(V, **sensor_options)
    use_thinc_cell = bvd.use_thinc_cell & waves.allow_thinc_cell
    return candidates, bvd, waves, use_thinc_cell



def assemble_density_bvd_face_states(candidates, use_thinc_cell):
    """Assemble les etats de Riemann avec THINC uniquement sur la densite.

    Le choix BVD est cellulaire. Pour la cellule ``i`` selectionnee THINC,
    ses deux traces de densite sont remplacees par les traces THINC. Les
    traces de vitesse et de pression restent celles de la reconstruction
    ENO2 historique. A la face ``i+1/2`` :

    * l'etat gauche provient de la face droite de la cellule ``i`` ;
    * l'etat droit provient de la face gauche de la cellule ``i+1``.

    Returns
    -------
    V_L, V_R : ndarray, shape (Nvars, Nx)
        Etats gauche/droit a transmettre au solveur de Riemann.
    thinc_face : ndarray bool, shape (Nx,)
        Face marquee si au moins une de ses deux cellules utilise THINC.
    """
    use_thinc_cell = np.asarray(use_thinc_cell, dtype=bool)
    nx = candidates.V_L_eno.shape[1]
    if use_thinc_cell.shape != (nx,):
        raise ValueError("use_thinc_cell doit etre de forme (Nx,)")

    rho_left_cell = np.where(
        use_thinc_cell,
        candidates.rho_thinc_left_cell,
        candidates.rho_eno_left_cell,
    )
    rho_right_cell = np.where(
        use_thinc_cell,
        candidates.rho_thinc_right_cell,
        candidates.rho_eno_right_cell,
    )

    V_L = candidates.V_L_eno.copy()
    V_R = candidates.V_R_eno.copy()
    V_L[0, :] = rho_right_cell
    V_R[0, :] = np.roll(rho_left_cell, -1)

    thinc_face = use_thinc_cell | np.roll(use_thinc_cell, -1)
    return V_L, V_R, thinc_face

def _reconstruct_all(V, limit_ho=False):
    """Compatibilite historique : retourne uniquement HO, ENO2 et masque.

    Le candidat THINC est bien calcule par :func:`reconstruct_candidates`,
    mais il n'est pas encore injecte dans les flux. Le format de retour reste
    strictement identique afin de ne modifier ni ``corrector.py`` ni les tests
    historiques pendant cette etape de developpement.
    """
    candidates = reconstruct_candidates(V, limit_ho=limit_ho)
    return (
        candidates.V_face_ho,
        candidates.V_L_eno,
        candidates.V_R_eno,
        candidates.limited_mask,
    )


def _assemble_rhs_fc(mass, momentum, energy, dx):
    mass_R, mass_L = mass, np.roll(mass, 1)
    momentum_R, momentum_L = momentum, np.roll(momentum, 1)
    energy_R, energy_L = energy, np.roll(energy, 1)
    Nx = len(mass)
    RHS = np.zeros((3, Nx))
    RHS[0, :] = -(mass_R - mass_L) / dx
    RHS[1, :] = -(momentum_R - momentum_L) / dx
    RHS[2, :] = -(energy_R - energy_L) / dx
    return RHS


def _assemble_rhs_df(mass, momentum, energy_left, energy_right, dx):
    mass_R, mass_L = mass, np.roll(mass, 1)
    momentum_R, momentum_L = momentum, np.roll(momentum, 1)
    energy_R = energy_left
    energy_L = np.roll(energy_right, 1)
    Nx = len(mass)
    RHS = np.zeros((3, Nx))
    RHS[0, :] = -(mass_R - mass_L) / dx
    RHS[1, :] = -(momentum_R - momentum_L) / dx
    RHS[2, :] = -(energy_R - energy_L) / dx
    return RHS


def get_hybrid_face_fluxes_fc(V, dx, zeta=0.2, mask_variant='direct', limit_ho=False):
    """Flux de face bruts, mode FULLY-CONSERVATIVE : (mass, momentum,
    energy, mask, V_L_eno, V_R_eno, limited_mask). Un seul flux d'energie (FC).

    limit_ho : si True, utilise reconstruct_high_order_limited (piste 1
        de l'investigation etalement, variante non specifiee dans le
        papier -- cf. docstring de module). Defaut False = methode
        d'origine, fidele au papier.

    IMPORTANT : chaque operateur (F^HO, F^ENO-HLLC) n'est evalue QUE sur
    les faces qui l'utilisent reellement (masquage booleen), PAS partout
    puis filtre a la fin -- evite d'appeler la thermodynamique reelle sur
    un etat HO potentiellement invalide (densite negative observee pres
    d'un saut abrupt, cf. etape 3C).
    """
    rho = V[0, :]
    Nx = V.shape[1]
    V_face_ho, V_L_eno, V_R_eno, limited_mask = _reconstruct_all(V, limit_ho=limit_ho)

    if mask_variant == 'direct':
        mask, _ = rs_face_mask_direct(rho, zeta)
    else:
        mask = rs_face_mask_from_cells(rho, zeta)

    idx_ho = ~mask
    idx_eno = mask

    mass = np.empty(Nx)
    momentum = np.empty(Nx)
    energy = np.empty(Nx)

    if idx_ho.any():
        ff_ho = flux_ho_fc(V_face_ho[:, idx_ho])
        mass[idx_ho] = ff_ho.mass
        momentum[idx_ho] = ff_ho.momentum
        energy[idx_ho] = ff_ho.energy_left_owner  # == energy_right_owner en FC

    if idx_eno.any():
        ff_eno = flux_hllc_fc(V_L_eno[:, idx_eno], V_R_eno[:, idx_eno])
        mass[idx_eno] = ff_eno.mass
        momentum[idx_eno] = ff_eno.momentum
        energy[idx_eno] = ff_eno.energy_left_owner

    return mass, momentum, energy, mask, V_L_eno, V_R_eno, limited_mask


def get_hybrid_face_fluxes_df(
    V,
    gammaS_frozen,
    e0S_frozen,
    dx,
    zeta=0.2,
    mask_variant='direct',
    limit_ho=False,
    use_thinc_bvd=False,
    thinc_beta=1.6,
    thinc_eps=1.0e-14,
    bvd_atol=1.0e-14,
    bvd_rtol=1.0e-12,
    sensor_options=None,
):
    """Flux de face bruts, mode DOUBLE-FLUX : (mass, momentum, energy_left,
    energy_right, mask, V_L_eno, V_R_eno, gammaS_left, e0S_left,
    gammaS_right, e0S_right, limited_mask). Deux flux d'energie distincts
    (DF, Eq. 21).

    limit_ho : voir get_hybrid_face_fluxes_fc. Defaut False = methode
        d'origine, fidele au papier.

    gammaS_frozen/e0S_frozen geles sur tout le pas de temps complet
    (fournis en argument, PAS recalcules ici). Le capteur RS, lui, EST
    recalcule ici sur l'etat courant V (cf. sensor.py).
    """
    rho = V[0, :]
    Nx = V.shape[1]
    candidates = reconstruct_candidates(
        V,
        limit_ho=limit_ho,
        thinc_beta=thinc_beta,
        thinc_eps=thinc_eps,
    )
    V_face_ho = candidates.V_face_ho
    V_L_eno = candidates.V_L_eno
    V_R_eno = candidates.V_R_eno
    limited_mask = candidates.limited_mask

    if mask_variant == 'direct':
        mask_rs, _ = rs_face_mask_direct(rho, zeta)
    else:
        mask_rs = rs_face_mask_from_cells(rho, zeta)

    use_thinc_cell = np.zeros(Nx, dtype=bool)
    thinc_face_mask = np.zeros(Nx, dtype=bool)
    V_L_riemann = V_L_eno
    V_R_riemann = V_R_eno

    if use_thinc_bvd:
        bvd = compare_bvd_candidates(
            candidates.rho_eno_left_cell,
            candidates.rho_eno_right_cell,
            candidates.rho_thinc_left_cell,
            candidates.rho_thinc_right_cell,
            atol=bvd_atol,
            rtol=bvd_rtol,
        )
        waves = classify_contact_shock(V, **(sensor_options or {}))
        use_thinc_cell = bvd.use_thinc_cell & waves.allow_thinc_cell
        V_L_riemann, V_R_riemann, thinc_face_mask = assemble_density_bvd_face_states(
            candidates, use_thinc_cell
        )

    # Toute face portant au moins une trace THINC doit passer par un solveur
    # de Riemann a deux etats. Les autres faces gardent exactement le chemin
    # historique RS : HO central ou ENO2-HLLC.
    mask = mask_rs | thinc_face_mask

    gammaS_left = gammaS_frozen
    e0S_left = e0S_frozen
    gammaS_right = np.roll(gammaS_frozen, -1)
    e0S_right = np.roll(e0S_frozen, -1)

    idx_ho = ~mask
    idx_eno = mask

    mass = np.empty(Nx)
    momentum = np.empty(Nx)
    energy_left = np.empty(Nx)
    energy_right = np.empty(Nx)

    if idx_ho.any():
        ff_ho = flux_ho(V_face_ho[:, idx_ho], gammaS_left[idx_ho], e0S_left[idx_ho],
                         gammaS_right[idx_ho], e0S_right[idx_ho])
        mass[idx_ho] = ff_ho.mass
        momentum[idx_ho] = ff_ho.momentum
        energy_left[idx_ho] = ff_ho.energy_left_owner
        energy_right[idx_ho] = ff_ho.energy_right_owner

    if idx_eno.any():
        ff_eno = flux_hllc_df(V_L_riemann[:, idx_eno], V_R_riemann[:, idx_eno],
                                gammaS_left[idx_eno], e0S_left[idx_eno],
                                gammaS_right[idx_eno], e0S_right[idx_eno])
        mass[idx_eno] = ff_eno.mass
        momentum[idx_eno] = ff_eno.momentum
        energy_left[idx_eno] = ff_eno.energy_left_owner
        energy_right[idx_eno] = ff_eno.energy_right_owner

    return (
        mass,
        momentum,
        energy_left,
        energy_right,
        mask,
        V_L_riemann,
        V_R_riemann,
        gammaS_left,
        e0S_left,
        gammaS_right,
        e0S_right,
        limited_mask,
        thinc_face_mask,
        use_thinc_cell,
    )


def get_rhs_hybrid_fc(V, dx, zeta=0.2, mask_variant='direct', limit_ho=False):
    """RHS hybride assemble, mode FULLY-CONSERVATIVE. Retourne (RHS, mask)."""
    mass, momentum, energy, mask, _, _, _ = get_hybrid_face_fluxes_fc(
        V, dx, zeta, mask_variant, limit_ho=limit_ho)
    RHS = _assemble_rhs_fc(mass, momentum, energy, dx)
    return RHS, mask


def get_rhs_hybrid_df(
    V,
    gammaS_frozen,
    e0S_frozen,
    dx,
    zeta=0.2,
    mask_variant='direct',
    limit_ho=False,
    use_thinc_bvd=False,
    **thinc_bvd_options,
):
    """RHS hybride assemble, mode DOUBLE-FLUX. Retourne (RHS, mask)."""
    res = get_hybrid_face_fluxes_df(
        V,
        gammaS_frozen,
        e0S_frozen,
        dx,
        zeta,
        mask_variant,
        limit_ho=limit_ho,
        use_thinc_bvd=use_thinc_bvd,
        **thinc_bvd_options,
    )
    mass, momentum, energy_left, energy_right, mask = res[:5]
    RHS = _assemble_rhs_df(mass, momentum, energy_left, energy_right, dx)
    return RHS, mask
