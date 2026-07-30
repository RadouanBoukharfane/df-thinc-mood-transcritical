"""Transport conservatif de composition pour le solveur 1D.

Cette etape ajoute des especes passives sans modifier la thermodynamique
monofluide actuelle. Les inconnues conservatives sont Z_k = rho Y_k.
Le flux d'espece est construit a partir du flux massique hydro accepte :

    F_{rho Y_k} = Y_{k,adv} F_rho,

avec un etat advecte amont, borne et renormalise. On obtient donc, a
l'arrondi pres,

    sum_k F_{rho Y_k} = F_rho.

Convention : tableaux de forme (Ns, Nx), axe spatial en dernier.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpeciesDiagnostics:
    min_mass_fraction: float
    max_mass_fraction: float
    max_sum_error: float
    max_rho_closure_error: float
    species_masses: np.ndarray


def normalize_mass_fractions(Y, *, eps=1.0e-14):
    """Borne puis renormalise des fractions massiques cellule/face."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y doit etre de forme (Ns, Nx)")
    if Y.shape[0] < 1:
        raise ValueError("Il faut au moins une espece")
    if not np.isfinite(Y).all():
        raise ValueError("Y contient des valeurs non finies")
    if eps <= 0.0:
        raise ValueError("eps doit etre strictement positif")

    Yb = np.clip(Y, 0.0, 1.0)
    sums = np.sum(Yb, axis=0)
    bad = sums <= eps
    if np.any(bad):
        # Etat de repli deterministe : premiere espece pure.
        Yb[:, bad] = 0.0
        Yb[0, bad] = 1.0
        sums[bad] = 1.0
    return Yb / sums[None, :]


def species_mass_from_fractions(rho, Y):
    rho = np.asarray(rho, dtype=float)
    Y = normalize_mass_fractions(Y)
    if rho.ndim != 1 or rho.shape[0] != Y.shape[1]:
        raise ValueError("rho doit etre de forme (Nx,) compatible avec Y")
    if not np.isfinite(rho).all() or np.any(rho <= 0.0):
        raise ValueError("rho doit etre finie et strictement positive")
    return Y * rho[None, :]


def mass_fractions_from_species_mass(species_mass, rho, *, atol=1.0e-14):
    species_mass = np.asarray(species_mass, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if species_mass.ndim != 2:
        raise ValueError("species_mass doit etre de forme (Ns, Nx)")
    if rho.ndim != 1 or rho.shape[0] != species_mass.shape[1]:
        raise ValueError("rho incompatible avec species_mass")
    if not np.isfinite(species_mass).all() or not np.isfinite(rho).all():
        raise ValueError("etat d'especes non fini")
    if np.any(rho <= 0.0):
        raise ValueError("rho doit etre strictement positive")
    if np.any(species_mass < -atol):
        raise ValueError("une masse partielle est negative")
    Z = np.maximum(species_mass, 0.0)
    return normalize_mass_fractions(Z / rho[None, :])


def species_flux_from_mass_flux(mass_flux, Y_cells):
    """Flux conservatif d'especes par amont sur le signe de F_rho.

    La face i represente i+1/2. L'etat gauche vient de la cellule i et
    l'etat droit de i+1 (periodique).
    """
    mass_flux = np.asarray(mass_flux, dtype=float)
    Y_cells = normalize_mass_fractions(Y_cells)
    if mass_flux.ndim != 1 or mass_flux.shape[0] != Y_cells.shape[1]:
        raise ValueError("mass_flux doit etre de forme (Nx,) compatible avec Y")
    if not np.isfinite(mass_flux).all():
        raise ValueError("mass_flux contient des valeurs non finies")

    Y_left = Y_cells
    Y_right = np.roll(Y_cells, -1, axis=1)
    Y_adv = np.where(mass_flux[None, :] >= 0.0, Y_left, Y_right)
    Y_adv = normalize_mass_fractions(Y_adv)
    return Y_adv * mass_flux[None, :]


def species_rhs(species_flux, dx):
    species_flux = np.asarray(species_flux, dtype=float)
    if species_flux.ndim != 2:
        raise ValueError("species_flux doit etre de forme (Ns, Nx)")
    if dx <= 0.0:
        raise ValueError("dx doit etre strictement positif")
    return -(species_flux - np.roll(species_flux, 1, axis=1)) / dx


def advance_species_euler(species_mass_before, mass_flux, rho_after, dx, dt, *, atol=1.0e-11):
    """Un pas Euler conservatif couple au flux massique hydro accepte."""
    species_mass_before = np.asarray(species_mass_before, dtype=float)
    rho_before = np.sum(species_mass_before, axis=0)
    Y_before = mass_fractions_from_species_mass(species_mass_before, rho_before, atol=atol)
    flux = species_flux_from_mass_flux(mass_flux, Y_before)
    after = species_mass_before + dt * species_rhs(flux, dx)

    if not np.isfinite(after).all():
        raise ValueError("mise a jour d'especes non finie")
    if np.any(after < -atol):
        idx = np.argwhere(after < -atol)[0]
        raise ValueError(
            f"masse partielle negative apres Euler: espece={idx[0]}, cellule={idx[1]}, "
            f"valeur={after[tuple(idx)]:.6e}"
        )

    # Les petites valeurs negatives d'arrondi sont seulement nettoyees.
    after = np.maximum(after, 0.0)
    closure = np.sum(after, axis=0) - np.asarray(rho_after)
    scale = np.maximum(1.0, np.abs(rho_after))
    if np.max(np.abs(closure) / scale) > 100.0 * atol:
        raise ValueError(
            "incoherence masse-especes: sum(rhoY_k) ne reproduit pas rho"
        )
    return after, flux


def diagnose_species(species_mass, rho):
    Y = mass_fractions_from_species_mass(species_mass, rho, atol=1.0e-10)
    return SpeciesDiagnostics(
        min_mass_fraction=float(np.min(Y)),
        max_mass_fraction=float(np.max(Y)),
        max_sum_error=float(np.max(np.abs(np.sum(Y, axis=0) - 1.0))),
        max_rho_closure_error=float(np.max(np.abs(np.sum(species_mass, axis=0) - rho))),
        species_masses=np.sum(species_mass, axis=1),
    )
