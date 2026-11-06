"""Species transport consistent with the mass flux."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SpeciesDiagnostics:
    """Bounds and closure errors of the mass fractions."""
    min_mass_fraction: float
    max_mass_fraction: float
    max_sum_error: float
    max_rho_closure_error: float
    species_masses: np.ndarray


def normalize_mass_fractions(Y, *, eps=1.0e-14):
    """Check and return mass fractions."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must have shape (Ns, Nx)")
    if Y.shape[0] < 1:
        raise ValueError("at least one species is required")
    if not np.isfinite(Y).all():
        raise ValueError("Y contains non-finite values")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    Yb = np.clip(Y, 0.0, 1.0)
    sums = np.sum(Yb, axis=0)
    bad = sums <= eps
    if np.any(bad):
        Yb[:, bad] = 0.0
        Yb[0, bad] = 1.0
        sums[bad] = 1.0
    return Yb / sums[None, :]


def species_mass_from_fractions(rho, Y):
    """Partial densities rho*Y_k."""
    rho = np.asarray(rho, dtype=float)
    Y = normalize_mass_fractions(Y)
    if rho.ndim != 1 or rho.shape[0] != Y.shape[1]:
        raise ValueError("rho must have shape (Nx,) compatible with Y")
    if not np.isfinite(rho).all() or np.any(rho <= 0.0):
        raise ValueError("rho must be finite and positive")
    return Y * rho[None, :]


def mass_fractions_from_species_mass(species_mass, rho, *, atol=1.0e-14):
    """Mass fractions from partial densities."""
    species_mass = np.asarray(species_mass, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if species_mass.ndim != 2:
        raise ValueError("species_mass must have shape (Ns, Nx)")
    if rho.ndim != 1 or rho.shape[0] != species_mass.shape[1]:
        raise ValueError("rho incompatible with species_mass")
    if not np.isfinite(species_mass).all() or not np.isfinite(rho).all():
        raise ValueError("non-finite species state")
    if np.any(rho <= 0.0):
        raise ValueError("rho must be positive")
    if np.any(species_mass < -atol):
        raise ValueError("a partial density is negative")
    Z = np.maximum(species_mass, 0.0)
    return normalize_mass_fractions(Z / rho[None, :])


def species_flux_from_mass_flux(mass_flux, Y_cells):
    """Upwind species fluxes from the mass flux."""
    mass_flux = np.asarray(mass_flux, dtype=float)
    Y_cells = normalize_mass_fractions(Y_cells)
    if mass_flux.ndim != 1 or mass_flux.shape[0] != Y_cells.shape[1]:
        raise ValueError("mass_flux must have shape (Nx,) compatible with Y")
    if not np.isfinite(mass_flux).all():
        raise ValueError("mass_flux contains non-finite values")

    Y_left = Y_cells
    Y_right = np.roll(Y_cells, -1, axis=1)
    Y_adv = np.where(mass_flux[None, :] >= 0.0, Y_left, Y_right)
    Y_adv = normalize_mass_fractions(Y_adv)
    return Y_adv * mass_flux[None, :]


def species_rhs(species_flux, dx):
    """Species residual."""
    species_flux = np.asarray(species_flux, dtype=float)
    if species_flux.ndim != 2:
        raise ValueError("species_flux must have shape (Ns, Nx)")
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    return -(species_flux - np.roll(species_flux, 1, axis=1)) / dx


def advance_species_euler(species_mass_before, mass_flux, rho_after, dx, dt, *, atol=1.0e-11):
    """Forward-Euler update of the partial densities."""
    species_mass_before = np.asarray(species_mass_before, dtype=float)
    rho_before = np.sum(species_mass_before, axis=0)
    Y_before = mass_fractions_from_species_mass(species_mass_before, rho_before, atol=atol)
    flux = species_flux_from_mass_flux(mass_flux, Y_before)
    after = species_mass_before + dt * species_rhs(flux, dx)

    if not np.isfinite(after).all():
        raise ValueError("non-finite species update")
    if np.any(after < -atol):
        idx = np.argwhere(after < -atol)[0]
        raise ValueError(
            f"negative partial density after Euler step: species={idx[0]}, cell={idx[1]}, "
            f"value={after[tuple(idx)]:.6e}"
        )

    after = np.maximum(after, 0.0)
    closure = np.sum(after, axis=0) - np.asarray(rho_after)
    scale = np.maximum(1.0, np.abs(rho_after))
    if np.max(np.abs(closure) / scale) > 100.0 * atol:
        raise ValueError(
            "mass-species inconsistency: sum(rhoY_k) differs from rho"
        )
    return after, flux


def diagnose_species(species_mass, rho):
    """Diagnostics of the species state."""
    Y = mass_fractions_from_species_mass(species_mass, rho, atol=1.0e-10)
    return SpeciesDiagnostics(
        min_mass_fraction=float(np.min(Y)),
        max_mass_fraction=float(np.max(Y)),
        max_sum_error=float(np.max(np.abs(np.sum(Y, axis=0) - 1.0))),
        max_rho_closure_error=float(np.max(np.abs(np.sum(species_mass, axis=0) - rho))),
        species_masses=np.sum(species_mass, axis=1),
    )
