"""Periodic nitrogen-interface advection (Section 8.1).

Sharp and smooth profiles, double-flux variants, full-vector DMP/FCT with
post-correction admissibility check, refinement and sensitivity runs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from corrector import CorrectionFailedError
from entropy import entropy_double_flux
from face_operators import flux_llf_df
from main_hybrid import _select_df_euler_corrector
from hybrid_rhs import _assemble_rhs_df
from mood import MoodLevel, _detect_candidate_df
from numerics import primitives_to_conservatives, update_primitives_df
from numerics_fc import update_primitives_fc
from sensor import classify_contact_shock
from thermo import (
    get_T_from_p_rho,
    get_gammaS_e0S,
    get_rho_from_T_p,
    get_sos,
)


METHODS = ("fc", "df_ho", "df_thinc", "full", "full_nomood")


@dataclass
class StudyConfig:
    """Parameters of one run."""
    profile: str = "sharp"
    method: str = "full"
    nx: int = 150
    length: float = 1.0
    pressure: float = 5.0e6
    temperature_low: float = 100.0
    temperature_high: float = 300.0
    velocity: float | None = None
    cfl: float = 0.8
    periods: float = 1.0
    zeta: float = 0.2
    mask_variant: str = "direct"
    limit_ho: bool = True
    mood_mode: str = "local_multilevel"
    check_entropy: bool = True
    rtol: float = 1.0e-6
    max_iter: int = 10
    thinc_beta: float = 1.6
    contact_threshold: float = 5.0e-2
    shock_pressure_threshold: float = 5.0e-2
    shock_compression_threshold: float = 1.0e-3
    shock_velocity_threshold: float = 5.0e-2
    use_dmp: bool = True
    dmp_relaxation: float = 0.0
    dmp_abs_tolerance: float = 1.0e-12
    history_stride: int = 1

    def normalized(self) -> "StudyConfig":
        cfg = StudyConfig(**asdict(self))
        if cfg.profile not in {"sharp", "smooth"}:
            raise ValueError("profile must be 'sharp' or 'smooth'")
        if cfg.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if cfg.velocity is None:
            cfg.velocity = 100.0 if cfg.profile == "sharp" else 1.0
        if cfg.nx < 8:
            raise ValueError("nx must be at least 8")
        if cfg.velocity == 0.0:
            raise ValueError("velocity must be non-zero for an advection period")
        if cfg.history_stride < 1:
            raise ValueError("history_stride must be at least one")
        return cfg


def make_initial_state(cfg: StudyConfig) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Cell centres, spacing, primitive state, and temperature."""
    cfg = cfg.normalized()
    dx = cfg.length / cfg.nx
    x = (np.arange(cfg.nx) + 0.5) * dx

    rho_min = float(get_rho_from_T_p(np.array([cfg.temperature_high]), np.array([cfg.pressure]))[0])
    rho_max = float(get_rho_from_T_p(np.array([cfg.temperature_low]), np.array([cfg.pressure]))[0])

    if cfg.profile == "sharp":
        inside = (x > 0.25 * cfg.length) & (x < 0.75 * cfg.length)
        rho = np.where(inside, rho_max, rho_min)
    else:
        rho_mid = 0.5 * (rho_min + rho_max)
        rho_amp = 0.5 * (rho_max - rho_min)
        rho = rho_mid + rho_amp * np.sin(2.0 * np.pi * x / cfg.length)

    pressure = np.full(cfg.nx, cfg.pressure)
    velocity = np.full(cfg.nx, float(cfg.velocity))
    temperature = get_T_from_p_rho(pressure, rho)
    V = np.vstack((rho, velocity, pressure))
    return x, dx, V, temperature


def periodic_exact(initial: np.ndarray, x: np.ndarray, shift: float, length: float) -> np.ndarray:
    """Periodic translation of the initial profile."""
    xp = np.concatenate((x - length, x, x + length))
    fp = np.concatenate((initial, initial, initial))
    x_departure = np.mod(x - shift, length)
    return np.interp(x_departure, xp, fp)


def interface_width_gradient(rho: np.ndarray, dx: float) -> float:
    """Interface thickness from the maximum density gradient."""
    rho = np.asarray(rho)
    jump = float(np.max(rho) - np.min(rho))
    if jump <= 0.0:
        return 0.0
    grad = np.abs(np.roll(rho, -1) - np.roll(rho, 1)) / (2.0 * dx)
    maximum = float(np.max(grad))
    return jump / maximum if maximum > 0.0 else 0.0


def interface_width_mixed_cells(rho: np.ndarray, dx: float, low: float = 0.1, high: float = 0.9) -> float:
    """Average 10-90% mixed-layer width."""
    rho_min = float(np.min(rho))
    rho_max = float(np.max(rho))
    jump = rho_max - rho_min
    if jump <= 0.0:
        return 0.0
    q = (rho - rho_min) / jump
    mixed = (q > low) & (q < high)
    return 0.5 * float(np.count_nonzero(mixed)) * dx


def _low_order_fluxes(V: np.ndarray, gamma: np.ndarray, e0: np.ndarray):
    V_L = V
    V_R = np.roll(V, -1, axis=1)
    return flux_llf_df(
        V_L,
        V_R,
        gamma,
        e0,
        np.roll(gamma, -1),
        np.roll(e0, -1),
    )


def _fct_density_limit(
    rho_before: np.ndarray,
    high_flux: np.ndarray,
    low_flux: np.ndarray,
    lam: float,
    active_cells: np.ndarray,
    relaxation: float,
    abs_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho_before = np.asarray(rho_before)
    high_flux = np.asarray(high_flux)
    low_flux = np.asarray(low_flux)
    active_cells = np.asarray(active_cells, dtype=bool)

    rho_low = rho_before - lam * (low_flux - np.roll(low_flux, 1))
    anti = high_flux - low_flux
    c_left = lam * np.roll(anti, 1)
    c_right = -lam * anti

    rho_min = np.minimum.reduce((np.roll(rho_before, 1), rho_before, np.roll(rho_before, -1)))
    rho_max = np.maximum.reduce((np.roll(rho_before, 1), rho_before, np.roll(rho_before, -1)))
    scale = np.maximum(np.abs(rho_min), np.abs(rho_max))
    delta = relaxation * (rho_max - rho_min) + abs_tolerance * np.maximum(1.0, scale)

    p_plus = np.maximum(c_left, 0.0) + np.maximum(c_right, 0.0)
    p_minus = np.minimum(c_left, 0.0) + np.minimum(c_right, 0.0)
    q_plus = rho_max + delta - rho_low
    q_minus = rho_min - delta - rho_low

    r_plus = np.ones_like(rho_before)
    r_minus = np.ones_like(rho_before)
    mask_plus = p_plus > 0.0
    mask_minus = p_minus < 0.0
    r_plus[mask_plus] = np.minimum(1.0, np.maximum(0.0, q_plus[mask_plus] / p_plus[mask_plus]))
    r_minus[mask_minus] = np.minimum(1.0, np.maximum(0.0, q_minus[mask_minus] / p_minus[mask_minus]))

    theta = np.ones_like(anti)
    positive = anti > 0.0
    negative = anti < 0.0
    theta[positive] = np.minimum(r_minus[positive], np.roll(r_plus, -1)[positive])
    theta[negative] = np.minimum(r_plus[negative], np.roll(r_minus, -1)[negative])

    active_faces = active_cells | np.roll(active_cells, -1)
    theta[~active_faces] = 1.0
    limited_flux = low_flux + theta * anti
    rho_limited = rho_before - lam * (limited_flux - np.roll(limited_flux, 1))
    return rho_limited, theta, active_faces


def _apply_density_dmp(
    result: dict[str, Any],
    V_before: np.ndarray,
    W_before: np.ndarray,
    gamma: np.ndarray,
    e0: np.ndarray,
    dx: float,
    dt: float,
    cfg: StudyConfig,
) -> tuple[dict[str, Any], np.ndarray]:
    if not cfg.use_dmp or cfg.method not in {"full", "full_nomood"}:
        return result, np.ones(V_before.shape[1])

    classification = classify_contact_shock(
        V_before,
        contact_threshold=cfg.contact_threshold,
        shock_pressure_threshold=cfg.shock_pressure_threshold,
        shock_compression_threshold=cfg.shock_compression_threshold,
        shock_velocity_threshold=cfg.shock_velocity_threshold,
    )
    active_cells = classification.contact_cell & (~classification.shock_cell)
    low = _low_order_fluxes(V_before, gamma, e0)
    _, theta, active_faces = _fct_density_limit(
        W_before[0],
        result["mass_flux"],
        low.mass,
        dt / dx,
        active_cells,
        cfg.dmp_relaxation,
        cfg.dmp_abs_tolerance,
    )

    high = (
        result["mass_flux"],
        result["momentum_flux"],
        result["energy_left_flux"],
        result["energy_right_flux"],
    )
    low_parts = (low.mass, low.momentum, low.energy_left_owner, low.energy_right_owner)

    def limited_update(theta_face):
        fluxes = [lo + theta_face * (hi - lo) for hi, lo in zip(high, low_parts)]
        rhs = _assemble_rhs_df(*fluxes, dx)
        return W_before + dt * rhs, fluxes[0]

    fallback_faces = np.zeros_like(active_faces)
    nx = V_before.shape[1]
    for _ in range(nx + 1):
        W_limited, mass_limited = limited_update(theta)
        detection = _detect_candidate_df(
            W_limited, V_before, gamma, e0,
            rtol=cfg.rtol, check_entropy=cfg.check_entropy,
        )
        if not detection.troubled.any() or cfg.method == "full_nomood":
            if detection.V_after is None:
                raise CorrectionFailedError("[FCT] non-finite corrected candidate.", [], detection.troubled)
            break
        faces = detection.troubled | np.roll(detection.troubled, -1)
        new_faces = faces & (theta > 0.0)
        if not new_faces.any():
            raise CorrectionFailedError(
                "[FCT] Cells remain inadmissible with the low-order flux on "
                "all their faces.",
                [],
                detection.troubled,
            )
        theta = np.where(faces, 0.0, theta)
        fallback_faces |= new_faces
    else:
        raise CorrectionFailedError("[FCT] Fallback loop did not converge.", [], detection.troubled)

    result = dict(result)
    result["W_after"] = W_limited
    result["V_after"] = detection.V_after
    result["mass_flux"] = mass_limited
    result["dmp_theta"] = theta
    result["dmp_active_faces"] = active_faces
    result["dmp_limited_faces"] = theta < 1.0 - 1.0e-14
    result["fct_fallback_faces"] = fallback_faces
    return result, theta


def _advance_df_step(V: np.ndarray, W: np.ndarray, dx: float, dt: float, cfg: StudyConfig) -> dict[str, Any]:
    gamma, e0 = get_gammaS_e0S(V)
    use_thinc = cfg.method in {"df_thinc", "full", "full_nomood"}
    if cfg.method == "full_nomood":
        mood_mode = "none"
    else:
        mood_mode = cfg.mood_mode if cfg.method == "full" else "local_multilevel"
    advance_euler = _select_df_euler_corrector(
        mood_mode,
        zeta=cfg.zeta,
        mask_variant=cfg.mask_variant,
        max_iter=cfg.max_iter,
        rtol=cfg.rtol,
        limit_ho=cfg.limit_ho,
        use_thinc_bvd=use_thinc,
        thinc_bvd_options={
            "thinc_beta": cfg.thinc_beta,
            "sensor_options": {
                "contact_threshold": cfg.contact_threshold,
                "shock_pressure_threshold": cfg.shock_pressure_threshold,
                "shock_compression_threshold": cfg.shock_compression_threshold,
                "shock_velocity_threshold": cfg.shock_velocity_threshold,
            },
        },
        check_entropy=cfg.check_entropy,
    )

    stage_results: list[dict[str, Any]] = []

    s0 = entropy_double_flux(V, gamma)
    r1 = advance_euler(V, W, s0, gamma, e0, dx, dt)
    r1, _ = _apply_density_dmp(r1, V, W, gamma, e0, dx, dt, cfg)
    W1, V1 = r1["W_after"], r1["V_after"]
    stage_results.append(r1)

    s1 = entropy_double_flux(V1, gamma)
    r2e = advance_euler(V1, W1, s1, gamma, e0, dx, dt)
    r2e, _ = _apply_density_dmp(r2e, V1, W1, gamma, e0, dx, dt, cfg)
    W2 = 0.75 * W + 0.25 * r2e["W_after"]
    V2 = update_primitives_df(W2, gamma, e0)
    stage_results.append(r2e)

    s2 = entropy_double_flux(V2, gamma)
    r3e = advance_euler(V2, W2, s2, gamma, e0, dx, dt)
    r3e, _ = _apply_density_dmp(r3e, V2, W2, gamma, e0, dx, dt, cfg)
    W_new = (1.0 / 3.0) * W + (2.0 / 3.0) * r3e["W_after"]
    V_new = update_primitives_df(W_new, gamma, e0)
    stage_results.append(r3e)

    return {"W_after": W_new, "V_after": V_new, "stage_results": stage_results}


def _advance_fc_step(V: np.ndarray, W: np.ndarray, dx: float, dt: float, cfg: StudyConfig) -> dict[str, Any]:
    from corrector import correct_hybrid_rhs_fc

    gamma_ref, _ = get_gammaS_e0S(V)
    s0 = entropy_double_flux(V, gamma_ref)
    r1 = correct_hybrid_rhs_fc(
        V, W, s0, gamma_ref, dx, dt,
        zeta=cfg.zeta, mask_variant=cfg.mask_variant, max_iter=cfg.max_iter,
        rtol=cfg.rtol, limit_ho=cfg.limit_ho,
    )
    W1, V1 = r1["W_after"], r1["V_after"]

    s1 = entropy_double_flux(V1, gamma_ref)
    r2 = correct_hybrid_rhs_fc(
        V1, W1, s1, gamma_ref, dx, dt,
        zeta=cfg.zeta, mask_variant=cfg.mask_variant, max_iter=cfg.max_iter,
        rtol=cfg.rtol, limit_ho=cfg.limit_ho,
    )
    W2 = 0.75 * W + 0.25 * r2["W_after"]
    V2 = update_primitives_fc(W2)

    s2 = entropy_double_flux(V2, gamma_ref)
    r3 = correct_hybrid_rhs_fc(
        V2, W2, s2, gamma_ref, dx, dt,
        zeta=cfg.zeta, mask_variant=cfg.mask_variant, max_iter=cfg.max_iter,
        rtol=cfg.rtol, limit_ho=cfg.limit_ho,
    )
    W_new = (1.0 / 3.0) * W + (2.0 / 3.0) * r3["W_after"]
    V_new = update_primitives_fc(W_new)
    return {"W_after": W_new, "V_after": V_new, "stage_results": [r1, r2, r3]}


def _stage_level_array(result: dict[str, Any], nx: int) -> np.ndarray:
    if "cell_levels" in result:
        return np.asarray(result["cell_levels"], dtype=np.int8)
    level = int(result.get("accepted_level", int(MoodLevel.THINC_BVD_HYBRID)))
    return np.full(nx, level, dtype=np.int8)


def run_case(cfg: StudyConfig) -> dict[str, Any]:
    """Run one case and collect histories and metrics."""
    cfg = cfg.normalized()
    x, dx, V0, T0 = make_initial_state(cfg)
    V = V0.copy()
    W = primitives_to_conservatives(V)
    W0 = W.copy()

    period = cfg.length / abs(float(cfg.velocity))
    t_final = cfg.periods * period
    energy0 = float(np.sum(W0[2]) * dx)

    history: dict[str, list[Any]] = {
        "time": [], "energy_error": [], "pressure_error": [], "velocity_error": [],
        "width_gradient": [], "width_mixed": [], "thinc_fraction": [],
        "riemann_fraction": [], "dmp_fraction": [], "fct_fallback_count": [], "mood_min_level": [],
        "thinc_map": [], "riemann_map": [], "dmp_map": [], "mood_map": [],
    }

    def record(time: float, stages: list[dict[str, Any]] | None = None) -> None:
        pressure_error = float(np.max(np.abs(V[2] - cfg.pressure)) / cfg.pressure)
        velocity_error = float(np.max(np.abs(V[1] - cfg.velocity)))
        energy = float(np.sum(W[2]) * dx)
        history["time"].append(time)
        history["energy_error"].append(abs(energy - energy0) / max(abs(energy0), 1.0e-300))
        history["pressure_error"].append(pressure_error)
        history["velocity_error"].append(velocity_error)
        history["width_gradient"].append(interface_width_gradient(V[0], dx))
        history["width_mixed"].append(interface_width_mixed_cells(V[0], dx))

        if stages:
            thinc = np.logical_or.reduce([np.asarray(s.get("use_thinc_cell", np.zeros(cfg.nx, bool))) for s in stages])
            riemann = np.logical_or.reduce([np.asarray(s.get("mask", np.zeros(cfg.nx, bool))) for s in stages])
            dmp = np.logical_or.reduce([np.asarray(s.get("dmp_limited_faces", np.zeros(cfg.nx, bool))) for s in stages])
            fallback_count = int(sum(np.count_nonzero(s.get("fct_fallback_faces", np.zeros(cfg.nx, bool))) for s in stages))
            levels = np.minimum.reduce([_stage_level_array(s, cfg.nx) for s in stages])
        else:
            thinc = np.zeros(cfg.nx, dtype=bool)
            riemann = np.zeros(cfg.nx, dtype=bool)
            dmp = np.zeros(cfg.nx, dtype=bool)
            levels = np.full(cfg.nx, 3, dtype=np.int8)
        history["thinc_fraction"].append(float(np.mean(thinc)))
        history["riemann_fraction"].append(float(np.mean(riemann)))
        history["dmp_fraction"].append(float(np.mean(dmp)))
        history["fct_fallback_count"].append(fallback_count if stages else 0)
        history["mood_min_level"].append(int(np.min(levels)))
        history["thinc_map"].append(thinc.astype(np.int8))
        history["riemann_map"].append(riemann.astype(np.int8))
        history["dmp_map"].append(dmp.astype(np.int8))
        history["mood_map"].append(levels.astype(np.int8))

    record(0.0)
    import time as _time
    tic = _time.perf_counter()
    t = 0.0
    step = 0
    status = "completed"
    failure = ""

    while t < t_final - 10.0 * np.finfo(float).eps * max(1.0, t_final):
        dt = cfg.cfl * dx / float(np.max(np.abs(V[1]) + get_sos(V)))
        dt = min(dt, t_final - t)
        if not np.isfinite(dt) or dt <= 0.0:
            status, failure = "failed", "invalid time step (non-finite sound speed)"
            break
        try:
            result = _advance_fc_step(V, W, dx, dt, cfg) if cfg.method == "fc" else _advance_df_step(V, W, dx, dt, cfg)
        except (CorrectionFailedError, RuntimeError, ValueError, FloatingPointError) as exc:
            status = "failed"
            failure = f"{type(exc).__name__}: {exc}"
            break

        V = result["V_after"]
        W = result["W_after"] if cfg.method == "fc" else primitives_to_conservatives(V)
        t += dt
        step += 1
        if step % cfg.history_stride == 0 or t >= t_final:
            record(t, result["stage_results"])

    shift = float(cfg.velocity) * t
    rho_exact = periodic_exact(V0[0], x, shift, cfg.length)
    T_exact = periodic_exact(T0, x, shift, cfg.length)
    l1_rho = float(np.mean(np.abs(V[0] - rho_exact)))
    l2_rho = float(np.sqrt(np.mean((V[0] - rho_exact) ** 2)))
    linf_rho = float(np.max(np.abs(V[0] - rho_exact)))
    l1_T = float(np.mean(np.abs(get_T_from_p_rho(V[2], V[0]) - T_exact)))

    arrays = {
        "x": x,
        "V0": V0,
        "T0": T0,
        "V": V,
        "W": W,
        "rho_exact": rho_exact,
        "T_exact": T_exact,
    }
    for key, values in history.items():
        arrays[key] = np.asarray(values)

    metrics = {
        "status": status,
        "failure": failure,
        "runtime_s": _time.perf_counter() - tic,
        "max_pressure_error": float(np.max(history["pressure_error"])),
        "max_velocity_error": float(np.max(history["velocity_error"])),
        "steps": step,
        "time_reached": t,
        "time_target": t_final,
        "period": period,
        "L1_rho": l1_rho,
        "L2_rho": l2_rho,
        "Linf_rho": linf_rho,
        "L1_T": l1_T,
        "pressure_error_final": float(history["pressure_error"][-1]),
        "velocity_error_final": float(history["velocity_error"][-1]),
        "energy_error_final": float(history["energy_error"][-1]),
        "width_gradient_final": float(history["width_gradient"][-1]),
        "width_mixed_final": float(history["width_mixed"][-1]),
        "max_thinc_fraction": float(np.max(history["thinc_fraction"])),
        "max_riemann_fraction": float(np.max(history["riemann_fraction"])),
        "max_dmp_fraction": float(np.max(history["dmp_fraction"])),
        "total_fct_fallback_faces": int(np.sum(history["fct_fallback_count"])),
        "minimum_mood_level": int(np.min(history["mood_min_level"])),
        "rho_min": float(np.min(V[0])),
        "rho_max": float(np.max(V[0])),
    }
    return {"config": asdict(cfg), "metrics": metrics, "arrays": arrays}


def save_result(result: dict[str, Any], output_dir: Path) -> Path:
    """Save arrays (npz) and metrics (json)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = result["config"]
    stem = f"{cfg['profile']}_{cfg['method']}_N{cfg['nx']}"
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"
    np.savez_compressed(npz_path, **result["arrays"])
    json_path.write_text(json.dumps({"config": cfg, "metrics": result["metrics"]}, indent=2))
    return npz_path


def run_many(configs: Iterable[StudyConfig], output_dir: Path) -> list[Path]:
    """Run and save several configurations."""
    paths = []
    for cfg in configs:
        print(f"Running {cfg.profile}/{cfg.method}, N={cfg.nx}, periods={cfg.periods} ...")
        result = run_case(cfg)
        path = save_result(result, output_dir)
        print(json.dumps(result["metrics"], indent=2))
        paths.append(path)
    return paths


def _base_config_from_args(args: argparse.Namespace, method: str | None = None, nx: int | None = None) -> StudyConfig:
    return StudyConfig(
        profile=args.profile,
        method=method or getattr(args, "method", "full"),
        nx=nx or args.nx,
        velocity=args.velocity,
        cfl=args.cfl,
        periods=args.periods,
        zeta=args.zeta,
        limit_ho=not args.unlimited_ho,
        mood_mode=args.mood_mode,
        check_entropy=not args.no_entropy,
        rtol=args.rtol,
        thinc_beta=args.thinc_beta,
        contact_threshold=args.contact_threshold,
        shock_pressure_threshold=args.shock_pressure_threshold,
        shock_compression_threshold=args.shock_compression_threshold,
        shock_velocity_threshold=args.shock_velocity_threshold,
        use_dmp=not args.no_dmp,
        dmp_relaxation=args.dmp_relaxation,
        history_stride=args.history_stride,
    )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", choices=("sharp", "smooth"), default="sharp")
    parser.add_argument("--nx", type=int, default=150)
    parser.add_argument("--velocity", type=float, default=None)
    parser.add_argument("--cfl", type=float, default=0.8)
    parser.add_argument("--periods", type=float, default=1.0)
    parser.add_argument("--zeta", type=float, default=0.2)
    parser.add_argument("--thinc-beta", type=float, default=1.6)
    parser.add_argument("--contact-threshold", type=float, default=5.0e-2)
    parser.add_argument("--shock-pressure-threshold", type=float, default=5.0e-2)
    parser.add_argument("--shock-compression-threshold", type=float, default=1.0e-3)
    parser.add_argument("--shock-velocity-threshold", type=float, default=5.0e-2)
    parser.add_argument("--mood-mode", choices=("legacy", "global_multilevel", "local_multilevel"), default="local_multilevel")
    parser.add_argument("--rtol", type=float, default=1.0e-6)
    parser.add_argument("--unlimited-ho", action="store_true")
    parser.add_argument("--no-entropy", action="store_true")
    parser.add_argument("--no-dmp", action="store_true")
    parser.add_argument("--dmp-relaxation", type=float, default=0.0)
    parser.add_argument("--history-stride", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("nitrogen_results"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run one or several method variants")
    add_common_arguments(p_run)
    p_run.add_argument("--methods", nargs="+", choices=METHODS, default=["df_ho", "df_thinc", "full"])

    p_ref = sub.add_parser("refinement", help="run a mesh-refinement sequence")
    add_common_arguments(p_ref)
    p_ref.add_argument("--method", choices=METHODS, default="full")
    p_ref.add_argument("--grids", nargs="+", type=int, default=[50, 100, 200, 400])

    p_sens = sub.add_parser("sensitivity", help="run a one-parameter sensitivity sweep")
    add_common_arguments(p_sens)
    p_sens.add_argument("--method", choices=METHODS, default="full")
    p_sens.add_argument("--parameter", choices=("zeta", "thinc_beta", "contact_threshold", "shock_pressure_threshold"), required=True)
    p_sens.add_argument("--values", nargs="+", type=float, required=True)

    args = parser.parse_args()
    if args.command == "run":
        configs = [_base_config_from_args(args, method=m) for m in args.methods]
    elif args.command == "refinement":
        configs = [_base_config_from_args(args, method=args.method, nx=n) for n in args.grids]
    else:
        configs = []
        for value in args.values:
            cfg = _base_config_from_args(args, method=args.method)
            setattr(cfg, args.parameter, value)
            configs.append(cfg)
    run_many(configs, args.output)


if __name__ == "__main__":
    main()
