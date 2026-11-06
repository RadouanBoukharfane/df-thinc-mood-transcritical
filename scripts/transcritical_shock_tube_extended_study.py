"""Transcritical nitrogen shock tube (Section 8.2): runs, refinement, maps, histories."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from boundary import adiabatic_wall
from nonperiodic import (
    advance_ssprk3_nonperiodic,
    transcritical_shock_tube_ic,
)
from numerics import primitives_to_conservatives
from thermo import get_sos, get_T_from_p_rho


METHODS = ("df_ho", "df_thinc", "full", "full_nomood")


@dataclass
class Config:
    """Parameters of one run."""
    method: str = "full"
    nx: int = 512
    cfl: float = 0.8
    t_final: float = 5.0e-4
    zeta: float = 0.2
    shock_pressure_threshold: float = 2.0e-2
    shock_compression_threshold: float = 0.0
    shock_buffer_faces: int = 2
    thinc_beta: float = 1.6
    max_steps: int = 2_000_000
    map_stride: int = 10
    save_maps: bool = True

    def validate(self) -> "Config":
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if self.nx < 8:
            raise ValueError("nx must be at least 8")
        if self.cfl <= 0.0:
            raise ValueError("cfl must be positive")
        if self.t_final <= 0.0:
            raise ValueError("t_final must be positive")
        if self.map_stride < 1:
            raise ValueError("map_stride must be >= 1")
        return self


def method_options(method: str) -> dict[str, Any]:
    """Options of the DF+HO, DF+THINC-BVD, and complete variants."""
    if method == "df_ho":
        return dict(
            use_thinc_bvd=False,
            limit_ho=False,
            mood_mode="local_multilevel",
            check_entropy=False,
        )
    if method == "df_thinc":
        return dict(
            use_thinc_bvd=True,
            limit_ho=False,
            mood_mode="local_multilevel",
            check_entropy=False,
        )
    if method == "full":
        return dict(
            use_thinc_bvd=True,
            limit_ho=False,
            mood_mode="local_multilevel",
            check_entropy=True,
        )
    if method == "full_nomood":
        return dict(
            use_thinc_bvd=True,
            limit_ho=False,
            mood_mode="local_multilevel",
            check_entropy=True,
            cascade=False,
        )
    raise ValueError(method)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def combine_stage_diagnostics(stages, nx: int):
    """Combine the activation masks of the three stages."""
    cell_levels = np.minimum.reduce(
        [np.asarray(stage["cell_levels"], dtype=np.int8) for stage in stages]
    )
    face_levels = np.minimum.reduce(
        [np.asarray(stage["face_levels"], dtype=np.int8) for stage in stages]
    )
    thinc_cells = np.logical_or.reduce(
        [np.asarray(stage["thinc_cell_mask"], dtype=bool) for stage in stages]
    )
    dissipative_faces = np.logical_or.reduce(
        [np.asarray(stage["hllc_face_mask"], dtype=bool) for stage in stages]
    )
    corrected_faces = np.logical_or.reduce(
        [np.asarray(stage["corrected_faces"], dtype=bool) for stage in stages]
    )

    if cell_levels.shape != (nx,):
        raise ValueError(f"Unexpected cell-level shape: {cell_levels.shape}")
    if face_levels.shape != (nx + 1,):
        raise ValueError(f"Unexpected face-level shape: {face_levels.shape}")

    return cell_levels, face_levels, thinc_cells, dissipative_faces, corrected_faces


def run_extended(cfg: Config) -> dict[str, Any]:
    """Run one shock tube and collect histories and maps."""
    cfg.validate()
    options = method_options(cfg.method)

    x, dx, V = transcritical_shock_tube_ic(cfg.nx)
    V_initial = V.copy()

    W = primitives_to_conservatives(V)
    W_initial = W.copy()

    left_bc = adiabatic_wall()
    right_bc = adiabatic_wall()

    mass_initial = float(dx * np.sum(W_initial[0]))
    energy_initial = float(dx * np.sum(W_initial[2]))

    t = 0.0
    step = 0

    history_time = [0.0]
    history_energy_error = [0.0]
    history_mass_error = [0.0]

    map_time = []
    mood_maps = []
    face_level_maps = []
    thinc_maps = []
    dissipative_maps = []
    corrected_maps = []

    max_thinc_fraction = 0.0
    max_dissipative_fraction = 0.0
    max_corrected_fraction = 0.0
    minimum_mood_level = 3

    import time as _time
    tic = _time.perf_counter()
    status, failure = "completed", ""
    while t < cfg.t_final:
        wavespeed = np.abs(V[1]) + get_sos(V)

        if not np.all(np.isfinite(wavespeed)) or np.max(wavespeed) <= 0.0:
            status, failure = "failed", "invalid characteristic speed"
            break

        dt = cfg.cfl * dx / np.max(wavespeed)
        dt = min(dt, cfg.t_final - t)

        if not np.isfinite(dt) or dt <= 0.0:
            raise RuntimeError("Invalid time step")

        step_options = dict(
            zeta=cfg.zeta,
            limit_ho=options["limit_ho"],
            use_thinc_bvd=options["use_thinc_bvd"],
            shock_pressure_threshold=cfg.shock_pressure_threshold,
            shock_compression_threshold=cfg.shock_compression_threshold,
            shock_buffer_faces=cfg.shock_buffer_faces,
            thinc_beta=cfg.thinc_beta,
            check_entropy=options["check_entropy"],
        )
        if not options.get("cascade", True):
            step_options["cascade"] = False

        try:
            Wn, Vn, stages = advance_ssprk3_nonperiodic(
                V, W, dx, dt, left_bc, right_bc,
                mood_mode=options["mood_mode"], **step_options,
            )
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            status, failure = "failed", f"{type(exc).__name__}: {exc}"
            break

        V = Vn
        W = primitives_to_conservatives(V)

        t += dt
        step += 1

        if step > cfg.max_steps:
            raise RuntimeError("max_steps reached before final time")

        (
            cell_levels,
            face_levels,
            thinc_cells,
            dissipative_faces,
            corrected_faces,
        ) = combine_stage_diagnostics(stages, cfg.nx)

        minimum_mood_level = min(minimum_mood_level, int(cell_levels.min()))
        max_thinc_fraction = max(
            max_thinc_fraction,
            float(np.mean(thinc_cells)),
        )
        max_dissipative_fraction = max(
            max_dissipative_fraction,
            float(np.mean(dissipative_faces)),
        )
        max_corrected_fraction = max(
            max_corrected_fraction,
            float(np.mean(corrected_faces)),
        )

        mass_now = float(dx * np.sum(W[0]))
        energy_now = float(dx * np.sum(W[2]))

        history_time.append(float(t))
        history_mass_error.append(
            abs(mass_now - mass_initial) / max(abs(mass_initial), 1.0)
        )
        history_energy_error.append(
            abs(energy_now - energy_initial) / max(abs(energy_initial), 1.0)
        )

        record_map = (
            cfg.save_maps
            and (
                step == 1
                or step % cfg.map_stride == 0
                or t >= cfg.t_final
            )
        )

        if record_map:
            map_time.append(float(t))
            mood_maps.append(cell_levels.copy())
            face_level_maps.append(face_levels.copy())
            thinc_maps.append(thinc_cells.astype(np.uint8))
            dissipative_maps.append(dissipative_faces.astype(np.uint8))
            corrected_maps.append(corrected_faces.astype(np.uint8))

        if step == 1 or step % 100 == 0 or t >= cfg.t_final:
            print(
                f"[shock tube] method={cfg.method:8s} N={cfg.nx:5d} "
                f"step={step:6d} t={t:.6e} "
                f"eps_E={history_energy_error[-1]:.3e}"
            )

    T = get_T_from_p_rho(V[2], V[0])
    c = get_sos(V)

    result = dict(
        x=x,
        dx=float(dx),
        V=V,
        T=T,
        speed_of_sound=c,
        V_initial=V_initial,
        W_initial=W_initial,
        history_time=np.asarray(history_time),
        history_energy_error=np.asarray(history_energy_error),
        history_mass_error=np.asarray(history_mass_error),
        map_time=np.asarray(map_time),
        mood_map=np.asarray(mood_maps, dtype=np.int8),
        face_level_map=np.asarray(face_level_maps, dtype=np.int8),
        thinc_map=np.asarray(thinc_maps, dtype=np.uint8),
        dissipative_map=np.asarray(dissipative_maps, dtype=np.uint8),
        corrected_map=np.asarray(corrected_maps, dtype=np.uint8),
        metrics=dict(
            status=status,
            failure=failure,
            runtime_s=_time.perf_counter() - tic,
            step=int(step),
            t_final=float(t),
            mass_rel_error=float(history_mass_error[-1]),
            energy_rel_error=float(history_energy_error[-1]),
            minimum_mood_level=int(minimum_mood_level),
            max_thinc_fraction=float(max_thinc_fraction),
            max_dissipative_fraction=float(max_dissipative_fraction),
            max_corrected_fraction=float(max_corrected_fraction),
            rho_min=float(V[0].min()),
            rho_max=float(V[0].max()),
            u_min=float(V[1].min()),
            u_max=float(V[1].max()),
            p_min=float(V[2].min()),
            p_max=float(V[2].max()),
            T_min=float(T.min()),
            T_max=float(T.max()),
        ),
    )
    return result


def save_result(result: dict[str, Any], cfg: Config, output: Path, stem: str | None = None):
    """Save arrays (npz) and metadata (json)."""
    output.mkdir(parents=True, exist_ok=True)

    if stem is None:
        stem = f"shock_tube_{cfg.method}_N{cfg.nx}"

    npz_path = output / f"{stem}.npz"
    json_path = output / f"{stem}.json"

    array_keys = [
        "x",
        "V",
        "T",
        "speed_of_sound",
        "V_initial",
        "W_initial",
        "history_time",
        "history_energy_error",
        "history_mass_error",
        "map_time",
        "mood_map",
        "face_level_map",
        "thinc_map",
        "dissipative_map",
        "corrected_map",
    ]

    np.savez_compressed(
        npz_path,
        **{key: np.asarray(result[key]) for key in array_keys},
    )

    metadata = dict(
        config=asdict(cfg),
        method_options=method_options(cfg.method),
        metrics=result["metrics"],
    )

    json_path.write_text(
        json.dumps(json_safe(metadata), indent=2)
    )

    print(f"Saved: {npz_path}")
    print(f"Saved: {json_path}")

    return npz_path, json_path


def config_from_args(args, method: str, nx: int | None = None, save_maps: bool | None = None):
    return Config(
        method=method,
        nx=args.nx if nx is None else nx,
        cfl=args.cfl,
        t_final=args.t_final,
        zeta=args.zeta,
        shock_pressure_threshold=args.shock_pressure_threshold,
        shock_compression_threshold=args.shock_compression_threshold,
        shock_buffer_faces=args.shock_buffer_faces,
        thinc_beta=args.thinc_beta,
        max_steps=args.max_steps,
        map_stride=args.map_stride,
        save_maps=args.save_maps if save_maps is None else save_maps,
    )


def add_common(parser):
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--cfl", type=float, default=0.8)
    parser.add_argument("--t-final", type=float, default=5.0e-4)
    parser.add_argument("--zeta", type=float, default=0.2)
    parser.add_argument("--shock-pressure-threshold", type=float, default=2.0e-2)
    parser.add_argument("--shock-compression-threshold", type=float, default=0.0)
    parser.add_argument("--shock-buffer-faces", type=int, default=2)
    parser.add_argument("--thinc-beta", type=float, default=1.6)
    parser.add_argument("--max-steps", type=int, default=2_000_000)
    parser.add_argument("--map-stride", type=int, default=10)
    parser.add_argument("--save-maps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", type=Path, default=Path("shock_tube_extended_results"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run")
    add_common(run_parser)
    run_parser.add_argument("--method", choices=METHODS, default="full")

    all_parser = sub.add_parser("all")
    add_common(all_parser)
    all_parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    all_parser.add_argument("--reference-nx", type=int, default=4096)

    refinement_parser = sub.add_parser("refinement")
    add_common(refinement_parser)
    refinement_parser.add_argument("--method", choices=METHODS, default="full")
    refinement_parser.add_argument("--grids", nargs="+", type=int, default=[128, 256, 512, 1024])

    args = parser.parse_args()

    if args.command == "run":
        cfg = config_from_args(args, args.method)
        result = run_extended(cfg)
        save_result(result, cfg, args.output)
        return

    if args.command == "all":
        for method in args.methods:
            cfg = config_from_args(args, method)
            result = run_extended(cfg)
            save_result(result, cfg, args.output)

        ref_cfg = config_from_args(
            args,
            "full",
            nx=args.reference_nx,
            save_maps=False,
        )
        ref_result = run_extended(ref_cfg)
        save_result(
            ref_result,
            ref_cfg,
            args.output,
            stem=f"shock_tube_reference_N{args.reference_nx}",
        )
        return

    for nx in args.grids:
        cfg = config_from_args(
            args,
            args.method,
            nx=nx,
            save_maps=False,
        )
        result = run_extended(cfg)
        save_result(result, cfg, args.output)


if __name__ == "__main__":
    main()
