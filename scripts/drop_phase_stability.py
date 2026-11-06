"""Phase stability (TPD test) of the computed drop states and of the adiabatic mixing line (Section 8.5)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import phase_stability as ps  # noqa: E402
import thermo_binary_pr as pr  # noqa: E402

RES = Path("drop_euler_results")
OUT = Path("additional_results") / "drop_phase_stability.json"
CASES = {"split_N100": "drop_full_N100", "split_N50": "drop_full_N50",
         "split_N25": "drop_full_N25", "oblique_N100": "drop_full_N100_oblique"}


def main():
    out = {}
    line = ps.adiabatic_mixing_line(nZ=81)
    unstable_Y = [r["Y"] for r in line if r["tpd"] < -1e-6]
    out["mixing_line"] = dict(Y_unstable_min=min(unstable_Y), Y_unstable_max=max(unstable_Y), states=line)
    Y_line, T_line = np.array([r["Y"] for r in line]), np.array([r["T"] for r in line])
    print("adiabatic mixing line unstable for Y in", min(unstable_Y), max(unstable_Y))
    for name, stem in CASES.items():
        d = np.load(RES / f"{stem}.npz")
        Y, T, p, rho = (d[k].ravel() for k in ("Yfuel_final", "T_final", "p_final", "rho_final"))
        idx = np.flatnonzero((Y > 1e-3) & (Y < 1 - 1e-3))  # mixed cells
        X = pr.mass_to_mole_fraction(Y[idx])[1]
        tpd = np.array([ps.tpd_min(T[i], p[i], X[k], rho=rho[i], ntrial=200) for k, i in enumerate(idx)])
        unstable = tpd < -1e-6
        band = (Y > 0.4) & (Y < 0.7)
        out[name] = dict(
            n_cells=int(Y.size), n_mixed=int(idx.size), n_unstable=int(unstable.sum()),
            fuel_mass_in_unstable=float((rho[idx] * Y[idx])[unstable].sum() / (rho * Y).sum()),
            # computed versus adiabatic-mixing temperature, cells with 0.4 < Y < 0.7
            T_median_04_07=float(np.median(T[band])) if band.any() else None,
            T_adiabatic_at_median_Y=float(np.interp(np.median(Y[band]), Y_line, T_line)) if band.any() else None,
            T_range_unstable=[float(T[idx][unstable].min()), float(T[idx][unstable].max())] if unstable.any() else None)
        print(name, out[name])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, default=float))
    print("saved", OUT)


if __name__ == "__main__":
    main()
