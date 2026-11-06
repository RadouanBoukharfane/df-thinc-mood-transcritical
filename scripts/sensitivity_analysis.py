"""Summary of the parameter-sensitivity runs (Table 3); shock tube compared with the exact solution."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fig_shock_tube import T_FINAL, exact, wave_crossings

ROOT = Path.cwd()
SENS = ROOT / "sensitivity"


def n2_row(label, value, path):
    meta = json.loads(path.with_suffix(".json").read_text())
    m = meta["metrics"]
    z = np.load(path)
    r0, r = z["V0"][0], z["V"][0]
    jump = r0.max() - r0.min()
    violation = max(0.0, r.max() - r0.max(), r0.min() - r.min()) / jump * 100
    return dict(case="N2 interface", parameter=label, value=value,
                width_cells=m["width_gradient_final"] * m.get("nx", 150) if "nx" in m else m["width_gradient_final"] * 150,
                L1_rho=m["L1_rho"], energy_defect=m["energy_error_final"],
                pressure_error=m["pressure_error_final"], velocity_error=m["velocity_error_final"],
                bound_violation_pct=violation, max_thinc=m["max_thinc_fraction"],
                max_dmp=m["max_dmp_fraction"], status=m["status"])


def st_row(label, value, path, sol):
    meta = json.loads(path.with_suffix(".json").read_text())
    m = meta["metrics"]
    case = dict(data=dict(np.load(path)), config=meta["config"])
    xc, xs = wave_crossings(case, sol)
    x, rho = case["data"]["x"], case["data"]["V"][0]
    rho_ex = exact.cell_average_profile(sol, x, x[1] - x[0], T_FINAL)["rho"]
    return dict(case="Shock tube", parameter=label, value=value,
                L1_rho=float(np.mean(np.abs(rho - rho_ex))),
                contact_error=abs(xc - (0.5 + sol["ustar"] * T_FINAL)),
                shock_error=abs(xs - (0.5 + sol["shock_speed"] * T_FINAL)),
                energy_defect=m["energy_rel_error"], max_thinc=m["max_thinc_fraction"],
                min_mood_level=m["minimum_mood_level"], status=m["status"])


def main():
    rows = []
    base_n2 = ROOT / "nitrogen_results" / "sharp" / "sharp_full_N150.npz"
    rows.append(n2_row("baseline", "", base_n2))
    for label, key, values in (("beta", "beta", ("1.2", "2.0", "2.5")),
                               ("zeta", "zeta", ("0.05", "0.1", "0.4")),
                               ("eta_c", "contact", ("0.01", "0.025", "0.1"))):
        for v in values:
            p = SENS / f"n2_{key}_{v}" / "sharp_full_N150.npz"
            if p.exists():
                rows.append(n2_row(label, v, p))

    reference = exact.solve()
    base_st = ROOT / "shock_tube_extended_results" / "shock_tube_full_N512.npz"
    rows.append(st_row("baseline", "", base_st, reference))
    for label, key, values in (("beta", "beta", ("1.2", "2.0", "2.5")),
                               ("zeta", "zeta", ("0.05", "0.1", "0.4")),
                               ("eta_p_mask", "mask", ("0.01", "0.05")),
                               ("n_buf", "buf", ("1", "3"))):
        for v in values:
            p = SENS / f"st_{key}_{v}" / "shock_tube_full_N512.npz"
            if p.exists():
                rows.append(st_row(label, v, p, reference))

    df = pd.DataFrame(rows)
    out = SENS / "sensitivity_summary.csv"
    df.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print("saved", out)


if __name__ == "__main__":
    main()
