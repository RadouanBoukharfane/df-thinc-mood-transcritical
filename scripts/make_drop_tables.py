"""Tables of the two-dimensional drop: refinement CSV, summary, Y extrema, split/unsplit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT = Path("drop_euler_results")
UNSPLIT = Path("drop_euler_results_unsplit")
CASES = ("df_ho_N100", "df_thinc_N100", "full_N100", "full_N25", "full_N50",
         "df_ho_N100_oblique", "full_N100_oblique")


def metrics(directory, case):
    p = directory / f"drop_{case}.json"
    return json.loads(p.read_text())["metrics"] if p.exists() else None


def refinement():
    rows = [metrics(SPLIT, f"full_N{n}") for n in (25, 50, 100)]
    df = pd.DataFrame(rows)
    df["mixed_width_cells"] = df["centerline_mixed_width_m"] / df["dx_m"]
    df.to_csv(SPLIT / "refinement_metrics.csv", index=False)
    dx = df["dx_m"].to_numpy()
    for col in ("rho_L1", "Y_L1", "energy_rel"):
        print(f"fitted rate of {col}: {np.polyfit(np.log(dx), np.log(df[col].to_numpy()), 1)[0]:.2f}")


def summary():
    print("\ncase                  L1(rho)  L1(Y)   width  viol[%]  eps_p     eps_u     eps_E   Y_min      Y_max-1")
    for case in CASES:
        m = metrics(SPLIT, case)
        if m is None:
            continue
        print(f"{case:20s}  {m['rho_L1']:7.3f}  {m['Y_L1']:.4f}  {m['centerline_mixed_width_m'] / m['dx_m']:5.0f}  "
              f"{m['overshoot_pct_jump']:7.3f}  {m['pressure_error_max']:.1e}  "
              f"{max(m['u_error_max'], m['v_error_max']):.1e}  {m['energy_rel']:.3f}   "
              f"{m['Y_min']:.1e}  {m['Y_max'] - 1:.1e}")


def split_vs_unsplit():
    print("\ncase                  L1(rho) split/unsplit   violation[%]     eps_E          eps_p")
    for case in CASES:
        a, b = metrics(SPLIT, case), metrics(UNSPLIT, case)
        if a is None or b is None:
            continue
        print(f"{case:20s}  {a['rho_L1']:6.2f} / {b['rho_L1']:6.2f}   {a['overshoot_pct_jump']:.3f} / {b['overshoot_pct_jump']:.3f}   "
              f"{a['energy_rel']:.3f} / {b['energy_rel']:.3f}   {a['pressure_error_max']:.1e} / {b['pressure_error_max']:.1e}")


if __name__ == "__main__":
    refinement()
    summary()
    split_vs_unsplit()
