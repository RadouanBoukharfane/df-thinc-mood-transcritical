"""Nitrogen shock tube versus the exact Peng-Robinson Riemann solution (Section 8.2)."""
import json
from pathlib import Path

import numpy as np

from fig_shock_tube import CASES, REFINEMENT, T_FINAL, exact, load, wave_crossings

OUT = Path("additional_results") / "shock_tube_vs_exact.json"


def main():
    sol = exact.solve()
    xc_ex = 0.5 + sol["ustar"] * T_FINAL
    xs_ex = 0.5 + sol["shock_speed"] * T_FINAL
    files = sorted(REFINEMENT.glob("shock_tube_full_N*.npz"), key=lambda f: int(f.stem.split("N")[-1]))
    files += [CASES / "shock_tube_reference_N4096.npz"]
    files += [CASES / f"shock_tube_{m}_N512.npz" for m in ("df_ho", "df_thinc")]
    star_l = lambda x: (x > 0.40) & (x < 0.52)  # middle of the left star region
    star_r = lambda x: (x > 0.60) & (x < 0.68)  # middle of the right star region
    rows = []
    for f in files:
        case = load(f)
        d = case["data"]
        x, V, dx = d["x"], d["V"], d["x"][1] - d["x"][0]
        ex = exact.cell_average_profile(sol, x, dx, T_FINAL)
        xc, xs = wave_crossings(case, sol)
        rows.append(dict(
            case=f.stem, N=len(x),
            L1_rho=float(np.sum(np.abs(V[0] - ex["rho"])) * dx),
            L1_u=float(np.sum(np.abs(V[1] - ex["u"])) * dx),
            L1_p=float(np.sum(np.abs(V[2] - ex["p"])) * dx),
            contact_err=float(xc - xc_ex), shock_err=float(xs - xs_ex),
            p_starL=float(V[2][star_l(x)].mean()), u_starL=float(V[1][star_l(x)].mean()),
            rho_starL=float(V[0][star_l(x)].mean()), rho_starR=float(V[0][star_r(x)].mean()),
            T_starR=float(d["T"][star_r(x)].mean())))
        r = rows[-1]
        print(f"{r['case']:30s} L1rho={r['L1_rho']:.3f} dx_contact={r['contact_err']:+.2e} "
              f"dx_shock={r['shock_err']:+.2e} p*={r['p_starL'] / 1e6:.4f} MPa")
    ex_vals = dict(pstar=sol["pstar"], ustar=sol["ustar"], rhoLs=sol["rhoLs"], rhoRs=sol["rhoRs"],
                   TRs=sol["TRs"], x_contact=xc_ex, x_shock=xs_ex)
    print("exact:", json.dumps(ex_vals, default=float))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(exact=ex_vals, runs=rows), indent=1, default=float))
    print("saved", OUT)


if __name__ == "__main__":
    main()
