# Double-flux THINC-BVD/MOOD/FCT solver for transcritical real-fluid flows

Code used to produce all numerical results and figures of

> S. El Oteby, R. Boukharfane, *Sharp and Conservative Numerical Simulation of
> Transcritical Material Interfaces in Real-Fluid Multicomponent Flows*.

- `src/`: the numerical method (thermodynamics, reconstructions, fluxes, MOOD,
  DMP/FCT, 1-D and 2-D solvers).
- `scripts/`: test-case drivers and the scripts producing the data, figures and
  tables of the paper.

Run everything from this directory, e.g. `python3 scripts/run_drop_2d.py`;
outputs go to sub-directories created on the fly. `reproduce_all.sh` lists
every command in the order of the paper.

## Requirements

Python 3 (tested with 3.13.9):

```bash
pip install -r requirements.txt
```

`CoolProp` is only needed for Fig. 1. The figure scripts render text with LaTeX
(`newtx` fonts), so a TeX installation is required for them.

## Files

### `src/`
| File | Content |
|---|---|
| `thermo.py` | Peng–Robinson closure of nitrogen |
| `thermo_binary_pr.py` | Peng–Robinson closure of the nitrogen/*n*-dodecane mixture (Appendix A) |
| `numerics.py`, `numerics_fc.py` | Primitive/conservative conversions |
| `reconstruction.py`, `eno2.py`, `thinc.py`, `bvd.py` | Centered (Eq. 16), ENO2 and THINC reconstructions, BVD selection |
| `sensor.py` | Oscillation sensor and contact/shock classification |
| `face_operators.py`, `hybrid_rhs.py` | Central, HLLC and LLF double-flux fluxes |
| `entropy.py`, `mood.py` | Entropy criterion and multilevel MOOD |
| `corrector.py`, `main_hybrid.py`, `species.py` | SSP-RK3 drivers and helpers |
| `nitrogen_interface_study.py` | 1-D periodic solver with full-vector DMP/FCT |
| `boundary.py`, `nonperiodic.py` | 1-D non-periodic solver (walls) |
| `drop_euler_2d.py`, `drop_euler_2d_unsplit.py` | 2-D multicomponent solver, split and unsplit |
| `exact_riemann_mixture.py` | Exact Riemann solver for the nitrogen/*n*-dodecane shock tubes |
| `exact_riemann_nitrogen.py` | Exact Riemann solver for the nitrogen shock tube (Section 8.2) |
| `phase_stability.py` | Tangent-plane-distance phase-stability test (Section 8.5) |

### `scripts/`
| File | Content |
|---|---|
| `run_nitrogen_interface.py` | 1-D nitrogen interface (Section 8.1, Table 3) |
| `transcritical_shock_tube_extended_study.py` | 1-D nitrogen shock tube (Section 8.2) |
| `shock_tube_mixture.py`, `run_exact_riemann_mixture.py` | Nitrogen/*n*-dodecane shock tubes (Section 8.3) |
| `run_drop_2d.py`, `run_drop_2d_unsplit.py` | 2-D drop (Section 8.5) |
| `mixing_layer_2d.py` | Temporal mixing layer (Section 8.6) |
| `fig_*.py`, `paperstyle.py` | Figures of the paper, with a common style (writes to `paper_figures/`) |
| `make_drop_tables.py`, `sensitivity_analysis.py`, `verify_clipping_order.py` | Tables and checks |
| `compare_shock_tube_exact.py` | Nitrogen shock tube versus the exact solution (Section 8.2) |
| `drop_phase_stability.py` | Phase stability of the computed drop states (Section 8.5) |
| `timing_benchmark.sh` | Sequential timings (Section 8.7, Table 8) |

## Correspondence with the paper

| Paper | Scripts (in `scripts/`) |
|---|---|
| Fig. 1 | `fig_nitrogen_properties.py` |
| Figs. 2–5 | `run_nitrogen_interface.py run --profile sharp`, `fig_nitrogen_sharp.py` |
| Figs. 6–7 | `run_nitrogen_interface.py refinement --profile smooth`, `fig_nitrogen_smooth.py`, `verify_clipping_order.py` |
| Figs. 8–10 | `transcritical_shock_tube_extended_study.py`, `fig_shock_tube.py` |
| Figs. 11–12 | `shock_tube_mixture.py`, `run_exact_riemann_mixture.py`, `fig_shock_tube_mixture.py` |
| Table 3 | parameter sweeps of `reproduce_all.sh` (Section 8.4), `sensitivity_analysis.py` |
| Figs. 14–18, Tables 4–6 | `run_drop_2d.py`, `run_drop_2d_unsplit.py`, `make_drop_tables.py`, `fig_drop.py` |
| Fig. 19 | `mixing_layer_2d.py`, `fig_mixing_layer.py` |
| Section 8.1 (fully conservative scheme, 10 periods), Table 7 | last block of `reproduce_all.sh` (output in `additional_results/`) |
| Section 8.2 (exact solution) | `compare_shock_tube_exact.py` |
| Section 8.5 (phase stability) | `drop_phase_stability.py` |
| Table 8 | `timing_benchmark.sh` |

Method names used by the scripts: `df_ho` (DF+HO), `df_thinc` (DF+THINC-BVD),
`full` (complete method), `full_nomood` (complete method without MOOD cascade and
post-correction fallback) and `fc` (fully conservative scheme, 1-D periodic case only).

Default parameters are those of Table 2. The high-order reconstruction is
unclipped in all reported runs (`--unlimited-ho` for the 1-D nitrogen tests);
the clipped version is used only for the comparison in Fig. 7.

## Run times (single core)

Seconds to minutes for the 1-D cases, except the smooth nitrogen case at
N = 800 (several hours). About 15 min per 2-D drop run on 100², about 3 h for
the 192² mixing layer.
