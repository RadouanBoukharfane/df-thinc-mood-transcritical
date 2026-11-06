#!/usr/bin/env bash
# Reproduce all numerical results and figures of the paper.
# Run from this directory (code/). Figures are written to paper_figures/. Independent commands may be run in parallel.
# Approximate single-core run times are given in brackets.
set -euo pipefail

# ---------------------------------------------------------------------------
# Figure 1: Peng--Robinson properties of nitrogen (requires CoolProp)
# ---------------------------------------------------------------------------
python3 scripts/fig_nitrogen_properties.py

# ---------------------------------------------------------------------------
# Section 8.1: one-dimensional nitrogen-interface advection
# (the reconstruction is unclipped in all reported computations)
# ---------------------------------------------------------------------------
# Discontinuous interface, N = 150, three variants [minutes]
python3 scripts/run_nitrogen_interface.py run --profile sharp --nx 150 --velocity 100 --periods 1 \
    --methods df_ho df_thinc full --unlimited-ho --output nitrogen_results/sharp
python3 scripts/fig_nitrogen_sharp.py

# Smooth profile, refinement of the complete method [N=800: several hours]
for n in 50 100 200 400 800; do
    python3 scripts/run_nitrogen_interface.py refinement --profile smooth --method full --unlimited-ho \
        --grids $n --output nitrogen_results/refinement_smooth_unclipped
done
# Clipped reconstruction, for the comparison of Fig. 7
for n in 50 100 200 400 800; do
    python3 scripts/run_nitrogen_interface.py refinement --profile smooth --method full \
        --grids $n --output nitrogen_results/refinement_smooth
done
python3 scripts/fig_nitrogen_smooth.py
python3 scripts/verify_clipping_order.py

# ---------------------------------------------------------------------------
# Section 8.2: transcritical nitrogen shock tube
# ---------------------------------------------------------------------------
python3 scripts/transcritical_shock_tube_extended_study.py refinement --method full \
    --grids 128 256 512 1024 --no-save-maps --output shock_tube_refinement_results
python3 scripts/transcritical_shock_tube_extended_study.py all --nx 512 --reference-nx 4096 \
    --map-stride 10 --output shock_tube_extended_results
python3 scripts/fig_shock_tube.py

# ---------------------------------------------------------------------------
# Section 8.3: nitrogen/n-dodecane shock tubes (Boyd & Jarrahbashi 2021)
# ---------------------------------------------------------------------------
for case in p60 p7; do
    python3 scripts/shock_tube_mixture.py --case $case --methods df_ho df_thinc full --grids 200
    python3 scripts/shock_tube_mixture.py --case $case --methods full --grids 400 800 1600 6400
done
python3 scripts/run_exact_riemann_mixture.py
python3 scripts/fig_shock_tube_mixture.py

# ---------------------------------------------------------------------------
# Section 8.4: sensitivity to the classification parameters (Table 3)
# ---------------------------------------------------------------------------
for b in 1.2 2.0 2.5; do
    python3 scripts/run_nitrogen_interface.py run --profile sharp --nx 150 --velocity 100 --methods full \
        --unlimited-ho --thinc-beta $b --output sensitivity/n2_beta_$b
    python3 scripts/transcritical_shock_tube_extended_study.py run --method full --nx 512 --no-save-maps \
        --thinc-beta $b --output sensitivity/st_beta_$b
done
for z in 0.05 0.1 0.4; do
    python3 scripts/run_nitrogen_interface.py run --profile sharp --nx 150 --velocity 100 --methods full \
        --unlimited-ho --zeta $z --output sensitivity/n2_zeta_$z
    python3 scripts/transcritical_shock_tube_extended_study.py run --method full --nx 512 --no-save-maps \
        --zeta $z --output sensitivity/st_zeta_$z
done
for c in 0.01 0.025 0.1; do
    python3 scripts/run_nitrogen_interface.py run --profile sharp --nx 150 --velocity 100 --methods full \
        --unlimited-ho --contact-threshold $c --output sensitivity/n2_contact_$c
done
for m in 0.01 0.05; do
    python3 scripts/transcritical_shock_tube_extended_study.py run --method full --nx 512 --no-save-maps \
        --shock-pressure-threshold $m --output sensitivity/st_mask_$m
done
for nb in 1 3; do
    python3 scripts/transcritical_shock_tube_extended_study.py run --method full --nx 512 --no-save-maps \
        --shock-buffer-faces $nb --output sensitivity/st_buf_$nb
done
python3 scripts/sensitivity_analysis.py

# ---------------------------------------------------------------------------
# Section 8.5: two-dimensional n-dodecane drop [100^2: about 15 min per run]
# ---------------------------------------------------------------------------
python3 scripts/run_drop_2d.py --methods df_ho df_thinc full --grids 100
python3 scripts/run_drop_2d.py --methods full --grids 25 50
python3 scripts/run_drop_2d.py --methods df_ho full --grids 100 --case oblique
# unsplit integration (Table "split vs unsplit")
python3 scripts/run_drop_2d_unsplit.py --methods df_ho df_thinc full --grids 100
python3 scripts/run_drop_2d_unsplit.py --methods full --grids 25 50
python3 scripts/run_drop_2d_unsplit.py --methods df_ho full --grids 100 --case oblique
python3 scripts/make_drop_tables.py
python3 scripts/fig_drop.py

# ---------------------------------------------------------------------------
# Section 8.6: temporal nitrogen/n-dodecane mixing layer [192^2: about 3 h]
# ---------------------------------------------------------------------------
python3 scripts/mixing_layer_2d.py --methods df_ho df_thinc full --grids 128
python3 scripts/mixing_layer_2d.py --methods full --grids 64 192
python3 scripts/fig_mixing_layer.py

# ---------------------------------------------------------------------------
# Additional verification (Sections 8.1, 8.2, 8.5, 8.7): exact Riemann solution,
# fully conservative (fc) comparison, long-time transport, MOOD ablation, cost, phase stability.
# Variant full_nomood: complete method without MOOD cascade and post-correction fallback.
# ---------------------------------------------------------------------------
A=additional_results
python3 src/exact_riemann_nitrogen.py
python3 scripts/compare_shock_tube_exact.py   # requires the Section 8.2 runs
for P in 1 10; do
    python3 scripts/run_nitrogen_interface.py run --profile sharp --nx 150 --velocity 100 --periods $P \
        --methods df_ho df_thinc full full_nomood --unlimited-ho --history-stride 5 --output $A/n2_sharp_P$P
done
python3 scripts/run_nitrogen_interface.py run --profile sharp --nx 150 --velocity 100 --methods fc \
    --unlimited-ho --output $A/n2_sharp_P1   # fails at the first step (negative pressure)
python3 scripts/run_nitrogen_interface.py run --profile smooth --nx 100 --methods fc full --unlimited-ho \
    --history-stride 20 --output $A/fc_smooth   # [about 10 min]
python3 scripts/transcritical_shock_tube_extended_study.py run --method full_nomood --nx 512 --no-save-maps \
    --output $A/shock_tube
for case in p60 p7; do
    python3 scripts/shock_tube_mixture.py --case $case --methods full full_nomood --grids 200 --output $A/mixture
done
python3 scripts/run_drop_2d.py --methods full_nomood --grids 50 100 --output $A/drop
bash scripts/timing_benchmark.sh               # sequential timings (Table 8)
python3 scripts/drop_phase_stability.py        # requires the Section 8.5 runs
