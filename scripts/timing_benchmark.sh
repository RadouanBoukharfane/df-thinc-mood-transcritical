#!/usr/bin/env bash
# Sequential timing benchmark (Table 8): one variant at a time, two repetitions.
# Run from the code/ directory; run times are stored in the metrics (runtime_s) of each output.
set -u
OUT=additional_results/timing
for rep in 1 2; do
  for m in df_ho df_thinc full full_nomood; do
    python3 scripts/run_nitrogen_interface.py run --profile sharp --nx 150 --velocity 100 --periods 1 \
        --methods $m --unlimited-ho --history-stride 50 --output $OUT/n2_rep$rep > /dev/null 2>&1
    python3 scripts/transcritical_shock_tube_extended_study.py run --method $m --nx 1024 --no-save-maps \
        --output $OUT/st_rep$rep > /dev/null 2>&1
    python3 scripts/run_drop_2d.py --methods $m --grids 50 --periods 0.25 --output $OUT/drop_rep$rep > /dev/null 2>&1
  done
done
