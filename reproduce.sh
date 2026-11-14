#!/usr/bin/env bash
# One-command reproduction of all analyses and figures.
# Usage:  bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")/code"
echo "== model constants and dynamics =="
python3 pk_model.py
python3 fast_sim.py
echo "== comparative reanalysis (asserts Table 1 values) =="
python3 reanalysis_pgls.py
echo "== analytic derivation: relative and absolute gradients, calibration, functional form =="
python3 analytic_care.py
echo "== simulation validation (fast vs reference): endpoints, final traits, basin by seed =="
python3 sim_benchmark.py
echo "== model robustness (gamma/V and static group-scaling) =="
python3 model_tightening.py
echo "== dynamic public good (care tracks group intelligence) =="
python3 public_good_dynamic.py
echo "== runaway basin: seed-level uncertainty and sensitivity =="
python3 basin_sensitivity.py
echo "== kin-structured care (helper relatedness), both demographic schemes =="
python3 kin_structured_care.py
echo "== early birth at small vs large brain size under private and shared care =="
python3 small_brain_early_birth.py
echo "== figures =="
python3 plot_figures.py
python3 plot_figure3_robustness.py
python3 plot_figure4_bodybrain.py
mkdir -p ../figures/regenerated/vector
mv -f figure1_reproduction.png ../figures/regenerated/figure1_reproduction.png
mv -f figure2_allocare.png     ../figures/regenerated/figure2_shared_care.png
mv -f figure3_robustness.png   ../figures/regenerated/figure3_robustness.png
mv -f figure4_bodybrain_trajectory.png ../figures/regenerated/figure4_bodybrain.png
mv -f figure1_reproduction.pdf ../figures/regenerated/vector/figure1_reproduction.pdf
mv -f figure2_allocare.pdf     ../figures/regenerated/vector/figure2_shared_care.pdf
mv -f figure3_robustness.pdf   ../figures/regenerated/vector/figure3_robustness.pdf
mv -f figure4_bodybrain_trajectory.pdf ../figures/regenerated/vector/figure4_bodybrain.pdf
rm -f figure4_taxon_summary.csv
echo "== done. Regenerated figures are in figures/regenerated/ =="
