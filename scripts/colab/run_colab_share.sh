#!/usr/bin/env bash
# Colab share of the extra runs, two lanes. The Kaggle L4 kernel runs every run; these are the
# ones at the END of its lanes (most likely to be cut by its 11.3 h budget), in reverse order.
#   lane A: LN-only without L2 (1024) ; LN-only + L2 seed 2 ; gamma-lr x3 seed 4
#   lane B: BitFit without L2 (1024)  ; LN-only + L2 seed 1 ; gamma-lr x3 seed 3
set -uo pipefail
cd /content/BiasLN
export SYNC=${SYNC:-/content/drive/MyDrive/BiasLN_runs3}
OUT=results/anchor_val; LOGS=logs/anchor_val
run() { ENTRIES="$1" SEEDS="$2" AMP=bf16 SELECTION=val SOUP_FROM= OUT=$OUT LOGS=$LOGS \
        EXTRA_OPTS="workers=5" bash scripts/run_protocol.sh; }
lane_A() { run "ln_nol2=BiasLN:tuning_mode=ln,l2_norm=false" 1024; run "ln_l2=BiasLN:tuning_mode=ln" 2
           run "glr3=BiasLN:ln_gamma_lr_scale=3" 4; echo "== lane A finished"; }
lane_B() { run "bitfit_nol2=BiasLN:tuning_mode=bitfit,l2_norm=false" 1024; run "ln_l2=BiasLN:tuning_mode=ln" 1
           run "glr3=BiasLN:ln_gamma_lr_scale=3" 3; echo "== lane B finished"; }
case "${LANE:-both}" in
  A) lane_A ;; B) lane_B ;;
  both) LANE=A bash "$0" > /content/laneA.log 2>&1 & sleep 120; LANE=B bash "$0" > /content/laneB.log 2>&1 & wait; echo "== colab share finished" ;;
esac
