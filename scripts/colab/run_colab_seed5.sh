#!/usr/bin/env bash
# Seed 5 of the four methods in the main comparison (n = 6 seeds with Kaggle's 3 and 4), two lanes.
#   lane A: BiasLN seed 5 ; BiasLN gamma-lr x3 seed 5
#   lane B: BitFit seed 5 ; GenD seed 5
set -uo pipefail
cd /content/BiasLN
export SYNC=${SYNC:-/content/drive/MyDrive/BiasLN_runs3}
OUT=results/anchor_val; LOGS=logs/anchor_val
run() { ENTRIES="$1" SEEDS="$2" AMP=bf16 SELECTION=val SOUP_FROM= OUT=$OUT LOGS=$LOGS \
        EXTRA_OPTS="workers=5" bash scripts/run_protocol.sh; }
lane_A() { run "biasln=BiasLN" 5; run "glr3=BiasLN:ln_gamma_lr_scale=3" 5; echo "== lane A finished"; }
lane_B() { run "bitfit=BiasLN:tuning_mode=bitfit" 5; run "gend=GenD" 5; echo "== lane B finished"; }
case "${LANE:-both}" in
  A) lane_A ;; B) lane_B ;;
  both) LANE=A bash "$0" > /content/laneA.log 2>&1 & sleep 120; LANE=B bash "$0" > /content/laneB.log 2>&1 & wait; echo "== colab seed5 finished" ;;
esac
