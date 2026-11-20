#!/usr/bin/env bash
# LN-first follow-up (no aux loss, no tuning):
#   ln_l2  = LayerNorm gamma/beta only + L2 head          -> seeds 1024 3 4 5 (Kaggle has 1 2)
#   lnb18  = LN gamma/beta everywhere + biases in blocks 0-18 only
#   lane A: ln_l2 1024 3 4 ; lane B: ln_l2 5, lnb18 1024 1
set -uo pipefail
cd /content/BiasLN
export SYNC=${SYNC:-/content/drive/MyDrive/BiasLN_runs3}
OUT=results/anchor_val; LOGS=logs/anchor_val
run() { ENTRIES="$1" SEEDS="$2" AMP=bf16 SELECTION=val SOUP_FROM= OUT=$OUT LOGS=$LOGS \
        EXTRA_OPTS="workers=5" bash scripts/run_protocol.sh; }
lane_A() { run "ln_l2=BiasLN:tuning_mode=ln" "1024 3 4"; echo "== lane A finished"; }
lane_B() { run "ln_l2=BiasLN:tuning_mode=ln" 5; run "lnb18=BiasLN:bias_blocks=0-18" "1024 1"; echo "== lane B finished"; }
case "${LANE:-both}" in
  A) lane_A ;; B) lane_B ;;
  both) LANE=A bash "$0" > /content/laneA2.log 2>&1 & sleep 120; LANE=B bash "$0" > /content/laneB2.log 2>&1 & wait; echo "== colab lnfirst finished" ;;
esac
