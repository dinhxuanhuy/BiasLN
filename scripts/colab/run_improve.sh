#!/usr/bin/env bash
# Larger changes to BiasLN, queued after run_beyond_depth.sh (seed 1024 first):
#  (no extra losses: the method stays BiasLN + weighted CE)
#  ml121824 : classifier on the mean of the CLS tokens of blocks 12, 18 and 24 (depth study: middle blocks matter)
set -uo pipefail
until grep -q "run_beyond_depth finished" /content/beyond.log 2>/dev/null; do sleep 120; done
export EXTRA_OPTS=${EXTRA_OPTS:-"workers=10"}
export SYNC=${SYNC:-/content/drive/MyDrive/BiasLN_runs2}
OUT=results/anchor_val; LOGS=logs/anchor_val
run() { ENTRIES="$1" SEEDS="$2" AMP=bf16 SELECTION=val SOUP_FROM= OUT=$OUT LOGS=$LOGS bash scripts/run_protocol.sh; }
run "ml121824=BiasLN:feature_layers=12+18+24" 1024
echo "== run_improve finished"
