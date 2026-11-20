#!/usr/bin/env bash
# Seeds 1 and 2 for the two best depth configs at seed 1024 (blocks 0-18: Fisher-guided, drops the
# five lowest-Fisher blocks 19-23; blocks 6-17: middle 12). Same protocol as run_depth.sh.
set -uo pipefail
export EXTRA_OPTS=${EXTRA_OPTS:-"workers=10"}
export SYNC=${SYNC:-/content/drive/MyDrive/BiasLN_runs2}
OUT=results/anchor_val; LOGS=logs/anchor_val
run() { ENTRIES="$1" SEEDS="$2" AMP=bf16 SELECTION=val SOUP_FROM= OUT=$OUT LOGS=$LOGS bash scripts/run_protocol.sh; }
run "depth_0to18=BiasLN:tune_blocks=0-18 depth_mid12=BiasLN:tune_blocks=6-17" "1 2"
echo "== run_depth_seeds finished"
