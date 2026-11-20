#!/usr/bin/env bash
# Block ranges chosen from the FF++-train Fisher profile (results/grad_profile/biasln_init.json),
# i.e. without test data: blocks 12-18 carry the largest Fisher mass, blocks 19-23 the smallest.
set -uo pipefail
until grep -q "run_beat_gend finished" /content/beat_gend.log 2>/dev/null; do sleep 120; done
export EXTRA_OPTS=${EXTRA_OPTS:-"workers=10"}
export SYNC=${SYNC:-/content/drive/MyDrive/BiasLN_runs2}
OUT=results/anchor_val; LOGS=logs/anchor_val
run() { ENTRIES="$1" SEEDS="$2" AMP=bf16 SELECTION=val SOUP_FROM= OUT=$OUT LOGS=$LOGS bash scripts/run_protocol.sh; }
run "depth_fisher12to18=BiasLN:tune_blocks=12-18 depth_0to18=BiasLN:tune_blocks=0-18" 1024
echo "== run_fisher_depth finished"
