#!/usr/bin/env bash
# Where to tune: BiasLN restricted to a range of encoder blocks (bias + LN of those blocks,
# plus the outer LNs). Same loss, schedule and FF++-val selection as BiasLN. Seed 1024 first.
set -uo pipefail
export EXTRA_OPTS=${EXTRA_OPTS:-"workers=10"}
export SYNC=${SYNC:-/content/drive/MyDrive/BiasLN_runs2}
OUT=results/anchor_val; LOGS=logs/anchor_val
run() { ENTRIES="$1" SEEDS="$2" AMP=bf16 SELECTION=val SOUP_FROM= OUT=$OUT LOGS=$LOGS bash scripts/run_protocol.sh; }
run "depth_last12=BiasLN:tune_blocks=12-23 depth_first12=BiasLN:tune_blocks=0-11 \
depth_last6=BiasLN:tune_blocks=last6 depth_mid12=BiasLN:tune_blocks=6-17 \
depth_first6=BiasLN:tune_blocks=first6" 1024
echo "== run_depth finished"
