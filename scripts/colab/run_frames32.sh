#!/usr/bin/env bash
# Data-quantity study: train on all 32 extracted frames per FF++ video (~115k frames, as in the
# GenD paper) instead of 8 evenly spaced frames (~29k). Same for BiasLN, BiasLN blocks 0-18 and GenD;
# nothing else changes (method, loss, epochs, selection on FF++ val).
set -uo pipefail
cd /content/BiasLN
export EXTRA_OPTS=${EXTRA_OPTS:-"workers=10 frame_num.train=32"}
export SYNC=${SYNC:-/content/drive/MyDrive/BiasLN_runs2}
OUT=results/anchor_val; LOGS=logs/anchor_val
run() { ENTRIES="$1" SEEDS="$2" AMP=bf16 SELECTION=val SOUP_FROM= OUT=$OUT LOGS=$LOGS bash scripts/run_protocol.sh; }
run "f32_0to18=BiasLN:tune_blocks=0-18 f32_gend=GenD f32_biasln=BiasLN" 1024
echo "== run_frames32 finished"
