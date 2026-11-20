#!/usr/bin/env bash
# Real Anchoring follow-up paper (paper/improved_ra): runs postponed behind the BiasLN revision.
set -uo pipefail
bash scripts/colab/run_depth.sh   # BiasLN depth study first (revision), then RA
export EXTRA_OPTS=${EXTRA_OPTS:-"workers=10"}
export SYNC=${SYNC:-/content/drive/MyDrive/BiasLN_runs2}
B="BiasLN:tuning_mode=bitfit"
OUT=results/anchor_val; LOGS=logs/anchor_val
run() { ENTRIES="$1" SEEDS="$2" AMP=bf16 SELECTION=val SOUP_FROM= OUT=$OUT LOGS=$LOGS bash scripts/run_protocol.sh; }
run "ra_real=$B,lambda_anchor=1.0,anchor_on=real ra_all=$B,lambda_anchor=1.0,anchor_on=all \
ra_fake=$B,lambda_anchor=1.0,anchor_on=fake" "1 2"
run "ra_biasln=BiasLN:lambda_anchor=1.0,anchor_on=real" "1024 1 2"
python tools/stats.py --results $OUT --models ra_real bitfit ra_all ra_fake gend --reference ra_real \
  --out results/summary_anchor || true
echo "== run_ra_later finished"
