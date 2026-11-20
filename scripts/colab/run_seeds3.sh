#!/usr/bin/env bash
# Third seed for the main-table baselines and two more seeds for the fake-only anchoring control,
# so every row of the main table and the asymmetry ablation has n=3 (queued after run_paper_extra.sh).
set -uo pipefail
export EXTRA_OPTS=${EXTRA_OPTS:-"workers=10"}
until grep -q "run_paper_extra finished" /content/extra.log 2>/dev/null; do sleep 120; done
B="BiasLN:tuning_mode=bitfit"
OUT=results/anchor_val; LOGS=logs/anchor_val
run() { ENTRIES="$1" SEEDS="$2" AMP=bf16 SELECTION=val SOUP_FROM= OUT=$OUT LOGS=$LOGS bash scripts/run_protocol.sh; }
run "biasln=BiasLN effort=Effort" 2
for s in 1 2; do run "ra_fake=$B,lambda_anchor=1.0,anchor_on=fake" $s; done
echo "== run_seeds3 finished"
