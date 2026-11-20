#!/usr/bin/env bash
# Extra runs the revised manuscript needs (queued after run_anchor2.sh):
#   1. real anchoring on the paper's own BiasLN parameter set (bias + LN), 3 seeds
#   2. properly trained full fine-tuning (backbone lr 1e-5) and a linear probe (R1.5)
set -uo pipefail
export EXTRA_OPTS=${EXTRA_OPTS:-"workers=10"}
until grep -q "run_anchor2 finished" /content/anchor2.log 2>/dev/null; do sleep 120; done
OUT=results/anchor_val; LOGS=logs/anchor_val
run() { ENTRIES="$1" SEEDS="$2" AMP=bf16 SELECTION=val SOUP_FROM= OUT=$OUT LOGS=$LOGS bash scripts/run_protocol.sh; }
run "ra_biasln=BiasLN:lambda_anchor=1.0,anchor_on=real" 1024
run "fullft=BiasLN:tuning_mode=full,backbone_lr=0.00001 linprobe=BiasLN:tuning_mode=frozen" 1024
for s in 1 2; do run "ra_biasln=BiasLN:lambda_anchor=1.0,anchor_on=real" $s; done
echo "== run_paper_extra finished"
