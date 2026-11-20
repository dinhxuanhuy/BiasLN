#!/usr/bin/env bash
# Round 2 of the real-anchoring study (after the seed-1024 pilot):
#  (a) seed 1024: anchor strength sweep + anchoring on the q/fc1 bias subset
#  (b) seeds 1, 2: ra_real vs its controls (bitfit, ra_all) and GenD -> mean +- std, paired tests
set -uo pipefail
export EXTRA_OPTS=${EXTRA_OPTS:-"workers=10"}
B="BiasLN:tuning_mode=bitfit"
OUT=results/anchor_val; LOGS=logs/anchor_val
# reuse the seed-1024 references trained on the first VM (same protocol)
R=/content/drive/MyDrive/BiasLN_runs/roles_val
for pair in bitfit:bitfit_l2 gend:gend; do
  dst=${pair%%:*}; src=${pair#*:}
  [ -d "$R/$src/seed1024" ] && [ ! -e "$OUT/$dst/seed1024" ] && mkdir -p $OUT/$dst && cp -r "$R/$src/seed1024" $OUT/$dst/
done
run() { ENTRIES="$1" SEEDS="$2" AMP=bf16 SELECTION=val SOUP_FROM= OUT=$OUT LOGS=$LOGS bash scripts/run_protocol.sh; }
run "ra_real_l03=$B,lambda_anchor=0.3,anchor_on=real \
ra_real_l3=$B,lambda_anchor=3.0,anchor_on=real \
ra_gate=$B,param_roles=q+fc1,lambda_anchor=1.0,anchor_on=real" 1024
for s in 1 2; do
  run "ra_real=$B,lambda_anchor=1.0,anchor_on=real bitfit=$B ra_all=$B,lambda_anchor=1.0,anchor_on=all gend=GenD" $s
  python tools/stats.py --results $OUT --models ra_real bitfit ra_all ra_fake gend --reference ra_real \
    --out results/summary_anchor || true
done
echo "== run_anchor2 finished"
