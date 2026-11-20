#!/usr/bin/env bash
# BiasLN differs from BitFit only by the LayerNorm scales gamma (init ~1). With the shared lr 3e-4
# gamma barely moves; lower gamma lr hurt (glr01/glr03). Test a larger gamma lr (3x, 10x).
# Same data/selection (FF++ val) as every other run; paired against bitfit on the same seeds.
set -uo pipefail
cd /content/BiasLN
export EXTRA_OPTS=${EXTRA_OPTS:-"workers=10"}
export SYNC=${SYNC:-/content/drive/MyDrive/BiasLN_runs2}
OUT=results/anchor_val; LOGS=logs/anchor_val
run() { ENTRIES="$1" SEEDS="$2" AMP=bf16 SELECTION=val SOUP_FROM= OUT=$OUT LOGS=$LOGS bash scripts/run_protocol.sh; }
run "glr3=BiasLN:ln_gamma_lr_scale=3 glr10=BiasLN:ln_gamma_lr_scale=10" 1024
run "glr3=BiasLN:ln_gamma_lr_scale=3" "${GLR3_SEEDS:-1 2}"
echo "== run_gamma_lr finished"
