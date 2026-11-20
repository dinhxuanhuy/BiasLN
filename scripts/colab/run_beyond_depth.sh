#!/usr/bin/env bash
# Two refinements of BiasLN on blocks 0-18 (the Fisher-guided depth), queued after run_depth_seeds.sh:
#  (1) roles: keep only the biases with the largest FF++-train Fisher mass (v, out, fc2) + LayerNorms;
#      q/fc1 biases are dropped and the k bias receives no gradient anyway;
#  (2) a separate, smaller learning rate for the LayerNorm scales (the only multiplicative parameters).
set -uo pipefail
until grep -q "run_depth_seeds finished" /content/job.log 2>/dev/null; do sleep 120; done
export EXTRA_OPTS=${EXTRA_OPTS:-"workers=10"}
export SYNC=${SYNC:-/content/drive/MyDrive/BiasLN_runs2}
OUT=results/anchor_val; LOGS=logs/anchor_val
run() { ENTRIES="$1" SEEDS="$2" AMP=bf16 SELECTION=val SOUP_FROM= OUT=$OUT LOGS=$LOGS bash scripts/run_protocol.sh; }
B="BiasLN:tune_blocks=0-18"
run "roles_vof_0to18=$B,param_roles=v+out+fc2+ln_gamma+ln_beta+outer_gamma+outer_beta \
glr03_0to18=$B,ln_gamma_lr_scale=0.3 glr01_0to18=$B,ln_gamma_lr_scale=0.1" 1024
echo "== run_beyond_depth finished"
