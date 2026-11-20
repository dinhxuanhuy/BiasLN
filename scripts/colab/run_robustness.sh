#!/usr/bin/env bash
# Robustness evaluation (no training): re-test the existing FF++-val-selected checkpoints of
# BiasLN, BitFit and GenD (3 seeds each) on corrupted copies of CDF-v2 and DFDCP.
# Corruptions are deterministic (training/dataset/abstract_dataset.py: apply_perturbation),
# so every model sees exactly the same images.
#   CKPT_ROOT=/content/drive/MyDrive/BiasLN_runs bash scripts/colab/run_robustness.sh
set -uo pipefail
cd /content/BiasLN
CKPT_ROOT=${CKPT_ROOT:-/content/drive/MyDrive/BiasLN_runs}
OUT=${OUT:-results/robust}
SYNC=${SYNC:-/content/drive/MyDrive/BiasLN_runs3}
DATASETS=${DATASETS:-"Celeb-DF-v2 DFDCP"}
PERTS=${PERTS:-"none jpeg:70 jpeg:40 blur:1 blur:2 down:0.5 down:0.25 noise:0.03 noise:0.06 bright:0.2"}
C=test/FaceForensics++/ckpt_best.pth

# method | config | seed | run dir (relative to CKPT_ROOT), from run_info.json final_ckpt
CKPTS="
biasln BiasLN 1024 val/biasln_s1024
biasln BiasLN 1 val/biasln_s1
biasln BiasLN 2 anchor_val/biasln_s2
bitfit BiasLN:tuning_mode=bitfit 1024 roles_val/bitfit_l2_s1024
bitfit BiasLN:tuning_mode=bitfit 1 anchor_val/bitfit_s1
bitfit BiasLN:tuning_mode=bitfit 2 anchor_val/bitfit_s2
gend GenD 1024 val/gend_s1024
gend GenD 1 anchor_val/gend_s1
gend GenD 2 anchor_val/gend_s2
"
mkdir -p logs/robust
echo "$CKPTS" | while read -r m spec seed dir; do
  [ -n "$m" ] || continue
  cfg=${spec%%:*}; opts=""; [[ "$spec" == *:* ]] && opts=$(echo "${spec#*:}" | tr ',' ' ')
  ck=logs/robust/${m}_s${seed}.pth
  [ -f "$ck" ] || cp "$CKPT_ROOT/$dir/$C" "$ck" || { echo "!! missing ckpt $dir"; continue; }
  for p in $PERTS; do
    tag=${p/:/_}; res=$OUT/$m/seed$seed/$tag
    [ -f "$res/metrics.json" ] && continue
    po=""; [ "$p" = none ] || po="test_perturb=$p"
    echo "== $(date '+%F %T') $m s$seed $p"
    python training/test.py --detector_path training/config/detector/$cfg.yaml --weights_path "$ck" \
      --test_dataset $DATASETS --out_dir "$res" --amp bf16 --opts $opts $po workers=10 \
      || echo "!! failed $m s$seed $p"
  done
  mkdir -p "$SYNC" && rsync -a "$OUT" "$SYNC/"
done
echo "== run_robustness finished"
