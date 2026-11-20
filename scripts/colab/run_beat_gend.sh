#!/usr/bin/env bash
# Focus: a BiasLN configuration that beats GenD. (Table-4 ablation and RA runs are paused.)
#  1. gradient / Fisher profile of bias+LN per block and role on FF++ *train* batches (no test data)
#  2. depth study: BiasLN restricted to encoder block ranges, seed 1024
set -uo pipefail
export EXTRA_OPTS=${EXTRA_OPTS:-"workers=10"}
export SYNC=${SYNC:-/content/drive/MyDrive/BiasLN_runs2}
mkdir -p results/grad_profile
python tools/grad_profile.py --detector_path training/config/detector/BiasLN.yaml \
  --batches 50 --warmup 100 --out results/grad_profile/biasln_init.json --opts amp=bf16 workers=8 \
  && cp -r results/grad_profile "$SYNC"/ || echo "!! grad profile failed"
bash scripts/colab/run_depth.sh
echo "== run_beat_gend finished"
