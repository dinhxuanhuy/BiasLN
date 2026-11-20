#!/usr/bin/env bash
# Additive PEFT branch on one Colab GPU, two lanes:
#   lane A: AdaptFormer-style adapter (bottleneck 16), seeds 1024 1 2
#   lane B: VPT-deep (8 prompts/block),             seeds 1024 1 2
set -uo pipefail
cd /content/BiasLN
export SYNC=${SYNC:-/content/drive/MyDrive/BiasLN_runs3}
OUT=results/anchor_val; LOGS=logs/anchor_val
run() { ENTRIES="$1" SEEDS="$2" AMP=bf16 SELECTION=val SOUP_FROM= OUT=$OUT LOGS=$LOGS \
        CROSS_DATASET="Celeb-DF-v2 DeepFakeDetection DFDC DFDCP" EXTRA_OPTS="workers=5" bash scripts/run_protocol.sh; }
lane_A() { run "adapter=BiasLN:tuning_mode=adapter" "1024 1 2"; echo "== lane A finished"; }
lane_B() { run "vpt=BiasLN:tuning_mode=vpt" "1024 1 2"; echo "== lane B finished"; }
case "${LANE:-both}" in
  A) lane_A ;; B) lane_B ;;
  both) LANE=A bash "$0" > /content/laneA3.log 2>&1 & sleep 120; LANE=B bash "$0" > /content/laneB3.log 2>&1 & wait; echo "== colab additive finished" ;;
esac
