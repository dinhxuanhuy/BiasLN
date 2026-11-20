#!/usr/bin/env bash
# Missing seeds for the revision / survey, run as two parallel lanes on one A100-40GB
# (each run needs ~8 GB in bf16 at batch 32; GenD ~5 GB at batch 16).
#   lane A: LN-only (CE) seeds 1,2  -> Table 6 gets n=3;  BiasLN, GenD seeds 3,4
#   lane B: LoRA seeds 1,2          -> Table 6 gets n=3;  BitFit seeds 3,4
# Same protocol as every other run: FF++ c23, 8 frames/video, selection on FF++ val.
# Resumable: run_protocol.sh skips (tag, seed) pairs whose metrics.json exists.
#   LANE=A bash scripts/colab/run_extra_seeds.sh   |   LANE=B ...   |   (no LANE: both, in background)
set -uo pipefail
cd /content/BiasLN
export SYNC=${SYNC:-/content/drive/MyDrive/BiasLN_runs3}
W=${WORKERS_PER_LANE:-5}
OUT=results/anchor_val; LOGS=logs/anchor_val
run() { ENTRIES="$1" SEEDS="$2" AMP=bf16 SELECTION=val SOUP_FROM= OUT=$OUT LOGS=$LOGS \
        EXTRA_OPTS="workers=$W" bash scripts/run_protocol.sh; }

lane_A() {
  run "ln_l2=BiasLN:tuning_mode=ln" "1 2"
  run "biasln=BiasLN gend=GenD" "3 4"
  echo "== lane A finished"
}
lane_B() {
  run "lora=BiasLN:tuning_mode=lora" "1 2"
  run "bitfit=BiasLN:tuning_mode=bitfit" "3 4"
  echo "== lane B finished"
}

case "${LANE:-both}" in
  A) lane_A ;;
  B) lane_B ;;
  both)
    LANE=A bash "$0" > /content/lane_A.log 2>&1 &
    sleep 120   # stagger start-up (CLIP load, dataset indexing)
    LANE=B bash "$0" > /content/lane_B.log 2>&1 &
    wait
    echo "== run_extra_seeds finished" ;;
esac
