#!/usr/bin/env bash
# Evaluate every per-epoch checkpoint of finished runs, then summarise the
# selection-protocol gap (oracle-on-test vs. FF++-val vs. last epoch).
#
#   scripts/eval_epochs.sh logs/val/biasln_s1024 [logs/val/biasln_s1 ...]
#   TEST_SETS="Celeb-DF-v2 DFDC" scripts/eval_epochs.sh logs/val/gend_s1
set -uo pipefail
TEST_SETS=${TEST_SETS:-"Celeb-DF-v2 DeepFakeDetection DFDC DFDCP UADFV"}
AMP=${AMP:-none}
EXTRA_OPTS=${EXTRA_OPTS:-}   # e.g. "rgb_dir=... dataset_json_folder=..."
for run in "$@"; do
  cfg=$run/config.yaml
  for ck in "$run"/epochs/ckpt_epoch*.pth; do
    ep=$(basename "$ck" .pth); ep=${ep#ckpt_}
    out=$run/epoch_eval/$ep
    [ -f "$out/metrics.json" ] && continue
    echo "== $run $ep"
    python training/test.py --detector_path "$cfg" --weights_path "$ck" \
      --test_dataset $TEST_SETS --out_dir "$out" --amp "$AMP" --opts $EXTRA_OPTS || echo "!! failed $ck"
  done
  python tools/epoch_table.py "$run"
done
