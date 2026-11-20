#!/usr/bin/env bash
# Multi-seed train + evaluate under ONE checkpoint-selection protocol.
# Runs from the repo root (locally or on a Colab VM). Resumable: a (tag, seed)
# pair whose metrics.json exists is skipped.
#
#   scripts/run_protocol.sh                       # main comparison, 5 seeds
#   ENTRIES="biasln=BiasLN gend=GenD" SEEDS="1 2" scripts/run_protocol.sh
#   ENTRIES="lora=BiasLN:tuning_mode=lora fullft=BiasLN:tuning_mode=full,backbone_lr=1e-5" \
#       scripts/run_protocol.sh
#   SELECTION=last scripts/run_protocol.sh        # target-agnostic: last epoch
#
# ENTRIES: space-separated  tag=ConfigName[:key=value,key=value]
# Outputs: $OUT/<tag>/seed<k>/{metrics.json,scores/*.npz,run_info.json,train/}
set -uo pipefail

ENTRIES=${ENTRIES:-"biasln=BiasLN gend=GenD effort=Effort"}
SEEDS=${SEEDS:-"1024 1 2 3 4"}
SELECTION=${SELECTION:-val}                 # val | last | test_oracle
VAL_DATASET=${VAL_DATASET:-FaceForensics++}
VAL_SPLIT=${VAL_SPLIT:-val}
CROSS_DATASET=${CROSS_DATASET:-"Celeb-DF-v2 DeepFakeDetection DFDC DFDCP UADFV"}
CROSS_METHOD=${CROSS_METHOD:-"uniface_ff blendface_ff e4s_ff facedancer_ff fsgan_ff inswap_ff simswap_ff"}
TEST_SETS=${TEST_SETS:-"$CROSS_DATASET $CROSS_METHOD FaceForensics++ FF-DF FF-F2F FF-FS FF-NT"}
OUT=${OUT:-results/$SELECTION}
LOGS=${LOGS:-logs/$SELECTION}
SYNC=${SYNC:-}                               # e.g. /content/drive/MyDrive/BiasLN_runs
AMP=${AMP:-none}
EXTRA_OPTS=${EXTRA_OPTS:-}                   # appended to every entry, e.g. "workers=4"
JSON_DIR=${JSON_DIR:-preprocessing/dataset_json}

# Only evaluate test sets whose JSON exists (DF40 may not be prepared yet).
present=""
for d in $TEST_SETS; do
  if [ -f "$JSON_DIR/$d.json" ]; then present="$present $d"; else echo "[warn] no $JSON_DIR/$d.json, skipping $d"; fi
done

# Selection-free variant: uniform average of epochs >= SOUP_FROM (empty = off).
SOUP_FROM=${SOUP_FROM-5}
soup_eval() {  # soup_eval <cfg> <run dir> <result dir> <opts>
  [ -n "$SOUP_FROM" ] || return 0
  [ -f "$3/metrics.json" ] && return 0
  [ -d "$2/epochs" ] || return 0
  python tools/average_ckpts.py --run "$2" --from_epoch "$SOUP_FROM" --out "$2/soup.pth" || return 0
  echo "== $(date '+%F %T') test soup (epochs >= $SOUP_FROM) of $2"
  python training/test.py --detector_path "$1" --weights_path "$2/soup.pth" \
    --test_dataset $present --out_dir "$3" --amp "$AMP" --opts $4 $EXTRA_OPTS \
    && cp "$2/run_info.json" "$2/config.yaml" "$3"/ || echo "!! soup test failed: $2"
}

mkdir -p "$OUT" "$LOGS"
for entry in $ENTRIES; do
  tag=${entry%%=*}; rest=${entry#*=}
  cfg=${rest%%:*}; opts=""
  [[ "$rest" == *:* ]] && opts=$(echo "${rest#*:}" | tr ',' ' ')
  cfg_path=training/config/detector/$cfg.yaml
  [ -f "$cfg_path" ] || { echo "missing $cfg_path"; exit 1; }
  for seed in $SEEDS; do
    res=$OUT/$tag/seed$seed
    run=${tag}_s${seed}
    soup_res=$OUT/${tag}_soup/seed$seed
    if [ -f "$res/metrics.json" ]; then
      echo "== skip $tag seed $seed (done)"
      soup_eval "$cfg_path" "$LOGS/$run" "$soup_res" "$opts"
      continue
    fi
    echo "== $(date '+%F %T') train $tag seed $seed ($cfg $opts) selection=$SELECTION"
    rm -rf "$LOGS/$run"
    python training/train.py --detector_path "$cfg_path" --selection "$SELECTION" \
      --val_dataset $VAL_DATASET --val_split "$VAL_SPLIT" --test_dataset $CROSS_DATASET \
      --seed "$seed" --log_dir "$LOGS" --run_name "$run" --no-save_feat \
      --opts amp=$AMP $opts $EXTRA_OPTS || { echo "!! train failed: $tag $seed"; continue; }
    ckpt=$(python -c "import json;print(json.load(open('$LOGS/$run/run_info.json'))['final_ckpt'])")
    echo "== $(date '+%F %T') test $tag seed $seed on:$present  ($ckpt)"
    python training/test.py --detector_path "$cfg_path" --weights_path "$ckpt" \
      --test_dataset $present --out_dir "$res" --amp "$AMP" --opts $opts $EXTRA_OPTS \
      || { echo "!! test failed: $tag $seed"; continue; }
    cp "$LOGS/$run/run_info.json" "$LOGS/$run/config.yaml" "$res"/
    soup_eval "$cfg_path" "$LOGS/$run" "$soup_res" "$opts"
    if [ -n "$SYNC" ]; then
      mkdir -p "$SYNC"
      rsync -a "$OUT" "$SYNC"/ && rsync -a --exclude 'data_dict_*.pickle' "$LOGS" "$SYNC"/
    fi
  done
done
echo "== $(date '+%F %T') all done -> $OUT"
