#!/usr/bin/env bash
# Post-hoc weight-space variants (no training): 3-seed uniform soup and WiSE-FT (alpha=0.5, fixed
# a priori) of that soup, for BiasLN, GenD and BiasLN blocks 0-18. Same test sets as run_protocol.sh.
set -uo pipefail
cd /content/BiasLN
D=/content/drive/MyDrive/BiasLN_runs
OUT=results/anchor_val; AMP=bf16; EXTRA_OPTS=${EXTRA_OPTS:-"workers=10"}
JSON_DIR=preprocessing/dataset_json
TEST_SETS="Celeb-DF-v2 DeepFakeDetection DFDC DFDCP UADFV uniface_ff blendface_ff e4s_ff facedancer_ff fsgan_ff inswap_ff simswap_ff FaceForensics++ FF-DF FF-F2F FF-FS FF-NT"
present=""; for d in $TEST_SETS; do [ -f "$JSON_DIR/$d.json" ] && present="$present $d"; done
C=FaceForensics++/ckpt_best.pth
mkdir -p logs/soups
fetch() { local dst=logs/soups/$1.pth; [ -f "$dst" ] || cp "$D/$2/test/$C" "$dst"; echo "$dst"; }

job() {  # job <tag> <cfg> <opts> <ckpt...>
  local tag=$1 cfg=training/config/detector/$2.yaml opts=$3; shift 3
  for alpha in 1.0 0.5; do
    local name=${tag}_seedsoup; [ "$alpha" = 1.0 ] || name=${tag}_seedsoup_wise05
    local res=$OUT/$name/seed0
    [ -f "$res/metrics.json" ] && { echo "== skip $name"; continue; }
    echo "== $(date '+%F %T') build $name"
    python tools/wise_soup.py --detector_path "$cfg" --ckpts "$@" --alpha $alpha \
      --out logs/soups/$name.pth --opts $opts || { echo "!! build failed $name"; continue; }
    echo "== $(date '+%F %T') test $name"
    python training/test.py --detector_path "$cfg" --weights_path logs/soups/$name.pth \
      --test_dataset $present --out_dir "$res" --amp $AMP --opts $opts $EXTRA_OPTS \
      || echo "!! test failed $name"
    mkdir -p /content/drive/MyDrive/BiasLN_runs2 && rsync -a $OUT /content/drive/MyDrive/BiasLN_runs2/
  done
}

job biasln BiasLN "" $(fetch biasln_s1024 val/biasln_s1024) $(fetch biasln_s1 val/biasln_s1) $(fetch biasln_s2 anchor_val/biasln_s2)
job gend GenD "" $(fetch gend_s1024 val/gend_s1024) $(fetch gend_s1 anchor_val/gend_s1) $(fetch gend_s2 anchor_val/gend_s2)
job depth_0to18 BiasLN "tune_blocks=0-18" $(fetch d18_s1024 anchor_val/depth_0to18_s1024) $(fetch d18_s1 anchor_val/depth_0to18_s1) $(fetch d18_s2 anchor_val/depth_0to18_s2)
echo "== run_soups finished"
