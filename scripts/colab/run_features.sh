#!/usr/bin/env bash
# Frame features on Celeb-DF-v2 test (8 frames/video) for the feature-space figures:
# frozen CLIP, BitFit, BitFit+RA, BiasLN, GenD, Effort (all seed 1024, FF++-val selection).
set -uo pipefail
cd /content/BiasLN
D=/content/drive/MyDrive/BiasLN_runs
ck() {  # ck <run dir>  -> selected checkpoint path (rebased onto the run dir)
  python - "$1" <<'PY'
import json,sys,os
r=sys.argv[1]; p=json.load(open(os.path.join(r,"run_info.json")))["final_ckpt"]
i=p.find("/test/"); print(os.path.join(r,p[i+1:]) if i>=0 else p)
PY
}
OPTS="workers=8 frame_num.test=8"
feat() {  # feat <name> <cfg> <weights|pretrained> [opts]
  out=results/features/$1; [ -f "$out/scores/Celeb-DF-v2.npz" ] && return
  if [ "$3" = pretrained ]; then w="--pretrained"; else w="--weights_path $3"; fi
  python training/test.py --detector_path "$2" $w --test_dataset Celeb-DF-v2 --out_dir "$out" \
    --amp bf16 --save_feat --feat_out_dir results/features/pkl --opts $OPTS ${4:-}
}
feat clip training/config/detector/BiasLN.yaml pretrained "tuning_mode=frozen"
r=logs/anchor_val/ra_real_s1024; feat ra_real $r/config.yaml "$(ck $r)"
r=$D/roles_val/bitfit_l2_s1024; feat bitfit $r/config.yaml "$(ck $r)"
r=$D/val/biasln_s1024; feat biasln $r/config.yaml "$(ck $r)"
r=$D/val/gend_s1024; feat gend $r/config.yaml "$(ck $r)"
r=$D/val/effort_s1024; feat effort $r/config.yaml "$(ck $r)"
tar -czf /content/features.tgz results/features/*/scores
echo "== features done"
