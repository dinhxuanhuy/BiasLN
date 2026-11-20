#!/usr/bin/env bash
# Re-evaluate released FF++-trained checkpoints of published detectors on UADFV
# (and on Celeb-DF-v2 as a sanity check against Table 1 of Effort).
# Stage 1: the six DeepfakeBench detectors with official weights (release v1.0.1).
set -uo pipefail
W=/content/litw; mkdir -p $W/logs; cd $W
[ -d DeepfakeBench ] || git clone -q --depth 1 https://github.com/SCLBD/DeepfakeBench.git
cd DeepfakeBench
pip -q install efficientnet_pytorch timm kornia segmentation_models_pytorch 2>/dev/null | tail -1
mkdir -p training/weights training/pretrained
for m in f3net spsl srm core recce ucf xception; do
  f=training/weights/${m}_best.pth
  [ -s $f ] || wget -q -O $f https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/${m}_best.pth
done
if [ ! -f training/pretrained/.done ]; then
  wget -q -O /tmp/pre.zip https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.0/pretrained.zip \
    && unzip -q -o /tmp/pre.zip -d training/pretrained && touch training/pretrained/.done
  # the zip may contain a nested pretrained/ folder
  [ -d training/pretrained/pretrained ] && mv training/pretrained/pretrained/* training/pretrained/
fi
# point DeepfakeBench at our frames and JSONs
python - <<'PY'
import yaml
p='training/config/test_config.yaml'; c=yaml.safe_load(open(p))
c['rgb_dir']='/content/BiasLN/datasets/rgb'; c['dataset_json_folder']='/content/BiasLN/preprocessing/dataset_json'
c['lmdb']=False
yaml.safe_dump(c, open(p,'w'))
PY
ls training/weights training/pretrained | head -30
for m in f3net spsl srm core recce ucf; do
  echo "== $(date +%T) $m"
  python training/test.py --detector_path training/config/detector/$m.yaml \
    --test_dataset UADFV Celeb-DF-v2 --weights_path training/weights/${m}_best.pth \
    > $W/logs/$m.log 2>&1 || echo "!! $m failed"
  grep -aiE "video_auc|auc" $W/logs/$m.log | tail -4
done
echo "== lit stage1 done"
