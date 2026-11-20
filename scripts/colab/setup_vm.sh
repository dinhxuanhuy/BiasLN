#!/usr/bin/env bash
# Prepare a Colab VM for BiasLN experiments. Runs ON the VM, from the repo root,
# after Google Drive is mounted at /content/drive (`colab drivemount -s <name>`).
#
#   bash scripts/colab/setup_vm.sh
#
# Drive layout:
#   $DRIVE/rgb/<Dataset>.zip     FaceForensics++ (incl. DFD), Celeb-DF-v2, DFDC, DFDCP, UADFV
#   $DRIVE/DF40/<method>.zip     DF40 methods (uniface, blendface, e4s, facedancer, fsgan, inswap, simswap)
# Archives are copied to local disk and only frames are extracted
# (preprocessing/prepare_data.py); JSONs are then built from the official
# split files shipped inside the archives (preprocessing/build_json.py).
set -euo pipefail

REPO=${REPO:-$(pwd)}
DRIVE=${DRIVE:-/content/drive/MyDrive}
RGB_DST=${RGB_DST:-$REPO/datasets/rgb}
JSON_DST=${JSON_DST:-$REPO/preprocessing/dataset_json}

cd "$REPO"
[ -d "$DRIVE" ] || { echo "Drive not mounted at $DRIVE. Run: colab drivemount -s <session>"; exit 1; }
mkdir -p "$RGB_DST" "$JSON_DST"

echo "== python deps"
python - <<'EOF'
import importlib, subprocess, sys
need = {"lmdb": "lmdb", "albumentations": "albumentations", "cv2": "opencv-python-headless",
        "transformers": "transformers", "sklearn": "scikit-learn", "tensorboard": "tensorboard"}
missing = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
import torch, transformers, albumentations
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "| transformers", transformers.__version__, "| albumentations", albumentations.__version__)
EOF

echo "== copy archives from Drive and extract frames only -> $RGB_DST"
python preprocessing/prepare_data.py --drive "$DRIVE" --dst "$RGB_DST" --workers "${WORKERS:-4}"

echo "== build dataset JSONs from official splits -> $JSON_DST"
python preprocessing/build_json.py --rgb_dir "$RGB_DST" --out "$JSON_DST"

echo "== CLIP ViT-L/14 weights (cached in ~/.cache/huggingface)"
python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-large-patch14'); print('   ok')"

echo "== split / path audit"
tests=""
for d in Celeb-DF-v2 DeepFakeDetection DFDC DFDCP UADFV uniface_ff blendface_ff e4s_ff facedancer_ff fsgan_ff inswap_ff simswap_ff; do
  [ -f "$JSON_DST/$d.json" ] && tests="$tests $d"
done
if [ -f "$JSON_DST/FaceForensics++.json" ] && [ -n "$tests" ]; then
  python tools/check_splits.py --json_dir "$JSON_DST" --rgb_dir "$RGB_DST" --check_paths 50 \
    --val FaceForensics++:val --test $tests --out results/split_audit.json || true
fi
df -h /content | tail -1
echo "== setup done"
