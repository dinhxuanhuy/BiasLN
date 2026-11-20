# CLIP ViT-L/14 as plain files (Kaggle outputs drop the symlinks of a HF cache).
import os, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
from huggingface_hub import snapshot_download
p = snapshot_download("openai/clip-vit-large-patch14", local_dir="/kaggle/working/clip-vit-large-patch14",
                      allow_patterns=["*.json", "*.txt", "model.safetensors"])
for f in sorted(os.listdir(p)):
    fp = os.path.join(p, f)
    print(f, os.path.islink(fp), os.path.getsize(fp) if os.path.isfile(fp) else "dir")
