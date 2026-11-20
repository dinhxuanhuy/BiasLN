# Lists the folder layout needed to fix the prep path resolution (no recursive walk).
import os
B = "/kaggle/input/datasets/xuanhuydinh"
def ls(p, n=12):
    try:
        x = sorted(os.listdir(p)); print(p, len(x), x[:n], flush=True); return x
    except Exception as e:
        print(p, "ERR", e, flush=True); return []
ls(f"{B}/deepfakebench/rgb", 50)
for d in ls(f"{B}/deepfakebench/rgb", 50):
    if "DFDC" in d.upper():
        for e in ls(f"{B}/deepfakebench/rgb/{d}"):
            for g in ls(f"{B}/deepfakebench/rgb/{d}/{e}")[:3]:
                ls(f"{B}/deepfakebench/rgb/{d}/{e}/{g}", 5)
for m in ["uniface", "blendface", "e4s"]:
    p = f"{B}/deepfake40/dataset/{m}"
    for a in ls(p):
        for b in ls(f"{p}/{a}")[:4]:
            for c in ls(f"{p}/{a}/{b}", 5)[:2]:
                ls(f"{p}/{a}/{b}/{c}", 5)
