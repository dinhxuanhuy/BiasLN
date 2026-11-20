# BiasLN prep kernel (CPU, internet ON). Produces, in /kaggle/working:
#   dataset_json/*.json  DeepfakeBench-format JSONs with frame paths rewritten as
#                        "DFB/<path in deepfakebench>" or "DF40/<path in deepfake40>",
#                        frames sorted numerically, DF40 labels mapped to FF-real / DF_fake
#   hf/                  HuggingFace cache with openai/clip-vit-large-patch14 (offline use)
#   wheels/              wheels for packages missing from the L4 image (no internet there)
#   prep_report.json     per-dataset video / frame counts per split
import glob, json, os, re, subprocess, sys, time

W = "/kaggle/working"

def sh(c):
    print("$", c, flush=True)
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-2000:], f"[rc={r.returncode}]", flush=True)
    return r.returncode

def find_root(slug):
    for c in [f"/kaggle/input/{slug}", f"/kaggle/input/datasets/xuanhuydinh/{slug}"] + \
             glob.glob(f"/kaggle/input/*/{slug}") + glob.glob(f"/kaggle/input/*/*/{slug}"):
        if os.path.isdir(c):
            return c
    raise SystemExit(f"dataset {slug} not mounted; /kaggle/input = {os.listdir('/kaggle/input')}")

DFB, DF40 = find_root("deepfakebench"), find_root("deepfake40")
print("roots", DFB, DF40, os.listdir(DFB), os.listdir(DF40), flush=True)
ROOTS = {"DFB": DFB, "DF40": DF40}

# --------------------------------------------------------------- path resolution
_prefix_cache = {}  # tuple(prefix parts) -> (tag, n_strip, insert)

def resolve(p):
    parts = [x for x in re.split(r"[\\/]+", p) if x]
    for k in range(1, len(parts)):
        key = tuple(parts[:k])
        if key in _prefix_cache:
            tag, strip, ins = _prefix_cache[key]
            rel = "/".join(ins + parts[strip:])
            if os.path.exists(os.path.join(ROOTS[tag], rel)):
                return f"{tag}/{rel}"
    # search: drop 0..4 leading parts, try a few insertion dirs under each root
    for strip in range(0, min(5, len(parts) - 1)):
        for tag, root in ROOTS.items():
            for ins in ([], ["rgb"], ["dataset"]):
                rel = "/".join(ins + parts[strip:])
                if os.path.exists(os.path.join(root, rel)):
                    for k in range(1, strip + 2):
                        _prefix_cache.setdefault(tuple(parts[:k]), (tag, strip, ins))
                    return f"{tag}/{rel}"
    return None

def fkey(p):
    s = os.path.splitext(os.path.basename(p))[0]
    return (0, int(s)) if s.isdigit() else (1, s)

def convert(src, name, relabel=None):
    j = json.load(open(src))
    top = list(j)[0]
    out, report, missing = {name: {}}, {}, 0
    for label, splits in j[top].items():
        new_label = relabel(label) if relabel else label
        node = out[name].setdefault(new_label, {})
        for split, vids in splits.items():
            dst = node.setdefault(split, {})
            items = vids
            comp = None
            if isinstance(vids, dict) and set(vids) & {"c23", "c40", "c0", "raw"}:
                comp, items = "c23", vids.get("c23", {})
                dst = dst.setdefault("c23", {})
            if not isinstance(items, dict):
                continue
            for vid, info in items.items():
                if not isinstance(info, dict) or not isinstance(info.get("frames"), list):
                    continue  # malformed entries (e.g. DFDC 'val' in the DeepfakeBench JSON)
                fr = sorted(info["frames"], key=fkey)
                # resolve the first frame on disk; the others share its directory mapping
                r0 = resolve(fr[0]) if fr else None
                if r0 is None:
                    missing += len(fr)
                    continue
                p0 = [x for x in re.split(r"[\\/]+", fr[0]) if x]
                tail0 = p0[-1]
                head = r0[: len(r0) - len(tail0)]
                frames = [head + [x for x in re.split(r"[\\/]+", f) if x][-1] for f in fr]
                if not os.path.exists(os.path.join(ROOTS[frames[-1].split("/")[0]],
                                                   frames[-1].split("/", 1)[1])):
                    missing += 1  # spot-check the last frame of each video
                if frames:
                    key = vid if vid not in dst else f"{new_label}/{vid}"
                    dst[key] = {"label": new_label, "frames": frames}
                    rs = report.setdefault(split, [0, 0]); rs[0] += 1; rs[1] += len(frames)
    os.makedirs(f"{W}/dataset_json", exist_ok=True)
    json.dump(out, open(f"{W}/dataset_json/{name}.json", "w"))
    print(f"{name}: {report} missing_frames={missing}", flush=True)
    return {"splits": report, "missing_frames": missing}

t0 = time.time()
report = {}
for n in ["FaceForensics++", "FF-DF", "FF-F2F", "FF-FS", "FF-NT", "Celeb-DF-v2",
          "DeepFakeDetection", "DFDC", "DFDCP", "UADFV"]:
    src = f"{DFB}/dataset_json/{n}.json"
    report[n] = convert(src, n) if os.path.exists(src) else "absent"
df40_json = sorted(glob.glob(f"{DF40}/**/dataset_json", recursive=True))[0]
for m in ["uniface", "blendface", "e4s", "facedancer", "fsgan", "inswap", "simswap"]:
    src = f"{df40_json}/{m}_ff.json"
    report[f"{m}_ff"] = (convert(src, f"{m}_ff",
                                 relabel=lambda l: "FF-real" if l.lower().endswith("real") else "DF_fake")
                         if os.path.exists(src) else "absent")
json.dump(report, open(f"{W}/prep_report.json", "w"), indent=1)
print(f"json conversion done in {time.time()-t0:.0f}s", flush=True)

# --------------------------------------------------------------- CLIP + wheels
os.environ["HF_HOME"] = f"{W}/hf"
sh(f"{sys.executable} -m pip install -q huggingface_hub")
sh(f"{sys.executable} -c \"from huggingface_hub import snapshot_download as s; "
   f"print(s('openai/clip-vit-large-patch14', allow_patterns=['*.json','*.txt','model.safetensors']))\"")
sh(f"mkdir -p {W}/wheels && {sys.executable} -m pip download -q -d {W}/wheels "
   f"--python-version 3.13 --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 "
   f"--platform manylinux_2_17_x86_64 --only-binary=:all: "
   f"albumentations opencv-python-headless lmdb scikit-learn pyyaml")
sh(f"ls {W}/wheels; du -sh {W}/hf {W}/wheels {W}/dataset_json")
