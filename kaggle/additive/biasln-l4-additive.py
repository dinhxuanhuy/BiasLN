# Additive PEFT branch (AdaptFormer-style adapter, VPT-deep; seeds 1024/1/2) and GenD seeds 6-9,
# same protocol as all other runs. Also writes the FF++ directory layout (mask availability for ForAda).
# Kaggle L4x4 (internet OFF). One lane per GPU.
# Inputs: xuanhuydinh/biasln-code (code tarball), kernel output xuanhuydinh/biasln-prep2
# (JSONs, HF cache with CLIP ViT-L/14, wheels), datasets deepfakebench and deepfake40.
# Protocol identical to the Colab runs: FF++ c23, 8 frames/video, 10 epochs, bf16,
# checkpoint selected on FF++ val, test on 4 cross-dataset + 7 DF40 sets + FF++.
# Outputs (in /kaggle/working/out): results/anchor_val/<tag>/seed<k>/{metrics.json,scores/,run_info.json},
# logs/<run>/{training.log,run_info.json,ckpt_best.pth (trainable tensors only)}, lane logs.
import glob, os, shutil, subprocess, sys, time

T0 = time.time()
OUTW = "/kaggle/working/out"
REPO = "/tmp/BiasLN"
os.makedirs(OUTW, exist_ok=True)

def sh(c, check=False):
    print("$", c, flush=True)
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], f"[rc={r.returncode} t={time.time()-T0:.0f}s]", flush=True)
    if check and r.returncode:
        raise SystemExit(f"failed: {c}")
    return r

def find(pattern):
    # depth-limited (never a recursive walk: the datasets hold millions of frames)
    tail = pattern.split("/**/", 1)[1]
    hits = sorted(h for d in ("*", "*/*", "*/*/*", "*/*/*/*") for h in glob.glob(f"/kaggle/input/{d}/{tail}"))
    if not hits:
        raise SystemExit(f"not found: {pattern}\n/kaggle/input: {os.listdir('/kaggle/input')}")
    return hits[0]

code_tgz = find("/kaggle/input/**/biasln_code.tgz")
prep = os.path.dirname(find("/kaggle/input/**/prep_report.json"))
DFB = os.path.dirname(find("/kaggle/input/**/deepfakebench/dataset_json"))
DF40 = os.path.dirname(os.path.dirname(find("/kaggle/input/**/deepfake40/dataset/uniface")))
print("inputs", code_tgz, prep, DFB, DF40, flush=True)

# ---- code, packages, data links, CLIP cache
os.makedirs(REPO, exist_ok=True)
sh(f"tar -xzf {code_tgz} -C {REPO}", check=True)
need = []
for mod, pkg in [("albumentations", "albumentations"), ("cv2", "opencv-python-headless"),
                 ("lmdb", "lmdb"), ("sklearn", "scikit-learn"), ("yaml", "pyyaml")]:
    if subprocess.run([sys.executable, "-c", f"import {mod}"], capture_output=True).returncode:
        need.append(pkg)
if need:
    sh(f"{sys.executable} -m pip install -q --no-index --find-links {prep}/wheels {' '.join(need)}", check=True)
os.makedirs(f"{REPO}/datasets/rgb", exist_ok=True)
for tag, root in (("DFB", DFB), ("DF40", DF40)):
    link = f"{REPO}/datasets/rgb/{tag}"
    if not os.path.islink(link):
        os.symlink(root, link)
shutil.copytree(f"{prep}/dataset_json", f"{REPO}/preprocessing/dataset_json", dirs_exist_ok=True)
uadfv = f"{REPO}/preprocessing/dataset_json/UADFV.json"
if os.path.exists(uadfv):
    os.remove(uadfv)  # empty on Kaggle (no UADFV frames); run_protocol.sh skips missing JSONs
# CLIP from the biasln-clip kernel (plain files): the HF cache in the prep output lost its symlinks
clip_dir = os.path.dirname(find("/kaggle/input/**/clip-vit-large-patch14/config.json"))
for y in glob.glob(f"{REPO}/training/config/detector/*.yaml"):
    t = open(y).read()
    if "openai/clip-vit-large-patch14" in t:
        open(y, "w").write(t.replace("openai/clip-vit-large-patch14", clip_dir))
print("CLIP", clip_dir, sorted(os.listdir(clip_dir)), flush=True)
os.environ.update(HF_HOME="/tmp/hf", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                  TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; free -g | head -2")
sh(f"cat {prep}/prep_report.json")

# ---- smoke test: build the model, read a few frames of every test set
smoke = sh(f"cd {REPO} && python training/test.py --detector_path training/config/detector/BiasLN.yaml "
           f"--pretrained --test_dataset Celeb-DF-v2 DFDCP uniface_ff FaceForensics++ --max_samples 64 "
           f"--out_dir /tmp/smoke --amp bf16 --opts workers=4")
if smoke.returncode:
    shutil.copytree(f"{REPO}/preprocessing/dataset_json", f"{OUTW}/dataset_json_debug", dirs_exist_ok=True)
    raise SystemExit("smoke test failed")

# ---- four lanes, one per GPU
# FF++ layout probe (depth-limited; never walks frame folders)
ff = f"{DFB}/rgb/FaceForensics++"
with open(f"{OUTW}/ff_layout.txt", "w") as fh:
    def probe(path, depth):
        try:
            entries = sorted(os.listdir(path))
        except Exception as e:
            fh.write(f"{path}: {e}\n"); return
        fh.write(f"{path}: {len(entries)} entries, e.g. {entries[:6]}\n")
        if depth < 4 and len(entries) <= 40:
            for e in entries:
                if os.path.isdir(f"{path}/{e}"):
                    probe(f"{path}/{e}", depth + 1)
    probe(ff, 0)
print(open(f"{OUTW}/ff_layout.txt").read()[:4000], flush=True)

LANES = {0: [("adapter=BiasLN:tuning_mode=adapter", "1024 1 2")],
         1: [("vpt=BiasLN:tuning_mode=vpt", "1024 1 2")],
         2: [("gend=GenD", "6 7")],
         3: [("gend=GenD", "8 9")]}
print("lanes", LANES, flush=True)
procs = []
for gpu, jobs in LANES.items():
    cmds = " ; ".join(
        f"CROSS_DATASET='Celeb-DF-v2 DeepFakeDetection DFDC DFDCP' ENTRIES='{e}' SEEDS='{s}' AMP=bf16 SELECTION=val SOUP_FROM= OUT={OUTW}/results/anchor_val "
        f"LOGS=/tmp/logs EXTRA_OPTS='workers=10' bash scripts/run_protocol.sh" for e, s in jobs)
    log = open(f"{OUTW}/lane{gpu}.log", "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    procs.append(subprocess.Popen(f"cd {REPO} && {cmds}; echo '== lane {gpu} finished'",
                                  shell=True, stdout=log, stderr=subprocess.STDOUT, env=env))
    time.sleep(90)  # stagger CLIP loading / dataset indexing

def keep_artifacts():
    for run in glob.glob("/tmp/logs/*"):
        dst = f"{OUTW}/logs/{os.path.basename(run)}"
        os.makedirs(dst, exist_ok=True)
        for f in ("training.log", "run_info.json", "config.yaml"):
            if os.path.exists(f"{run}/{f}"):
                shutil.copy(f"{run}/{f}", dst)
        ck = f"{run}/test/FaceForensics++/ckpt_best.pth"
        if os.path.exists(ck):
            shutil.copy(ck, dst)

while any(p.poll() is None for p in procs):
    time.sleep(600)
    keep_artifacts()
    print(f"[{(time.time()-T0)/3600:.2f} h] lanes running: "
          f"{[i for i, p in enumerate(procs) if p.poll() is None]}", flush=True)
    if time.time() - T0 > 11.3 * 3600:   # leave time to save before the 12 h limit
        print("time budget reached; stopping lanes (finished runs are kept)", flush=True)
        for p in procs:
            p.terminate()
        break
keep_artifacts()
sh(f"du -sh {OUTW}; ls {OUTW}/results/anchor_val/*")
print(f"done in {(time.time()-T0)/3600:.2f} h", flush=True)
