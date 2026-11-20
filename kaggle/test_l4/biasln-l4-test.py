# Test-only pass over the 16 checkpoints trained by biasln-l4-seeds (its test step crashed on an
# empty UADFV.json). One lane per GPU. Same test sets and options as scripts/run_protocol.sh.
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

ck_tgz = find("/kaggle/input/**/l4_ckpts.tgz")
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
os.remove(f"{REPO}/preprocessing/dataset_json/UADFV.json")  # empty on Kaggle (no UADFV frames)
os.makedirs("/tmp/ck", exist_ok=True)
sh(f"tar -xzf {ck_tgz} -C /tmp/ck", check=True)
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
TESTS = ("Celeb-DF-v2 DeepFakeDetection DFDC DFDCP uniface_ff blendface_ff e4s_ff facedancer_ff fsgan_ff "
         "inswap_ff simswap_ff FaceForensics++ FF-DF FF-F2F FF-FS FF-NT")
E = {"biasln": ("BiasLN", ""), "gend": ("GenD", ""), "glr3": ("BiasLN", "ln_gamma_lr_scale=3"),
     "ln_l2": ("BiasLN", "tuning_mode=ln"), "bitfit": ("BiasLN", "tuning_mode=bitfit"),
     "lora": ("BiasLN", "tuning_mode=lora"), "linprobe": ("BiasLN", "tuning_mode=frozen"),
     "bitfit_nol2": ("BiasLN", "tuning_mode=bitfit l2_norm=false"),
     "ln_nol2": ("BiasLN", "tuning_mode=ln l2_norm=false")}
LANES = {0: ["biasln_s3", "gend_s3", "glr3_s3", "ln_l2_s1"],
         1: ["biasln_s4", "gend_s4", "glr3_s4", "ln_l2_s2"],
         2: ["bitfit_s3", "lora_s1", "linprobe_s1", "bitfit_nol2_s1024"],
         3: ["bitfit_s4", "lora_s2", "linprobe_s2", "ln_nol2_s1024"]}
procs = []
for gpu, runs in LANES.items():
    cmds = []
    for run in runs:
        tag, seed = run.rsplit("_s", 1)
        cfg, opts = E[tag]
        res = f"{OUTW}/results/anchor_val/{tag}/seed{seed}"
        ck = f"/tmp/ck/logs/{run}"
        cmds.append(f"echo '== $(date +%T) test {run}'; mkdir -p {res}; python training/test.py "
                    f"--detector_path training/config/detector/{cfg}.yaml --weights_path {ck}/ckpt_best.pth "
                    f"--test_dataset {TESTS} --out_dir {res} --amp bf16 --opts {opts} workers=10 "
                    f"&& cp {ck}/run_info.json {ck}/config.yaml {res}/ || echo '!! test failed: {run}'")
    log = open(f"{OUTW}/lane{gpu}.log", "w")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    procs.append(subprocess.Popen(f"cd {REPO} && " + " ; ".join(cmds) + f"; echo '== lane {gpu} finished'",
                                  shell=True, stdout=log, stderr=subprocess.STDOUT, env=env))
    time.sleep(60)
while any(p.poll() is None for p in procs):
    time.sleep(300)
    print(f"[{(time.time()-T0)/3600:.2f} h] lanes running: {[i for i, p in enumerate(procs) if p.poll() is None]}", flush=True)
    if time.time() - T0 > 11.3 * 3600:
        for p in procs: p.terminate()
        break
sh(f"du -sh {OUTW}; ls {OUTW}/results/anchor_val/*; grep -h 'test failed' {OUTW}/lane*.log")
print(f"done in {(time.time()-T0)/3600:.2f} h", flush=True)
