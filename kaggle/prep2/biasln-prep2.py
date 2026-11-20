# Fix-up of the biasln-prep output: rebuilds the test split of DFDC and of the 7 DF40 sets,
# whose frame paths did not resolve, and copies everything else (JSONs, CLIP cache, wheels).
# The video folder is located by a breadth-first search over directory names only
# (folders with many entries are not descended into), so the frames are never walked.
import collections, glob, json, os, re, shutil, time

W = "/kaggle/working"
T0 = time.time()

def first(pattern_tail):
    for d in ("*", "*/*", "*/*/*", "*/*/*/*"):
        h = sorted(glob.glob(f"/kaggle/input/{d}/{pattern_tail}"))
        if h:
            return h[0]
    raise SystemExit(f"not found: {pattern_tail}; /kaggle/input={os.listdir('/kaggle/input')}")

PREP = os.path.dirname(first("prep_report.json"))
DFB = os.path.dirname(first("deepfakebench/dataset_json"))
DF40 = os.path.dirname(first("deepfake40/dataset_json"))
ROOTS = {"DFB": DFB, "DF40": DF40}
print("inputs", PREP, DFB, DF40, flush=True)

for sub in ("dataset_json", "hf", "wheels"):
    shutil.copytree(f"{PREP}/{sub}", f"{W}/{sub}", dirs_exist_ok=True)
report = json.load(open(f"{PREP}/prep_report.json"))
print(f"copied prep output in {time.time()-T0:.0f}s", flush=True)

def parts(p):
    return [x for x in re.split(r"[\\/]+", p) if x]

def find_parent(start, video, max_depth=6, max_entries=3000):
    """Directory under `start` that holds a sub-folder named `video` (BFS on directory names)."""
    q = collections.deque([(start, 0)])
    seen = 0
    while q:
        d, k = q.popleft()
        if os.path.isdir(os.path.join(d, video)):
            return d
        if k >= max_depth:
            continue
        try:
            ents = [e for e in os.scandir(d) if e.is_dir()]
        except OSError:
            continue
        seen += 1
        if len(ents) > max_entries or seen > 4000:
            continue
        q.extend((e.path, k + 1) for e in sorted(ents, key=lambda e: e.name))
    return None

def fkey(p):
    s = os.path.splitext(os.path.basename(p))[0]
    return (0, int(s)) if s.isdigit() else (1, s)

def tagged(abs_path):
    for tag, root in ROOTS.items():
        if abs_path.startswith(root + "/"):
            return f"{tag}/{abs_path[len(root) + 1:]}"
    raise ValueError(abs_path)

def rebuild_test(name, src, start_dirs, relabel=None):
    """Rewrite the test split of `name` from the original JSON `src`."""
    orig = json.load(open(src))
    otop = list(orig)[0]
    out = json.load(open(f"{W}/dataset_json/{name}.json"))
    top = list(out)[0]
    cache = {}   # tuple(original parent parts) -> absolute parent dir
    nv = nf = miss = 0
    for label, splits in orig[otop].items():
        new_label = relabel(label) if relabel else label
        items = splits.get("test", {})
        if isinstance(items, dict) and set(items) & {"c23", "c40"}:
            items = items.get("c23", {})
        dst = out[top].setdefault(new_label, {})
        dst["test"] = {}
        for vid, info in items.items():
            if not isinstance(info, dict) or not isinstance(info.get("frames"), list) or not info["frames"]:
                continue
            fr = sorted(info["frames"], key=fkey)
            p0 = parts(fr[0])
            video, key = p0[-2], tuple(p0[:-2])
            base = cache.get(key)
            if base is None or not os.path.isdir(os.path.join(base, video)):
                base = None
                for s in start_dirs:
                    base = find_parent(s, video)
                    if base:
                        break
                if base is None:
                    miss += len(fr)
                    continue
                cache[key] = base
                print(f"  {name} {label}: {'/'.join(key)} -> {base}", flush=True)
            frames = [tagged(os.path.join(base, video, parts(f)[-1])) for f in fr]
            last = os.path.join(ROOTS[frames[-1].split('/')[0]], frames[-1].split('/', 1)[1])
            if not os.path.exists(last):
                miss += 1
            k = vid if vid not in dst["test"] else f"{new_label}/{vid}"
            dst["test"][k] = {"label": new_label, "frames": frames}
            nv += 1; nf += len(frames)
    json.dump(out, open(f"{W}/dataset_json/{name}.json", "w"))
    sp = report.get(name, {}).get("splits", {}) if isinstance(report.get(name), dict) else {}
    sp["test"] = [nv, nf]
    report[name] = {"splits": sp, "missing_frames": miss, "rebuilt_test": True}
    print(f"{name}: test {nv} videos {nf} frames, missing {miss} ({time.time()-T0:.0f}s)", flush=True)

def rgb_dirs(token):
    r = f"{DFB}/rgb"
    return [os.path.join(r, d) for d in sorted(os.listdir(r)) if token.lower() in d.lower()] + [r]

rebuild_test("DFDC", f"{DFB}/dataset_json/DFDC.json", rgb_dirs("DFDC"))
df40_json = os.path.dirname(first("deepfake40/dataset_json/uniface_ff.json"))
lab = lambda l: "FF-real" if l.lower().endswith("real") else "DF_fake"
for m in ["uniface", "blendface", "e4s", "facedancer", "fsgan", "inswap", "simswap"]:
    starts = [f"{DF40}/dataset/{m}", f"{DFB}/rgb/FaceForensics++", f"{DF40}/dataset", DFB]
    rebuild_test(f"{m}_ff", f"{df40_json}/{m}_ff.json", starts, relabel=lab)

json.dump(report, open(f"{W}/prep_report.json", "w"), indent=1)
print(json.dumps({k: v for k, v in report.items()}, indent=0)[:3000])
print(f"done in {time.time()-T0:.0f}s", flush=True)
