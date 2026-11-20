#!/usr/bin/env python
"""Build DeepfakeBench-format dataset JSONs from extracted frames + OFFICIAL splits.

    python preprocessing/build_json.py --rgb_dir datasets/rgb --out preprocessing/dataset_json

Splits (no split is ever aliased to another; in particular there is no
Celeb-DF "val" that equals its test set, unlike common DeepfakeBench JSONs):

  FaceForensics++ / FF-DF / FF-F2F / FF-FS / FF-NT
      official FF++ pairs: train 360 / val 70 / test 70 pairs
      -> 720 / 140 / 140 real videos and the corresponding fakes, c23
  DeepFakeDetection  all videos (actors real, DFD fakes) -> test
  Celeb-DF-v2        official List_of_testing_videos.txt (518) -> test, rest -> train
  DFDC               public test set (metadata.json labels) -> test
  DFDCP              dataset.json 'set' field -> train / test
  UADFV              all videos -> test
  <m>_ff (DF40)      DF40 FF++-based fakes of method m + FF++ c23 *test* reals -> test

Frame paths are written relative to --rgb_dir, as the data loader expects.
"""

import argparse
import glob
import json
import os
from collections import defaultdict


def frames_of(video_dir, rgb_dir):
    files = [f for f in os.listdir(video_dir) if f.lower().endswith((".png", ".jpg"))]
    files.sort(key=lambda f: int(os.path.splitext(f)[0]) if os.path.splitext(f)[0].isdigit() else f)
    rel = os.path.relpath(video_dir, rgb_dir)
    return [f"{rel}/{f}" for f in files]


def find_dir(root, pattern):
    """First directory under root matching pattern (searched at depths 0..3)."""
    for depth in range(4):
        hits = sorted(glob.glob(os.path.join(root, *(["*"] * depth), pattern)))
        hits = [h for h in hits if os.path.isdir(h)]
        if hits:
            return hits[0]
    return None


def find_file(root, name):
    for depth in range(4):
        hits = glob.glob(os.path.join(root, *(["*"] * depth), name))
        if hits:
            return hits[0]
    return None


def add(store, label, split, vid, frames, compression=None):
    if not frames:
        return 0
    node = store[label][split]
    if compression:
        node = node.setdefault(compression, {})
    node[vid] = {"label": label, "frames": frames}
    return 1


def new_store():
    return defaultdict(lambda: defaultdict(dict))


def plain(store):
    return {lab: {sp: dict(v) for sp, v in splits.items()} for lab, splits in store.items()}


def dump(out_dir, name, store):
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump({name: plain(store)}, f)
    counts = {}
    for lab, splits in store.items():
        for sp, v in splits.items():
            nested = "c23" in v  # FF++-style {split: {compression: {video: ...}}}
            counts[f"{lab}/{sp}"] = len(v["c23"]) if nested else len(v)
    print(f"{name}: {counts}")


# ------------------------------------------------------------------ FF++
FF_METHODS = {"FF-DF": "Deepfakes", "FF-F2F": "Face2Face", "FF-FS": "FaceSwap", "FF-NT": "NeuralTextures"}


def build_ffpp(rgb, out):
    root = os.path.join(rgb, "FaceForensics++")
    yt = find_dir(root, "original_sequences")
    base = os.path.dirname(yt)
    splits = {}
    for sp in ("train", "val", "test"):
        with open(find_file(root, f"{sp}.json")) as f:
            splits[sp] = json.load(f)
    real_dir = os.path.join(base, "original_sequences", "youtube", "c23", "frames")
    reals, fakes = {}, {}
    for sp, pairs in splits.items():
        reals[sp] = sorted({v for p in pairs for v in p})
        fakes[sp] = sorted({f"{a}_{b}" for a, b in pairs} | {f"{b}_{a}" for a, b in pairs})

    full = new_store()
    per_method = {k: new_store() for k in FF_METHODS}
    for sp in splits:
        for v in reals[sp]:
            fr = frames_of(os.path.join(real_dir, v), rgb) if os.path.isdir(os.path.join(real_dir, v)) else []
            add(full, "FF-real", sp, v, fr, "c23")
            for st in per_method.values():
                add(st, "FF-real", sp, v, fr, "c23")
        for label, method in FF_METHODS.items():
            mdir = os.path.join(base, "manipulated_sequences", method, "c23", "frames")
            for v in fakes[sp]:
                d = os.path.join(mdir, v)
                if os.path.isdir(d):
                    fr = frames_of(d, rgb)
                    add(full, label, sp, v, fr, "c23")
                    add(per_method[label], label, sp, v, fr, "c23")
    dump(out, "FaceForensics++", full)
    for label, st in per_method.items():
        dump(out, label, st)

    # FaceShifter (FF-FH) on the same official split, for completeness.
    fh = new_store()
    mdir = os.path.join(base, "manipulated_sequences", "FaceShifter", "c23", "frames")
    for sp in splits:
        for v in reals[sp]:
            d = os.path.join(real_dir, v)
            if os.path.isdir(d):
                add(fh, "FF-real", sp, v, frames_of(d, rgb), "c23")
        for v in fakes[sp]:
            d = os.path.join(mdir, v)
            if os.path.isdir(d):
                add(fh, "FF-FH", sp, v, frames_of(d, rgb), "c23")
    dump(out, "FaceShifter", fh)

    # DeepFakeDetection: all videos for testing.
    dfd = new_store()
    for label, sub in (("DFD_real", "original_sequences/actors"), ("DFD_fake", "manipulated_sequences/DeepFakeDetection")):
        d = os.path.join(base, sub, "c23", "frames")
        for v in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            add(dfd, label, "test", v, frames_of(os.path.join(d, v), rgb), "c23")
    dump(out, "DeepFakeDetection", dfd)
    return {sp: [os.path.join(real_dir, v) for v in reals[sp]] for sp in reals}


# ----------------------------------------------------------- Celeb-DF-v2
def build_cdf(rgb, out):
    root = os.path.join(rgb, "Celeb-DF-v2")
    lst = find_file(root, "List_of_testing_videos.txt")
    base = os.path.dirname(lst)
    test = set()
    for line in open(lst):
        parts = line.split()
        if len(parts) == 2:
            test.add(os.path.splitext(parts[1])[0])  # e.g. YouTube-real/00170
    st = new_store()
    for folder, label in (("Celeb-real", "CelebDFv2_real"), ("YouTube-real", "CelebDFv2_real"),
                          ("Celeb-synthesis", "CelebDFv2_fake")):
        d = os.path.join(base, folder, "frames")
        for v in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            sp = "test" if f"{folder}/{v}" in test else "train"
            add(st, label, sp, v, frames_of(os.path.join(d, v), rgb))
    dump(out, "Celeb-DF-v2", st)


# ------------------------------------------------------------------ DFDC
def build_dfdc(rgb, out):
    meta_path = find_file(os.path.join(rgb, "DFDC"), "metadata.json")
    meta = json.load(open(meta_path))
    fdir = os.path.join(os.path.dirname(meta_path), "frames")
    st = new_store()
    for name, info in meta.items():
        v = os.path.splitext(name)[0]
        d = os.path.join(fdir, v)
        if os.path.isdir(d):
            add(st, "DFDC_Fake" if int(info["is_fake"]) else "DFDC_Real", "test", v, frames_of(d, rgb))
    dump(out, "DFDC", st)


# ----------------------------------------------------------------- DFDCP
def build_dfdcp(rgb, out):
    meta_path = find_file(os.path.join(rgb, "DFDCP"), "dataset.json")
    base = os.path.dirname(meta_path)
    meta = json.load(open(meta_path))
    st = new_store()
    for key, info in meta.items():
        top = key.split("/")[0]                      # method_A | method_B | original_videos
        v = os.path.splitext(os.path.basename(key))[0]
        d = os.path.join(base, top, "frames", v)
        if not os.path.isdir(d):
            continue
        if info["label"] == "real":
            label = "DFDCP_Real"
        else:
            label = "DFDCP_FakeA" if top == "method_A" else "DFDCP_FakeB"
        add(st, label, info["set"], v, frames_of(d, rgb))
    dump(out, "DFDCP", st)


# ----------------------------------------------------------------- UADFV
def build_uadfv(rgb, out):
    root = os.path.join(rgb, "UADFV")
    st = new_store()
    for sub, label in (("real", "UADFV_Real"), ("fake", "UADFV_Fake")):
        hits = sorted(glob.glob(os.path.join(root, "**", sub, "frames"), recursive=True))
        d = hits[0] if hits else None
        for v in sorted(os.listdir(d)) if d else []:
            add(st, label, "test", v, frames_of(os.path.join(d, v), rgb))
    dump(out, "UADFV", st)


# ------------------------------------------------------------------ DF40
def build_df40(rgb, out, ff_test_real_dirs):
    for mdir in sorted(glob.glob(os.path.join(rgb, "DF40", "*"))):
        m = os.path.basename(mdir)
        ff = None
        for cand in glob.glob(os.path.join(mdir, "**", "ff", "frames"), recursive=True):
            ff = cand
            break
        if ff is None:
            print(f"[warn] DF40 {m}: no ff/frames")
            continue
        st = new_store()
        for v in sorted(os.listdir(ff)):
            add(st, f"{m}_Fake", "test", v, frames_of(os.path.join(ff, v), rgb))
        for d in ff_test_real_dirs:
            if os.path.isdir(d):
                add(st, f"{m}_Real", "test", os.path.basename(d), frames_of(d, rgb))
        dump(out, f"{m}_ff", st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb_dir", default="datasets/rgb")
    ap.add_argument("--out", default="preprocessing/dataset_json")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rgb = args.rgb_dir
    ff_reals = build_ffpp(rgb, args.out) if os.path.isdir(os.path.join(rgb, "FaceForensics++")) else None
    for name, fn in (("Celeb-DF-v2", build_cdf), ("DFDC", build_dfdc), ("DFDCP", build_dfdcp), ("UADFV", build_uadfv)):
        if os.path.isdir(os.path.join(rgb, name)):
            fn(rgb, args.out)
    if ff_reals and os.path.isdir(os.path.join(rgb, "DF40")):
        build_df40(rgb, args.out, ff_reals["test"])


if __name__ == "__main__":
    main()
