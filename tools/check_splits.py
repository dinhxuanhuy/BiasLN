#!/usr/bin/env python
"""Audit dataset JSONs: split sizes, val/test disjointness and frame paths.

Answers the reviewers' main concern with evidence instead of assertions:
is the data used for checkpoint selection disjoint from every test set?

    python tools/check_splits.py --json_dir preprocessing/dataset_json \
        --val FaceForensics++:val \
        --test Celeb-DF-v2 DeepFakeDetection DFDC DFDCP UADFV \
        --rgb_dir datasets/rgb --check_paths 200 --out results/split_audit.json

Overlap is checked at three levels:
  * frame files   (identical image paths)
  * video folders (identical parent directory of frames)
  * identities    (Celeb-DF ``idX`` tokens; FF++/DF40 numeric source ids)
"""

import argparse
import json
import os
import random
import re
from collections import defaultdict

FF_LIKE = {"FF-DF", "FF-F2F", "FF-FS", "FF-NT", "FaceForensics++", "DeepFakeDetection", "FaceShifter"}


def load_split(json_dir, name, split, compression="c23"):
    with open(os.path.join(json_dir, name + ".json")) as f:
        info = json.load(f)
    key = name if name in info else next(iter(info))
    videos = {}
    available = set()
    for label, parts in info[key].items():
        available |= set(parts.keys())
        if split not in parts:
            continue
        sub = parts[split]
        if name in FF_LIKE and isinstance(sub, dict) and compression in sub:
            sub = sub[compression]
        for vname, vinfo in sub.items():
            frames = [str(p).replace("\\", "/") for p in vinfo.get("frames", [])]
            videos[f"{vinfo.get('label', label)}/{vname}"] = frames
    return videos, sorted(available)


def identities(name, video_dirs):
    ids = set()
    for d in video_dirs:
        base = d.rstrip("/").split("/")[-1]
        if "Celeb" in name:
            ids |= set(re.findall(r"id\d+", base))
        else:
            ids |= set(re.findall(r"\d{3}", base))
    return ids


def summarize(name, split, videos):
    frames = {f for fr in videos.values() for f in fr}
    dirs = {os.path.dirname(f) for f in frames}
    return {"dataset": name, "split": split, "videos": len(videos), "frames": len(frames),
            "_frames": frames, "_dirs": dirs, "_ids": identities(name, dirs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", default="preprocessing/dataset_json")
    ap.add_argument("--val", nargs="+", default=["FaceForensics++:val"], help="name:split")
    ap.add_argument("--test", nargs="+", required=True)
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--rgb_dir", default=None)
    ap.add_argument("--check_paths", type=int, default=0, help="sample N frames per set and stat() them")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sets = []
    for item in args.val:
        name, _, split = item.partition(":")
        v, avail = load_split(args.json_dir, name, split or "val")
        print(f"[val ] {name}:{split or 'val'} -> {len(v)} videos (splits in json: {avail})")
        sets.append(("val", summarize(name, split or "val", v)))
    for name in args.test:
        v, avail = load_split(args.json_dir, name, args.test_split)
        print(f"[test] {name}:{args.test_split} -> {len(v)} videos (splits in json: {avail})")
        sets.append(("test", summarize(name, args.test_split, v)))

    report = {"sets": [{k: v for k, v in s.items() if not k.startswith("_")} for _, s in sets],
              "overlaps": []}
    vals = [s for r, s in sets if r == "val"]
    tests = [s for r, s in sets if r == "test"]
    clean = True
    for a in vals:
        for b in tests:
            o = {"val": f"{a['dataset']}:{a['split']}", "test": f"{b['dataset']}:{b['split']}",
                 "shared_frames": len(a["_frames"] & b["_frames"]),
                 "shared_video_dirs": len(a["_dirs"] & b["_dirs"])}
            if ("Celeb" in a["dataset"]) == ("Celeb" in b["dataset"]):
                o["shared_identities"] = len(a["_ids"] & b["_ids"])
            report["overlaps"].append(o)
            bad = o["shared_frames"] or o["shared_video_dirs"]
            clean &= not bad
            print(("  !! " if bad else "  ok ") + json.dumps(o))
    report["val_test_disjoint"] = clean
    print(f"\nValidation data disjoint from every test set: {clean}")

    if args.check_paths and args.rgb_dir:
        rng = random.Random(0)
        missing = defaultdict(list)
        for _, s in sets:
            sample = rng.sample(sorted(s["_frames"]), min(args.check_paths, len(s["_frames"])))
            for p in sample:
                full = p if p.startswith("/") else os.path.join(args.rgb_dir, p.lstrip("./"))
                if not os.path.exists(full):
                    missing[s["dataset"]].append(full)
            n = len(missing[s["dataset"]])
            print(f"[paths] {s['dataset']}: {len(sample) - n}/{len(sample)} sampled frames exist"
                  + (f"  e.g. missing {missing[s['dataset']][0]}" if n else ""))
        report["missing_examples"] = {k: v[:5] for k, v in missing.items()}

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
