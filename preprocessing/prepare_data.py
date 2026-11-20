#!/usr/bin/env python
"""Copy dataset archives from Google Drive to local disk and extract frames only.

    python preprocessing/prepare_data.py --drive /content/drive/MyDrive --dst datasets/rgb

Layout produced (rgb_dir = --dst):
    <dst>/FaceForensics++/...      (FF++ c23 incl. DeepFakeDetection + actors, split jsons)
    <dst>/Celeb-DF-v2/...          (+ List_of_testing_videos.txt)
    <dst>/DFDC/..., <dst>/DFDCP/..., <dst>/UADFV/...
    <dst>/DF40/<method>/...
Only ``*/frames/*`` images and split/metadata files are extracted (landmarks and
masks are not used by BiasLN), which cuts disk use and time by about 3x.
Re-running skips archives whose ``.extracted`` marker exists.
"""

import argparse
import os
import shutil
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor

RGB = ["FaceForensics++", "Celeb-DF-v2", "DFDC", "DFDCP", "UADFV"]
DF40 = ["uniface", "blendface", "e4s", "facedancer", "fsgan", "inswap", "simswap"]
KEEP_EXT = (".json", ".txt", ".csv")


def keep(name: str) -> bool:
    if name.endswith("/"):
        return False
    return "/frames/" in "/" + name or name.lower().endswith(KEEP_EXT)


def one(job):
    src, dst, tmp = job
    marker = os.path.join(dst, ".extracted")
    if os.path.exists(marker):
        return f"skip {dst}"
    t0 = time.time()
    os.makedirs(tmp, exist_ok=True)
    local = os.path.join(tmp, os.path.basename(src))
    shutil.copyfile(src, local)  # sequential read from Drive is much faster than random access
    t1 = time.time()
    n = 0
    with zipfile.ZipFile(local) as z:
        for info in z.infolist():
            if keep(info.filename):
                z.extract(info, dst)
                n += 1
    os.remove(local)
    open(marker, "w").write(f"{n} files from {src}\n")
    return f"{os.path.basename(src)}: copy {t1 - t0:.0f}s, extract {time.time() - t1:.0f}s, {n} files"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", default="/content/drive/MyDrive")
    ap.add_argument("--dst", default="datasets/rgb")
    ap.add_argument("--tmp", default="/content/zips")
    ap.add_argument("--rgb", nargs="*", default=RGB)
    ap.add_argument("--df40", nargs="*", default=DF40)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs = []
    for name in args.rgb:
        src = os.path.join(args.drive, "rgb", f"{name}.zip")
        if os.path.exists(src):
            jobs.append((src, os.path.join(args.dst, name), args.tmp))
        else:
            print(f"[warn] missing {src}")
    for m in args.df40:
        src = os.path.join(args.drive, "DF40", f"{m}.zip")
        if not os.path.exists(src):
            src = os.path.join(args.drive, "df40", f"{m}.zip")
        if os.path.exists(src):
            jobs.append((src, os.path.join(args.dst, "DF40", m), args.tmp))
        else:
            print(f"[warn] missing DF40 {m}")
    # Largest first so the long FF++ archive overlaps with the small ones.
    jobs.sort(key=lambda j: -os.path.getsize(j[0]))
    with ProcessPoolExecutor(args.workers) as ex:
        for msg in ex.map(one, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
