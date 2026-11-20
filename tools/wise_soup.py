#!/usr/bin/env python
"""Post-hoc weight-space variants of trained checkpoints (no training, no new loss).

* seed soup  : uniform average of the trainable tensors of several runs that share the
               same pre-trained initialisation (model soups, Wortsman et al. 2022a);
* WiSE-FT    : theta = theta_0 + alpha * (theta_ft - theta_0) for the backbone tensors,
               theta_0 = pre-trained CLIP (Wortsman et al. 2022b). The linear head has no
               zero-shot counterpart and is kept as trained.

alpha is fixed a priori (e.g. 0.5); it is never tuned on a test set.

    python tools/wise_soup.py --detector_path training/config/detector/BiasLN.yaml \
        --ckpts logs/a/ckpt_best.pth logs/b/ckpt_best.pth --alpha 1.0 --out soup.pth \
        [--opts tune_blocks=0-18]
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_state(path, torch):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "state_dict" in ck:
        return ck["state_dict"], bool(ck.get("trainable_only", False))
    return ck, False


def strip(k):
    for p in ("module.",):
        if k.startswith(p):
            k = k[len(p):]
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector_path", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=1.0, help="1.0 = plain soup / fine-tuned weights")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opts", nargs="*", default=[])
    a = ap.parse_args()

    import torch

    states = [load_state(p, torch) for p in a.ckpts]
    keys = set(strip(k) for k in states[0][0])
    for s, _ in states[1:]:
        if set(strip(k) for k in s) != keys:
            raise SystemExit("key mismatch between checkpoints")
    avg = {}
    for k in keys:
        vs = [dict((strip(kk), vv) for kk, vv in s.items())[k] for s, _ in states]
        avg[k] = (torch.stack([v.double() for v in vs]).mean(0).to(vs[0].dtype)
                  if vs[0].is_floating_point() else vs[0])

    if a.alpha != 1.0:
        # pre-trained reference = a freshly built detector (pre-trained CLIP + random head)
        sys.path.insert(0, os.path.join(ROOT, "training"))
        os.chdir(ROOT)
        sys.argv = ["train.py", "--detector_path", os.path.join(ROOT, a.detector_path), "--opts", *a.opts]
        import train as T

        cfg = T.load_config()
        ref = T.DETECTOR[cfg["model_name"]](cfg).state_dict()
        n_interp = 0
        for k, v in avg.items():
            if k.startswith("backbone.") and v.is_floating_point():
                if k not in ref:
                    raise SystemExit(f"no pre-trained value for {k}")
                p0 = ref[k].to(v.dtype)
                avg[k] = p0 + a.alpha * (v - p0)
                n_interp += 1
        print(f"WiSE alpha={a.alpha}: interpolated {n_interp} backbone tensors, kept {len(avg) - n_interp} as trained")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.save({"state_dict": avg, "trainable_only": all(t for _, t in states),
                "ckpt_info": f"soup of {len(a.ckpts)} ckpts, wise alpha={a.alpha}: {a.ckpts}"}, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
