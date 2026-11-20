#!/usr/bin/env python
"""Uniform weight averaging of per-epoch checkpoints (selection-free "soup").

Averaging the trained tensors of the last epochs removes the need to pick a
checkpoint at all (no validation data, no target data), and weight averaging
is known to land in flatter minima that transfer better out of domain
(SWA, Izmailov et al. 2018; model soups, Wortsman et al. 2022).

    python tools/average_ckpts.py --run logs/val/biasln_s1024 --from_epoch 5 \
        --out logs/val/biasln_s1024/soup_e5-9.pth

Works on compact checkpoints (trainable tensors only) and on full state dicts.
"""

import argparse
import glob
import os
import re

import torch


def load_state(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"], ckpt.get("trainable_only", False)
    return ckpt, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run dir containing epochs/ckpt_epochXX.pth")
    ap.add_argument("--ckpts", nargs="*", default=None, help="explicit checkpoint list")
    ap.add_argument("--from_epoch", type=int, default=5, help="first epoch included (with --run)")
    ap.add_argument("--to_epoch", type=int, default=None, help="last epoch included (with --run)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    paths = args.ckpts
    if not paths:
        paths = []
        for p in sorted(glob.glob(os.path.join(args.run, "epochs", "ckpt_epoch*.pth"))):
            ep = int(re.search(r"epoch(\d+)", p).group(1))
            if ep >= args.from_epoch and (args.to_epoch is None or ep <= args.to_epoch):
                paths.append(p)
    if not paths:
        raise SystemExit("no checkpoints to average")

    avg, compact = None, None
    for p in paths:
        state, trainable_only = load_state(p)
        compact = trainable_only if compact is None else compact and trainable_only
        if avg is None:
            avg = {k: v.detach().clone().double() if v.is_floating_point() else v.clone()
                   for k, v in state.items()}
        else:
            if set(state) != set(avg):
                raise SystemExit(f"key mismatch in {p}")
            for k, v in state.items():
                if v.is_floating_point():
                    avg[k] += v.double()
    n = len(paths)
    ref, _ = load_state(paths[0])
    out = {k: (v / n).to(ref[k].dtype) if v.is_floating_point() else v for k, v in avg.items()}
    payload = {"state_dict": out, "trainable_only": bool(compact),
               "ckpt_info": f"uniform average of {n} checkpoints: {[os.path.basename(p) for p in paths]}"}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(payload, args.out)
    print(f"averaged {n} checkpoints -> {args.out}")


if __name__ == "__main__":
    main()
