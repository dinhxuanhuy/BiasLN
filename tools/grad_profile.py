#!/usr/bin/env python
"""Gradient / Fisher profile of the bias and LayerNorm parameters, per encoder block and role.

Uses only FF++ *training* batches, so it can guide where to tune without touching any test set.
For every trainable tensor we accumulate over N batches
    fisher  = mean_b  g^2        (empirical Fisher, per parameter)
    gabs    = mean_b |g|
    rel     = ||mean_b g|| / ||theta||    (relative size of the average update direction)
and aggregate them per (block, role). The classifier head is first fitted for a few steps with
the backbone frozen, so that backbone gradients are not dominated by a random head.

    python tools/grad_profile.py --detector_path training/config/detector/BiasLN.yaml \
        --batches 50 --warmup 100 --out results/grad_profile/biasln_init.json \
        [--weights logs/.../ckpt_best.pth]  [--opts tuning_mode=bitfit_ln workers=8]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ROLE_RULES = [(r"q_proj\.bias$", "q"), (r"k_proj\.bias$", "k"), (r"v_proj\.bias$", "v"),
              (r"out_proj\.bias$", "out"), (r"fc1\.bias$", "fc1"), (r"fc2\.bias$", "fc2"),
              (r"layer_norm1\.weight$", "ln1_gamma"), (r"layer_norm1\.bias$", "ln1_beta"),
              (r"layer_norm2\.weight$", "ln2_gamma"), (r"layer_norm2\.bias$", "ln2_beta")]
ROLES = [r for _, r in ROLE_RULES]


def where(name):
    m = re.search(r"layers\.(\d+)\.", name)
    block = int(m.group(1)) if m else None
    for pat, r in ROLE_RULES:
        if re.search(pat, name):
            return block, r
    return block, ("outer_ln" if re.search(r"layrnorm|layernorm", name) else "other")


def to_device(d, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in d.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector_path", required=True)
    ap.add_argument("--weights", default=None, help="optional trained checkpoint to profile instead of init")
    ap.add_argument("--batches", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=100, help="head-only steps before profiling (init only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--opts", nargs="*", default=[])
    a = ap.parse_args()

    # reuse train.py's config loading and data pipeline
    sys.path.insert(0, os.path.join(ROOT, "training"))
    os.chdir(ROOT)  # dataset json paths in the configs are relative to the repo root
    sys.argv = ["train.py", "--detector_path", os.path.join(ROOT, a.detector_path), "--seed", "1024",
                "--opts", "tuning_mode=bitfit_ln", *a.opts]
    import torch
    import train as T

    cfg = T.load_config()
    T.init_seed(cfg)
    loader = T.prepare_training_data(cfg)
    model = T.DETECTOR[cfg["model_name"]](cfg)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.weights:
        sd = torch.load(a.weights, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        print(model.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()}, strict=False))
    model.to(dev).train()
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    back = {n: p for n, p in params.items() if not n.startswith("head")}
    head = [p for n, p in params.items() if n.startswith("head")]
    it = iter(loader)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    if not a.weights and a.warmup > 0:  # fit the random head first
        for p in back.values():
            p.requires_grad_(False)
        opt = torch.optim.Adam(head, lr=1e-3)
        for _ in range(a.warmup):
            d = to_device(next_batch(), dev)
            loss = model.get_losses(d, model(d))["overall"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        for p in back.values():
            p.requires_grad_(True)

    fisher = {n: torch.zeros_like(p) for n, p in back.items()}
    gsum = {n: torch.zeros_like(p) for n, p in back.items()}
    gabs = {n: torch.zeros_like(p) for n, p in back.items()}
    for _ in range(a.batches):
        d = to_device(next_batch(), dev)
        model.zero_grad(set_to_none=True)
        model.get_losses(d, model(d))["overall"].backward()
        for n, p in back.items():
            if p.grad is not None:
                g = p.grad.detach().float()
                fisher[n] += g * g
                gsum[n] += g
                gabs[n] += g.abs()

    agg = defaultdict(lambda: {"numel": 0, "fisher_sum": 0.0, "gabs_sum": 0.0, "gmean_sq": 0.0, "theta_sq": 0.0})
    for n, p in back.items():
        b, r = where(n)
        key = f"{'outer' if b is None else b}|{r}"
        e = agg[key]
        e["numel"] += p.numel()
        e["fisher_sum"] += float(fisher[n].sum()) / a.batches
        e["gabs_sum"] += float(gabs[n].sum()) / a.batches
        e["gmean_sq"] += float((gsum[n] / a.batches).pow(2).sum())
        e["theta_sq"] += float(p.detach().float().pow(2).sum())
    rows = []
    for key, e in agg.items():
        b, r = key.split("|")
        rows.append({"block": b, "role": r, "numel": e["numel"],
                     "fisher_per_param": e["fisher_sum"] / e["numel"],
                     "gabs_per_param": e["gabs_sum"] / e["numel"],
                     "rel_update": (e["gmean_sq"] ** 0.5) / max(e["theta_sq"] ** 0.5, 1e-12),
                     "fisher_total": e["fisher_sum"]})
    per_block = defaultdict(float)
    for r_ in rows:
        per_block[r_["block"]] += r_["fisher_total"]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"weights": a.weights, "batches": a.batches, "rows": rows,
               "fisher_total_per_block": dict(per_block)}, open(a.out, "w"), indent=1)
    print("wrote", a.out)
    top = sorted(per_block.items(), key=lambda kv: -kv[1])
    print("blocks by total Fisher:", [k for k, _ in top])


if __name__ == "__main__":
    main()
