#!/usr/bin/env python
"""Measure training / inference cost of each adaptation scheme on one GPU.

Uses synthetic 224x224 inputs so it needs no dataset and isolates model cost.
For every entry it reports trainable params, compact checkpoint size, training
step time / throughput / peak memory (forward+backward+Adam step), inference
latency (batch 1) and throughput (batch N), peak inference memory and forward
GFLOPs.

    python tools/benchmark_efficiency.py --out results/efficiency \
        --entry BiasLN=training/config/detector/BiasLN.yaml \
        --entry GenD=training/config/detector/GenD.yaml \
        --entry Effort=training/config/detector/Effort.yaml \
        --entry "LoRA=training/config/detector/BiasLN.yaml:tuning_mode=lora" \
        --entry "FullFT=training/config/detector/BiasLN.yaml:tuning_mode=full,backbone_lr=1e-5"
"""

import argparse
import io
import json
import os
import sys
import time

import torch
import yaml

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))

from config_utils import apply_overrides  # noqa: E402
from detectors import DETECTOR  # noqa: E402


def parse_entry(text):
    name, rest = text.split("=", 1)
    path, _, opts = rest.partition(":")
    return name, path, [o for o in opts.split(",") if o]


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync()
    return (time.perf_counter() - t0) / iters


def forward_gflops(model, x):
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except ImportError:
        return None
    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        model({"image": x}, inference=True)
    return counter.get_total_flops() / x.shape[0] / 1e9


def bench(name, path, opts, args, device):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    apply_overrides(cfg, opts)
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.amp)

    torch.manual_seed(0)
    model = DETECTOR[cfg["model_name"]](cfg).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    buf = io.BytesIO()
    torch.save(model.trainable_state_dict(), buf)
    ckpt_mb = buf.getbuffer().nbytes / 1024 ** 2

    lr = cfg["optimizer"]["adam"]["lr"]
    opt = torch.optim.Adam(model.get_param_groups(lr), lr=lr)
    x = torch.randn(args.train_batch, 3, args.resolution, args.resolution, device=device)
    y = torch.randint(0, 2, (args.train_batch,), device=device)
    y[0], y[1] = 0, 1
    scaler = torch.cuda.amp.GradScaler() if amp_dtype == torch.float16 else None

    def train_step():
        model.train()
        with torch.autocast(device.type, dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
            out = model({"image": x, "label": y})
            loss = model.get_losses({"image": x, "label": y}, out)["overall"]
        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t_train = timed(train_step, args.iters, args.warmup)
    train_mem = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0

    opt.zero_grad(set_to_none=True)
    del opt
    model.eval()
    xb = torch.randn(args.infer_batch, 3, args.resolution, args.resolution, device=device)
    x1 = xb[:1]

    def infer(inp):
        with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype or torch.float32,
                                             enabled=amp_dtype is not None):
            model({"image": inp}, inference=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    t_b1 = timed(lambda: infer(x1), args.iters, args.warmup)
    t_bn = timed(lambda: infer(xb), args.iters, args.warmup)
    infer_mem = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0

    return {
        "name": name, "config": path, "opts": opts,
        "trainable_params": trainable, "total_params": total,
        "trainable_percent": 100.0 * trainable / total,
        "compact_ckpt_mb": ckpt_mb,
        "train_batch": args.train_batch,
        "train_step_ms": t_train * 1e3,
        "train_images_per_s": args.train_batch / t_train,
        "train_peak_mem_gb": train_mem,
        "infer_latency_b1_ms": t_b1 * 1e3,
        "infer_batch": args.infer_batch,
        "infer_images_per_s": args.infer_batch / t_bn,
        "infer_peak_mem_gb": infer_mem,
        "forward_gflops_per_image": forward_gflops(model, x1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", action="append", required=True, help="name=yaml[:k=v,k=v]")
    ap.add_argument("--train_batch", type=int, default=32)
    ap.add_argument("--infer_batch", type=int, default=32)
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--amp", default="none", choices=["none", "fp16", "bf16"])
    ap.add_argument("--out", default="results/efficiency")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    rows = []
    for text in args.entry:
        name, path, opts = parse_entry(text)
        print(f"== {name} ({path} {opts})", flush=True)
        try:
            row = bench(name, path, opts, args, device)
        except torch.cuda.OutOfMemoryError as e:  # e.g. full FT at batch 32 on a T4
            row = {"name": name, "config": path, "opts": opts, "error": f"OOM: {e}".splitlines()[0]}
            torch.cuda.empty_cache()
        print(json.dumps(row, indent=1), flush=True)
        rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump({"gpu": gpu, "amp": args.amp, "torch": torch.__version__, "rows": rows}, f, indent=2)
    hdr = ("| Method | Trainable | Ckpt (MB) | Train ms/step (bs{tb}) | Train img/s | Train mem (GB) "
           "| Infer ms (bs1) | Infer img/s (bs{ib}) | Infer mem (GB) | GFLOPs/img |").format(
        tb=args.train_batch, ib=args.infer_batch)
    lines = [f"GPU: {gpu}, AMP: {args.amp}, torch {torch.__version__}", "", hdr, "|---" * 10 + "|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['name']} | {r['error']} |" + " |" * 8)
            continue
        g = r["forward_gflops_per_image"]
        lines.append(
            f"| {r['name']} | {r['trainable_params']/1e6:.3f}M ({r['trainable_percent']:.3f}%) "
            f"| {r['compact_ckpt_mb']:.1f} | {r['train_step_ms']:.0f} | {r['train_images_per_s']:.1f} "
            f"| {r['train_peak_mem_gb']:.2f} | {r['infer_latency_b1_ms']:.1f} | {r['infer_images_per_s']:.1f} "
            f"| {r['infer_peak_mem_gb']:.2f} | {'-' if g is None else f'{g:.1f}'} |")
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
