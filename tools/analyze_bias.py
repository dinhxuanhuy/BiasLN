"""How far did each parameter *type* move from its CLIP init?

Groups trainable tensors of a compact checkpoint by role inside a ViT block and
reports ||delta|| and ||delta||/||init|| per group (and per layer with --per_layer).
Roles: q/k/v/out biases, fc1 (pre-GELU, gating) / fc2 biases, LN gamma/beta,
embedding/post-LN. k-bias has an analytically zero gradient (softmax is
invariant to a constant added to every key), so its delta must be ~0.

    python tools/analyze_bias.py --ckpt logs/val/biasln_s1024/epochs/ckpt_epoch07.pth
"""
import argparse, json, re, sys
from collections import defaultdict
import torch

def role(name):
    rules = [(r"q_proj\.bias", "attn.q.bias"), (r"k_proj\.bias", "attn.k.bias"),
             (r"v_proj\.bias", "attn.v.bias"), (r"out_proj\.bias", "attn.out.bias"),
             (r"fc1\.bias", "mlp.fc1.bias"), (r"fc2\.bias", "mlp.fc2.bias"),
             (r"layer_norm\d\.weight", "ln.gamma"), (r"layer_norm\d\.bias", "ln.beta"),
             (r"(pre_layrnorm|post_layernorm)\.weight", "ln_outer.gamma"),
             (r"(pre_layrnorm|post_layernorm)\.bias", "ln_outer.beta"),
             (r"embeddings", "embed"), (r"^head", "head")]
    for pat, r in rules:
        if re.search(pat, name):
            return r
    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--clip", default="openai/clip-vit-large-patch14")
    ap.add_argument("--per_layer", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    from transformers import CLIPModel
    init = {k: v.float() for k, v in CLIPModel.from_pretrained(a.clip).vision_model.state_dict().items()}
    report = {}
    for path in a.ckpt:
        sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        agg = defaultdict(lambda: [0.0, 0.0, 0]); layer = defaultdict(lambda: [0.0, 0.0])
        for k, v in sd.items():
            r = role(k)
            if r == "head":
                continue
            key = re.sub(r"^(module\.)?backbone\.(vision_model\.)?", "", k)
            if key not in init:
                print("  [skip, no init]", k, file=sys.stderr); continue
            d = (v.float() - init[key]); agg[r][0] += d.pow(2).sum().item()
            agg[r][1] += init[key].pow(2).sum().item(); agg[r][2] += v.numel()
            m = re.search(r"layers\.(\d+)\.", key)
            if m:
                layer[(int(m.group(1)), r)][0] += d.pow(2).sum().item()
                layer[(int(m.group(1)), r)][1] += init[key].pow(2).sum().item()
        rows = {r: {"n": n, "delta": dd ** .5, "rel": (dd / max(ii, 1e-12)) ** .5}
                for r, (dd, ii, n) in sorted(agg.items())}
        print(f"\n== {path}")
        print(f"{'role':16s}{'#params':>9s}{'||delta||':>11s}{'rel':>9s}")
        for r, x in rows.items():
            print(f"{r:16s}{x['n']:9d}{x['delta']:11.4f}{x['rel']:9.4f}")
        if a.per_layer:
            roles = sorted({r for _, r in layer})
            print("layer " + " ".join(f"{r[:11]:>11s}" for r in roles))
            for L in sorted({l for l, _ in layer}):
                print(f"{L:5d} " + " ".join(f"{(layer[(L, r)][0] / max(layer[(L, r)][1], 1e-12)) ** .5:11.4f}" for r in roles))
        report[path] = {"roles": rows, "per_layer": {f"{l}:{r}": (dd / max(ii, 1e-12)) ** .5 for (l, r), (dd, ii) in layer.items()}}
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
