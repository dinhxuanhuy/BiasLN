"""Parameter-selection utilities shared by BiasLN and its ablations.

Every adaptation scheme in the paper's ablation (Table 4) is expressed as a
``tuning_mode`` so all variants share one detector, one data pipeline and one
training loop. This isolates the effect of *which* backbone parameters are
trained from every other design choice (head, loss, normalisation, schedule).

Supported modes
---------------
frozen      linear probe: backbone fully frozen, only the head is trained
bitfit      every ``bias`` vector (incl. LayerNorm beta)            ~0.27M
ln          LayerNorm gamma + beta only (GenD-style)                ~0.10M
bitfit_ln   union of the two above -> **BiasLN**                     ~0.32M
lora        LoRA on attention projections, everything else frozen
full        full fine-tuning
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

TUNING_MODES = ("frozen", "bitfit", "ln", "bitfit_ln", "lora", "full", "adapter", "vpt")


class LoRALinear(nn.Module):
    """``y = W x + b + (alpha / r) * B A x`` with ``W`` and ``b`` frozen."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be > 0")
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B)
        return self.base(x) + update * self.scaling


def inject_lora(
    backbone: nn.Module,
    targets: Sequence[str] = ("q_proj", "v_proj"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    last_k_blocks: int = 0,
) -> List[str]:
    """Wrap attention projections of a HF CLIP vision encoder with LoRA."""
    layers = backbone.encoder.layers
    start = 0 if last_k_blocks <= 0 else max(0, len(layers) - last_k_blocks)
    injected = []
    for idx in range(start, len(layers)):
        attn = layers[idx].self_attn
        for target in targets:
            module = getattr(attn, target)
            if isinstance(module, LoRALinear):
                continue
            if not isinstance(module, nn.Linear):
                raise TypeError(f"layer {idx}.{target} is {type(module)}, expected nn.Linear")
            setattr(attn, target, LoRALinear(module, rank, alpha, dropout))
            injected.append(f"encoder.layers.{idx}.self_attn.{target}")
    return injected


class AdaptMLP(nn.Module):
    """AdaptFormer-style parallel adapter: ``mlp(x) + s * up(relu(down(x)))``; ``up`` starts at zero,
    so the wrapped block equals the frozen block at initialization."""

    def __init__(self, mlp: nn.Module, dim: int, bottleneck: int, scale: float):
        super().__init__()
        self.mlp = mlp
        self.scale = scale
        self.adapter_down = nn.Linear(dim, bottleneck)
        self.adapter_up = nn.Linear(bottleneck, dim)
        nn.init.kaiming_uniform_(self.adapter_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x) + self.scale * self.adapter_up(F.relu(self.adapter_down(x)))


def inject_adapters(backbone: nn.Module, bottleneck: int = 16, scale: float = 0.1) -> int:
    """Wrap the MLP of every encoder block with a parallel bottleneck adapter (additive PEFT)."""
    n = 0
    for layer in backbone.encoder.layers:
        if isinstance(layer.mlp, AdaptMLP):
            continue
        dim = layer.mlp.fc1.in_features
        layer.mlp = AdaptMLP(layer.mlp, dim, bottleneck, scale)
        n += 1
    return n


def inject_vpt(backbone: nn.Module, num_prompts: int = 8) -> int:
    """VPT-deep: learnable prompt tokens inserted after [CLS] before block 0 and replaced
    before every later block (Jia et al., ECCV 2022). The class token is read as usual."""
    layers = backbone.encoder.layers
    dim = layers[0].mlp.fc1.in_features
    if hasattr(backbone, "vpt_prompts"):
        return 0
    prompts = nn.ParameterList()
    for _ in layers:
        p = nn.Parameter(torch.empty(1, num_prompts, dim))
        nn.init.uniform_(p, -0.02, 0.02)  # small init, close to the embedding scale
        prompts.append(p)
    backbone.vpt_prompts = prompts

    def make_hook(idx):
        def hook(module, args, kwargs):
            if args:
                h, rest = args[0], args[1:]
            else:
                h, rest = kwargs.pop("hidden_states"), ()
            pr = backbone.vpt_prompts[idx].to(h.dtype).expand(h.shape[0], -1, -1)
            if idx == 0:
                h = torch.cat([h[:, :1], pr, h[:, 1:]], dim=1)
            else:
                h = torch.cat([h[:, :1], pr, h[:, 1 + num_prompts:]], dim=1)
            if args:
                return (h,) + tuple(rest), kwargs
            kwargs["hidden_states"] = h
            return args, kwargs
        return hook

    for i, layer in enumerate(layers):
        layer.register_forward_pre_hook(make_hook(i), with_kwargs=True)
    return len(layers)


def layernorm_param_names(module: nn.Module) -> set:
    """Names of LayerNorm affine params, found by module type (not by name)."""
    names = set()
    for mod_name, mod in module.named_modules():
        if isinstance(mod, nn.LayerNorm):
            for p_name, _ in mod.named_parameters(recurse=False):
                names.add(f"{mod_name}.{p_name}" if mod_name else p_name)
    return names


PARAM_ROLES = ("q", "k", "v", "out", "fc1", "fc2", "ln_gamma", "ln_beta",
               "outer_gamma", "outer_beta", "embed")


def param_role(name: str) -> str:
    """Role of a CLIP-ViT parameter inside the network (used by ``param_roles``).

    q/k/v/out/fc1/fc2 are the biases of those linears; ln_* are the per-block
    LayerNorms, outer_* the pre/post LayerNorms around the encoder.
    """
    for pat, r in ((r"q_proj\.bias$", "q"), (r"k_proj\.bias$", "k"), (r"v_proj\.bias$", "v"),
                   (r"out_proj\.bias$", "out"), (r"fc1\.bias$", "fc1"), (r"fc2\.bias$", "fc2"),
                   (r"layer_norm\d\.weight$", "ln_gamma"), (r"layer_norm\d\.bias$", "ln_beta"),
                   (r"(pre_layrnorm|post_layernorm)\.weight$", "outer_gamma"),
                   (r"(pre_layrnorm|post_layernorm)\.bias$", "outer_beta"),
                   (r"embeddings", "embed")):
        if re.search(pat, name):
            return r
    return "other"


def restrict_roles(backbone: nn.Module, roles: Iterable[str]) -> Dict[str, int]:
    """Keep ``requires_grad`` only on already-trainable params whose role is in ``roles``."""
    roles = set(roles)
    bad = roles - set(PARAM_ROLES)
    if bad:
        raise ValueError(f"unknown param_roles {sorted(bad)}; choose from {PARAM_ROLES}")
    counts: Dict[str, int] = {}
    for name, p in backbone.named_parameters():
        if p.requires_grad:
            r = param_role(name)
            if r in roles:
                counts[r] = counts.get(r, 0) + p.numel()
            else:
                p.requires_grad = False
    counts["backbone_total_trainable"] = sum(counts.values())
    return counts


def parse_blocks(spec, n_blocks: int = 24) -> set:
    """'12-23' -> {12..23}; 'last6' -> {18..23}; 'first6' -> {0..5}; '0,5,7-9' -> {0,5,7,8,9}."""
    spec = str(spec).strip().lower()
    if spec.startswith("last"):
        k = int(spec[4:]); return set(range(n_blocks - k, n_blocks))
    if spec.startswith("first"):
        k = int(spec[5:]); return set(range(k))
    out = set()
    for part in re.split(r"[,+\s]+", spec):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-"); out |= set(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    bad = [b for b in out if not 0 <= b < n_blocks]
    if bad:
        raise ValueError(f"tune_blocks out of range: {sorted(bad)}")
    return out


def restrict_bias_blocks(backbone: nn.Module, blocks: Iterable[int]) -> Dict[str, int]:
    """Keep every LayerNorm gamma/beta trainable, but keep attention/MLP biases trainable only
    inside encoder blocks ``blocks`` (LN-everywhere + partial-bias variant)."""
    blocks = set(blocks)
    ln_roles = ("ln_gamma", "ln_beta", "outer_gamma", "outer_beta")
    counts: Dict[str, int] = {}
    for name, p in backbone.named_parameters():
        if not p.requires_grad:
            continue
        r = param_role(name)
        if r in ln_roles:
            key = "layernorm"
        else:
            m = re.search(r"layers\.(\d+)\.", name)
            if m and int(m.group(1)) in blocks:
                key = "bias"
            else:
                p.requires_grad = False
                continue
        counts[key] = counts.get(key, 0) + p.numel()
    counts["backbone_total_trainable"] = sum(counts.values())
    return counts


def restrict_blocks(backbone: nn.Module, blocks: Iterable[int], keep_outer: bool = True) -> Dict[str, int]:
    """Keep ``requires_grad`` only on already-trainable params inside encoder blocks ``blocks``
    (plus the pre/post LayerNorms around the encoder if ``keep_outer``)."""
    blocks = set(blocks)
    counts: Dict[str, int] = {}
    for name, p in backbone.named_parameters():
        if not p.requires_grad:
            continue
        m = re.search(r"layers\.(\d+)\.", name)
        if m:
            keep = int(m.group(1)) in blocks
            key = f"block{int(m.group(1)):02d}"
        else:
            keep = keep_outer and param_role(name) in ("outer_gamma", "outer_beta")
            key = "outer"
        if keep:
            counts[key] = counts.get(key, 0) + p.numel()
        else:
            p.requires_grad = False
    counts["backbone_total_trainable"] = sum(v for k, v in counts.items())
    return counts


def select_trainable(
    backbone: nn.Module,
    mode: str,
    last_k_blocks: int = 0,
) -> Dict[str, int]:
    """Set ``requires_grad`` on ``backbone`` according to ``mode``.

    ``last_k_blocks`` > 0 additionally unfreezes the full weights of the last
    k transformer blocks (used by the "BiasLN + last blocks" variant). It is
    reported separately because it breaks the PEFT budget.
    Returns a per-group count of trainable parameters.
    """
    if mode not in TUNING_MODES:
        raise ValueError(f"tuning_mode must be one of {TUNING_MODES}, got {mode!r}")

    ln_names = layernorm_param_names(backbone)
    counts = {"bias": 0, "layernorm_gamma": 0, "lora": 0, "other": 0, "last_blocks": 0}

    for name, p in backbone.named_parameters():
        is_ln = name in ln_names
        is_bias = name.endswith(".bias") or name == "bias"
        is_lora = name.endswith("lora_A") or name.endswith("lora_B")
        is_add = ".adapter_" in name or name.startswith("vpt_prompts")

        if mode in ("adapter", "vpt"):
            train = is_add
        elif mode == "full":
            train = not is_lora and not is_add
        elif mode == "frozen":
            train = False
        elif mode == "bitfit":
            train = is_bias and not is_lora and not is_add
        elif mode == "ln":
            train = is_ln
        elif mode == "bitfit_ln":
            train = (is_bias or is_ln) and not is_lora and not is_add
        else:  # lora
            train = is_lora
        p.requires_grad = train

        if train:
            if is_add:
                counts["additive"] = counts.get("additive", 0) + p.numel()
            elif is_lora:
                counts["lora"] += p.numel()
            elif is_bias:
                counts["bias"] += p.numel()
            elif is_ln:
                counts["layernorm_gamma"] += p.numel()
            else:
                counts["other"] += p.numel()

    if last_k_blocks > 0:
        layers = backbone.encoder.layers
        for layer in list(layers)[-last_k_blocks:]:
            for p in layer.parameters():
                if not p.requires_grad:
                    p.requires_grad = True
                    counts["last_blocks"] += p.numel()

    counts["backbone_total_trainable"] = sum(
        v for k, v in counts.items() if k != "backbone_total_trainable"
    )
    return counts


def count_parameters(params: Iterable[torch.nn.Parameter]) -> int:
    return sum(p.numel() for p in params)
