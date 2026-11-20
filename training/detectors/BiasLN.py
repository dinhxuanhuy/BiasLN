"""BiasLN: bias + LayerNorm tuning of a frozen CLIP ViT-L/14 for deepfake detection.

One detector class implements the paper's model, every ablation row and the
re-implemented PEFT baselines, so that the *only* difference between two runs
is the YAML config (same data pipeline, head, schedule, selection protocol):

=====================  =================================================
model_name / config     what is trained
=====================  =================================================
biasln (BiasLN.yaml)    all biases + LayerNorm gamma/beta, L2-norm, w-CE
biasln + aux_loss       BiasLN + asymmetric SupCon (improved variant)
gend   (GenD.yaml)      LayerNorm only + alignment/uniformity losses
effort (Effort.yaml)    SVD residual (rank-1) of every attention linear
tuning_mode=...         frozen | bitfit | ln | bitfit_ln | lora | full
=====================  =================================================

Labels follow DeepfakeBench: 0 = real, 1 = fake; ``prob`` is P(fake).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_detector import AbstractDetector
from .losses import AsymmetricSupConLoss, RealPrototypeLoss, alignment_loss, anchor_loss, uniformity_loss
from .peft import TUNING_MODES, inject_adapters, inject_lora, inject_vpt, param_role, parse_blocks, restrict_bias_blocks, restrict_blocks, restrict_roles, select_trainable
from detectors import DETECTOR

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Effort baseline: orthogonal-subspace (SVD residual) adaptation
# --------------------------------------------------------------------------
class SVDResidualLinear(nn.Module):
    """``W = U_r S_r V_r^T (frozen) + U_res diag(S_res) V_res^T (trainable)``."""

    def __init__(self, base: nn.Linear, keep_rank: int):
        super().__init__()
        weight = base.weight.data.float()
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        r = min(keep_rank, S.numel())
        self.weight_main = nn.Parameter(
            (U[:, :r] @ torch.diag(S[:r]) @ Vh[:r]).to(base.weight.dtype),
            requires_grad=False,
        )
        self.U_residual = nn.Parameter(U[:, r:].clone())
        self.S_residual = nn.Parameter(S[r:].clone())
        self.V_residual = nn.Parameter(Vh[r:].clone())
        self.bias = None
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.data.clone(), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.U_residual @ torch.diag(self.S_residual) @ self.V_residual
        weight = self.weight_main + residual.to(self.weight_main.dtype)
        return F.linear(x, weight, self.bias)

    def orthogonal_loss(self) -> torch.Tensor:
        u = self.U_residual.T @ self.U_residual
        v = self.V_residual @ self.V_residual.T
        eye_u = torch.eye(u.size(0), device=u.device)
        eye_v = torch.eye(v.size(0), device=v.device)
        return 0.5 * torch.norm(u - eye_u, p="fro") + 0.5 * torch.norm(v - eye_v, p="fro")


def apply_svd_residual(backbone: nn.Module, keep_rank: int) -> int:
    """Replace every Linear inside self-attention with ``SVDResidualLinear``."""
    replaced = 0
    for layer in backbone.encoder.layers:
        attn = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            module = getattr(attn, name)
            if isinstance(module, nn.Linear):
                setattr(attn, name, SVDResidualLinear(module, keep_rank))
                replaced += 1
    for name, p in backbone.named_parameters():
        p.requires_grad = any(k in name for k in ("U_residual", "S_residual", "V_residual"))
    return replaced


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------
class BiasLNDetector(AbstractDetector):
    DEFAULTS: Dict = {}

    def __init__(self, config: Optional[dict] = None, backbone: Optional[nn.Module] = None):
        super().__init__()
        cfg = dict(self.DEFAULTS)
        cfg.update(config or {})
        self.config = cfg

        self.tuning_mode = str(cfg.get("tuning_mode", "bitfit_ln"))
        self.feature_dim = int(cfg.get("feature_dim", 1024))
        self.l2_norm = bool(cfg.get("l2_norm", True))
        self.normalize_eps = float(cfg.get("normalize_eps", 1e-6))
        self.aux_loss = str(cfg.get("aux_loss", "none") or "none")
        self.lambda_aux = float(cfg.get("lambda_aux", 0.0))

        self.backbone = backbone if backbone is not None else self.build_backbone(cfg)
        # Real-anchoring needs the frozen pretrained encoder; copy it BEFORE adaptation.
        # Kept outside the module tree (no state_dict / param count / optimizer).
        self.lambda_anchor = float(cfg.get("lambda_anchor", 0.0))
        self.anchor_on = str(cfg.get("anchor_on", "real"))  # real | all | fake
        self._teacher = None
        if self.lambda_anchor > 0:
            import copy
            t = copy.deepcopy(self.backbone).eval()
            for p in t.parameters():
                p.requires_grad = False
            object.__setattr__(self, "_teacher", [t])
        self.trainable_counts = self._adapt_backbone()

        self.head = nn.Linear(self.feature_dim, 2)
        proj_dim = int(cfg.get("projection_dim", 0))
        self.projection_head = None
        if self.aux_loss == "asym_supcon" and proj_dim > 0:
            self.projection_head = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feature_dim, proj_dim),
            )
        self.build_loss(cfg)
        self.trainable_param_summary = self._summarize()
        logger.info(
            "%s ready: mode=%s trainable=%s / %s (%.4f%%) breakdown=%s",
            type(self).__name__,
            self.tuning_mode,
            f"{self.trainable_param_summary['trainable']:,}",
            f"{self.trainable_param_summary['total']:,}",
            self.trainable_param_summary["percent"],
            self.trainable_counts,
        )

    # ------------------------------------------------------------- building
    def build_backbone(self, config):
        from transformers import CLIPModel

        name = config.get("clip_model_name", "openai/clip-vit-large-patch14")
        try:
            clip = CLIPModel.from_pretrained(name)
        except Exception:
            clip = CLIPModel.from_pretrained(name, local_files_only=True)
        return clip.vision_model

    def _adapt_backbone(self) -> Dict[str, int]:
        cfg = self.config
        mode = self.tuning_mode
        if mode == "effort":
            keep = int(cfg.get("effort_keep_rank", 1023))
            n = apply_svd_residual(self.backbone, keep)
            trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            return {"svd_residual": trainable, "replaced_linears": n,
                    "backbone_total_trainable": trainable}
        if mode not in TUNING_MODES:
            raise ValueError(f"unknown tuning_mode {mode!r}")
        if mode == "lora":
            inject_lora(
                self.backbone,
                targets=cfg.get("lora_targets", ["q_proj", "v_proj"]),
                rank=int(cfg.get("lora_rank", 8)),
                alpha=float(cfg.get("lora_alpha", 16.0)),
                dropout=float(cfg.get("lora_dropout", 0.0)),
                last_k_blocks=int(cfg.get("lora_last_k_blocks", 0)),
            )
        if mode == "adapter":
            inject_adapters(self.backbone, bottleneck=int(cfg.get("adapter_bottleneck", 16)),
                            scale=float(cfg.get("adapter_scale", 0.1)))
        if mode == "vpt":
            inject_vpt(self.backbone, num_prompts=int(cfg.get("vpt_num_prompts", 8)))
        counts = select_trainable(self.backbone, mode, int(cfg.get("last_k_blocks", 0)))
        roles = cfg.get("param_roles")  # e.g. [q, fc1] or "q+fc1": tune only these roles
        if roles:
            if isinstance(roles, str):
                roles = [r for r in re.split(r"[+,\s]+", roles) if r]
            counts = restrict_roles(self.backbone, roles)
        blocks = cfg.get("tune_blocks")  # e.g. "12-23", "last6": tune only these encoder blocks
        if blocks not in (None, "", "all"):
            n = len(self.backbone.encoder.layers) if hasattr(self.backbone, "encoder") else 24
            counts = restrict_blocks(self.backbone, parse_blocks(blocks, n),
                                     keep_outer=bool(cfg.get("tune_outer_ln", True)))
        bias_blocks = cfg.get("bias_blocks")  # e.g. "0-18": LN gamma/beta everywhere, biases only here
        if bias_blocks not in (None, "", "all"):
            n = len(self.backbone.encoder.layers) if hasattr(self.backbone, "encoder") else 24
            counts = restrict_bias_blocks(self.backbone, parse_blocks(bias_blocks, n))
        return counts

    def build_loss(self, config) -> None:
        weights = torch.tensor(
            [float(config.get("weight_real", 1.0)), float(config.get("weight_fake", 1.0))]
        )
        self.loss_ce = nn.CrossEntropyLoss(
            weight=weights, label_smoothing=float(config.get("label_smoothing", 0.0))
        )
        self.loss_asym = None
        if self.aux_loss == "asym_supcon":
            self.loss_asym = AsymmetricSupConLoss(
                temperature=float(config.get("temperature", 0.07)),
                base_temperature=float(config.get("base_temperature", 0.07)),
                lambda_fake=float(config.get("lambda_fake", 0.1)),
                fake_margin=float(config.get("fake_margin", 0.0)),
            )
        elif self.aux_loss == "real_proto":
            self.loss_proto = RealPrototypeLoss(
                self.feature_dim, margin=float(config.get("proto_margin", 0.3)),
                lambda_fake=float(config.get("lambda_fake", 1.0)))
        elif self.aux_loss not in ("none", "gend"):
            raise ValueError(f"unknown aux_loss {self.aux_loss!r}")

    # ------------------------------------------------------------ utilities
    def _summarize(self) -> Dict[str, float]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable,
                "percent": 100.0 * trainable / max(total, 1)}

    def get_trainable_summary(self) -> Dict[str, float]:
        return self.trainable_param_summary

    def get_param_groups(self, base_lr: float) -> List[dict]:
        """Head/projection at ``base_lr``; backbone at ``backbone_lr`` if set.

        A separate, smaller backbone LR is what keeps the full fine-tuning
        baseline from diverging (Reviewer 1: "Full FT looks under-trained").
        """
        backbone_lr = self.config.get("backbone_lr", None)
        backbone_lr = base_lr if backbone_lr is None else float(backbone_lr)
        # ln_gamma_lr_scale: LayerNorm scales (the only multiplicative parameters of BiasLN)
        # get backbone_lr * scale instead of sharing the additive biases' learning rate.
        g_scale = self.config.get("ln_gamma_lr_scale", None)
        gamma_roles = ("ln_gamma", "outer_gamma")
        bb, gam = [], []
        for n, p in self.backbone.named_parameters():
            if p.requires_grad:
                (gam if g_scale is not None and param_role(n) in gamma_roles else bb).append(p)
        rest = [p for n, p in self.named_parameters()
                if p.requires_grad and not n.startswith("backbone.")]
        groups = [{"params": rest, "lr": base_lr, "name": "head"}]
        if bb:
            groups.append({"params": bb, "lr": backbone_lr, "name": "backbone"})
        if gam:
            groups.append({"params": gam, "lr": backbone_lr * float(g_scale), "name": "ln_gamma"})
        return groups

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """Only the trained tensors (~1.3 MB for BiasLN instead of ~1.2 GB)."""
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in keep}

    # -------------------------------------------------------------- forward
    def features(self, data_dict: dict) -> torch.Tensor:
        layers = self.config.get("feature_layers")  # e.g. "12,18,24": mean of CLS tokens of these blocks
        if layers in (None, "", "last"):
            return self.backbone(data_dict["image"]).pooler_output
        idx = [int(i) for i in re.split(r"[+,\s]+", str(layers)) if i]
        out = self.backbone(data_dict["image"], output_hidden_states=True)
        post = self.backbone.post_layernorm  # hidden_states[i] = output of block i (0 = embeddings)
        return torch.stack([post(out.hidden_states[i][:, 0]) for i in idx]).mean(0)

    def classifier(self, features: torch.Tensor) -> torch.Tensor:
        if self.l2_norm:
            features = F.normalize(features, p=2, dim=1, eps=self.normalize_eps)
        return self.head(features)

    def forward(self, data_dict: dict, inference: bool = False) -> dict:
        raw = self.features(data_dict)
        feat = F.normalize(raw, p=2, dim=1, eps=self.normalize_eps) if self.l2_norm else raw
        logits = self.head(feat)
        out = {
            "cls": logits,
            "prob": torch.softmax(logits.float(), dim=1)[:, 1],
            "feat": raw,
            "feat_norm": feat,
        }
        if self.training and self._teacher is not None:
            t = self._teacher[0]
            dev = data_dict["image"].device
            if next(t.parameters()).device != dev:
                t.to(dev)
            labels = data_dict["label"].view(-1)
            mask = {"real": labels.eq(0), "fake": labels.eq(1)}.get(
                self.anchor_on, torch.ones_like(labels, dtype=torch.bool))
            out["anchor_mask"] = mask
            with torch.no_grad():  # frozen teacher only on the anchored subset
                out["feat_teacher"] = (t(data_dict["image"][mask]).pooler_output
                                       if mask.any() else None)
        if self.training and self.aux_loss == "asym_supcon":
            out["feat_proj"] = (
                self.projection_head(raw) if self.projection_head is not None else feat
            )
        return out

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        labels = data_dict["label"].view(-1)
        logits = pred_dict["cls"].float()
        loss_ce = self.loss_ce(logits, labels)
        zero = logits.sum() * 0.0
        # Every auxiliary term carries its own weight; loss_aux is the weighted sum.
        loss_aux = zero
        if self.training:
            if self.aux_loss == "asym_supcon" and self.lambda_aux > 0:
                loss_aux = loss_aux + self.lambda_aux * self.loss_asym(
                    pred_dict["feat_proj"], labels
                )
            elif self.aux_loss == "real_proto" and self.lambda_aux > 0:
                loss_aux = loss_aux + self.lambda_aux * self.loss_proto(pred_dict["feat"], labels)
            elif self.aux_loss == "gend":
                f = F.normalize(pred_dict["feat"].float(), p=2, dim=1)
                loss_aux = loss_aux + (
                    float(self.config.get("lambda_align", 0.1)) * alignment_loss(f, labels)
                    + float(self.config.get("lambda_unif", 0.5)) * uniformity_loss(f)
                )
            if self.lambda_anchor > 0 and pred_dict.get("feat_teacher") is not None:
                loss_aux = loss_aux + self.lambda_anchor * anchor_loss(
                    pred_dict["feat"][pred_dict["anchor_mask"]], pred_dict["feat_teacher"])
            orth_lambda = float(self.config.get("effort_orth_lambda", 0.0))
            if self.tuning_mode == "effort" and orth_lambda > 0:
                loss_aux = loss_aux + orth_lambda * sum(
                    m.orthogonal_loss() for m in self.backbone.modules()
                    if isinstance(m, SVDResidualLinear)
                )
        overall = loss_ce + loss_aux

        losses = {"overall": overall, "loss_ce": loss_ce, "loss_aux": loss_aux}
        with torch.no_grad():
            for key, cls in (("real_loss", 0), ("fake_loss", 1)):
                mask = labels.eq(cls)
                losses[key] = self.loss_ce(logits[mask], labels[mask]) if mask.any() else zero.detach()
        return losses

    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        from metrics.base_metrics_class import calculate_metrics_for_train

        auc, eer, acc, ap = calculate_metrics_for_train(
            data_dict["label"].detach(), pred_dict["cls"].detach().float()
        )
        return {"acc": acc, "auc": auc, "eer": eer, "ap": ap}


@DETECTOR.register_module(module_name="biasln")
class BiasLN(BiasLNDetector):
    DEFAULTS = {"tuning_mode": "bitfit_ln", "l2_norm": True}


@DETECTOR.register_module(module_name="gend")
class GenD(BiasLNDetector):
    """GenD (Yermakov et al., WACV 2026): LayerNorm-only + align/uniform losses."""

    DEFAULTS = {"tuning_mode": "ln", "l2_norm": True, "aux_loss": "gend",
                "lambda_align": 0.1, "lambda_unif": 0.5}


@DETECTOR.register_module(module_name="effort")
class Effort(BiasLNDetector):
    """Effort (Yan et al., ICML 2025): rank-1 SVD residual of attention weights."""

    DEFAULTS = {"tuning_mode": "effort", "l2_norm": False, "effort_keep_rank": 1023}
