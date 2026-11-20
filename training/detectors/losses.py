"""Loss functions used by BiasLN, its improved variant and the GenD baseline.

Class-weighted cross-entropy with label smoothing (the BiasLN objective)
------------------------------------------------------------------------
With labels ``y in {0: real, 1: fake}``, class weights ``w_y`` and smoothing
``alpha``, the smoothed target is ``q_c = (1 - alpha) * 1[c = y] + alpha / C``
and the per-sample loss is ``w_y * sum_c -q_c log p_c``. PyTorch's
``CrossEntropyLoss(weight=w, label_smoothing=alpha, reduction='mean')`` then
normalises by ``sum_i w_{y_i}``:

    L_ce = sum_i w_{y_i} * sum_c -q_{i,c} log p_{i,c}  /  sum_i w_{y_i}

This is the exact formula the paper should state (Reviewer 2's comment that
the equation omitted the class weights). ``weighted_ce`` below makes that
explicit and is unit-tested against ``nn.CrossEntropyLoss``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Explicit form of class-weighted, label-smoothed cross-entropy."""
    num_classes = logits.size(1)
    log_p = F.log_softmax(logits, dim=1)
    q = F.one_hot(labels, num_classes).float() * (1.0 - label_smoothing)
    q = q + label_smoothing / num_classes
    # torch weights each target class term by w_c (not only w_{y_i}); we
    # reproduce that exactly so numbers match CrossEntropyLoss.
    w = class_weights.to(logits.dtype)
    per_sample = -(q * log_p * w.view(1, -1)).sum(dim=1)
    return per_sample.sum() / w[labels].sum()


class AsymmetricSupConLoss(nn.Module):
    """Asymmetric supervised contrastive loss (real-real pull, fake-real push).

    * real-real pairs: SupCon positives (pulled together)
    * fake-real pairs: penalised when ``cos(fake, real) > margin``
    * fake-fake pairs: ignored, so heterogeneous forgeries are not collapsed
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
        real_label: int = 0,
        lambda_fake: float = 0.1,
        fake_margin: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.real_label = real_label
        self.lambda_fake = lambda_fake
        self.fake_margin = fake_margin
        self.eps = eps

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        feats = F.normalize(features.float(), dim=-1)
        labels = labels.view(-1)
        n = feats.size(0)
        if n < 2:
            return feats.sum() * 0.0
        not_self = ~torch.eye(n, dtype=torch.bool, device=feats.device)
        is_real = labels.eq(self.real_label)

        cos = feats @ feats.T
        logits = cos / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * not_self
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)

        pos = (is_real[:, None] & is_real[None, :] & not_self).float()
        pos_cnt = pos.sum(1)
        valid_real = is_real & (pos_cnt > 0)
        real_term = -(self.temperature / self.base_temperature) * (
            (pos * log_prob).sum(1) / pos_cnt.clamp_min(1.0)
        )
        real_loss = (real_term * valid_real).sum() / (valid_real.sum() + self.eps)

        fr = ((~is_real)[:, None] & is_real[None, :]).float()
        fr_cnt = fr.sum(1)
        valid_fake = (~is_real) & (fr_cnt > 0)
        fake_term = (fr * F.relu(cos - self.fake_margin)).sum(1) / fr_cnt.clamp_min(1.0)
        fake_loss = (fake_term * valid_fake).sum() / (valid_fake.sum() + self.eps)

        return real_loss + self.lambda_fake * fake_loss


def alignment_loss(embeddings: torch.Tensor, labels: torch.Tensor, alpha: float = 2.0):
    """GenD alignment loss (Wang & Isola, 2020) over same-class pairs."""
    if embeddings.size(0) < 2:
        return embeddings.sum() * 0.0
    same = (labels[:, None] == labels[None, :]).triu(diagonal=1)
    idx = torch.nonzero(same, as_tuple=False)
    if idx.numel() == 0:
        return embeddings.sum() * 0.0
    diff = embeddings[idx[:, 0]] - embeddings[idx[:, 1]]
    return diff.norm(p=2, dim=1).pow(alpha).mean()


def uniformity_loss(x: torch.Tensor, t: float = 2.0, clip_value: float = 1e-6):
    """GenD uniformity loss on the L2 hypersphere."""
    if x.size(0) < 2:
        return x.sum() * 0.0
    sq = torch.pdist(x.float(), p=2).pow(2)
    return sq.mul(-t).exp().mean().clamp(min=clip_value).log()


class RealPrototypeLoss(nn.Module):
    """Asymmetric one-class loss on the hypersphere with a learnable real prototype.

    Real features are pulled to a unit prototype ``c``; fakes are only pushed
    below a cosine margin ``m`` from it (no fake-fake term, no fake centre):

        L = mean_real (1 - cos(f, c)) + lambda_fake * mean_fake relu(cos(f, c) - m)

    Unlike batch SupCon it does not need several reals per batch (FF++ is 1:4).
    """

    def __init__(self, dim: int, margin: float = 0.3, lambda_fake: float = 1.0, real_label: int = 0):
        super().__init__()
        self.center = nn.Parameter(F.normalize(torch.randn(dim), dim=0))
        self.margin = margin
        self.lambda_fake = lambda_fake
        self.real_label = real_label

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        f = F.normalize(features.float(), dim=-1)
        cos = f @ F.normalize(self.center.float(), dim=0)
        labels = labels.view(-1)
        real = labels.eq(self.real_label)
        zero = cos.sum() * 0.0
        l_real = (1.0 - cos[real]).mean() if real.any() else zero
        l_fake = F.relu(cos[~real] - self.margin).mean() if (~real).any() else zero
        return l_real + self.lambda_fake * l_fake


def anchor_loss(student: torch.Tensor, teacher: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """1 - cos between tuned and frozen-pretrained features (optionally on a subset)."""
    if mask is not None:
        if not mask.any():
            return student.sum() * 0.0
        student, teacher = student[mask], teacher[mask]
    return (1.0 - F.cosine_similarity(student.float(), teacher.float(), dim=-1)).mean()
