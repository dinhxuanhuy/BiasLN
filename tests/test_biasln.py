import io
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "training"
for p in (str(TRAINING_DIR), str(ROOT / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from metrics.registry import DETECTOR  # noqa: E402

if "detectors" not in sys.modules:
    pkg = types.ModuleType("detectors")
    pkg.__path__ = [str(TRAINING_DIR / "detectors")]
    pkg.DETECTOR = DETECTOR
    sys.modules["detectors"] = pkg

from detectors.BiasLN import BiasLN, Effort, GenD, SVDResidualLinear  # noqa: E402
from detectors.losses import AsymmetricSupConLoss, weighted_ce  # noqa: E402
from detectors.peft import layernorm_param_names, select_trainable  # noqa: E402
from config_utils import apply_overrides  # noqa: E402
import stats  # noqa: E402

transformers = pytest.importorskip("transformers")
from transformers import CLIPVisionConfig, CLIPVisionModel  # noqa: E402

TINY = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, image_size=28, patch_size=14)


def tiny_backbone():
    torch.manual_seed(0)
    model = CLIPVisionModel(CLIPVisionConfig(**TINY))
    return getattr(model, "vision_model", model)  # transformers 4.x nests, 5.x is flat


def make(cls=BiasLN, **cfg):
    base = {"feature_dim": 32, "weight_real": 1.0, "weight_fake": 2.0, "label_smoothing": 0.1}
    base.update(cfg)
    return cls(base, backbone=tiny_backbone())


def batch(n=6):
    torch.manual_seed(1)
    return {"image": torch.randn(n, 3, 28, 28), "label": torch.tensor([0, 1] * (n // 2))}


def trainable_names(model):
    return {n for n, p in model.backbone.named_parameters() if p.requires_grad}


# ------------------------------------------------------------ selection
def test_vit_l14_parameter_budget_matches_paper():
    cfg = CLIPVisionConfig(hidden_size=1024, intermediate_size=4096, num_hidden_layers=24,
                           num_attention_heads=16, image_size=224, patch_size=14)
    with torch.device("meta"):
        vm = CLIPVisionModel(cfg)
        vm = getattr(vm, "vision_model", vm)
    counts = select_trainable(vm, "bitfit_ln")
    assert counts["bias"] == 272_384          # all biases incl. LayerNorm beta
    assert counts["layernorm_gamma"] == 51_200  # 50 LayerNorms x 1024
    assert counts["backbone_total_trainable"] + 1024 * 2 + 2 == 325_634  # ~0.3M in paper
    assert select_trainable(vm, "ln")["backbone_total_trainable"] == 102_400  # GenD ~0.1M


def test_bitfit_ln_trains_exactly_biases_and_layernorm():
    m = make(tuning_mode="bitfit_ln")
    ln = layernorm_param_names(m.backbone)
    expected = {n for n, _ in m.backbone.named_parameters() if n.endswith(".bias") or n in ln}
    assert trainable_names(m) == expected
    assert all(p.requires_grad for p in m.head.parameters())
    assert not any("weight" in n and n not in ln for n in trainable_names(m))


@pytest.mark.parametrize("mode", ["frozen", "bitfit", "ln", "lora", "full"])
def test_ablation_modes(mode):
    m = make(tuning_mode=mode, lora_rank=2, lora_alpha=4)
    names = trainable_names(m)
    ln = layernorm_param_names(m.backbone)
    if mode == "frozen":
        assert names == set()
    elif mode == "bitfit":
        assert names and all(n.endswith(".bias") for n in names)
        assert not any(n.endswith(".weight") for n in names)
    elif mode == "ln":
        assert names == ln
    elif mode == "lora":
        assert names and all(n.endswith(("lora_A", "lora_B")) for n in names)
    else:
        assert names == {n for n, _ in m.backbone.named_parameters()}


def test_bias_blocks_keeps_ln_everywhere_and_bias_only_in_blocks():
    m = make(tuning_mode="bitfit_ln", bias_blocks="0")
    ln = layernorm_param_names(m.backbone)
    names = trainable_names(m)
    assert ln <= names
    extra = names - ln
    assert extra and all(".layers.0." in n and n.endswith(".bias") for n in extra)


@pytest.mark.parametrize("mode", ["adapter", "vpt"])
def test_additive_modes_train_only_new_params(mode):
    m = make(tuning_mode=mode, adapter_bottleneck=4, vpt_num_prompts=3)
    names = trainable_names(m)
    assert names and all((".adapter_" in n) or n.startswith("vpt_prompts") for n in names)
    m.eval()
    with torch.no_grad():
        out = m(batch(), inference=True)
    assert out["cls"].shape == (6, 2) and torch.isfinite(out["cls"]).all()


def test_adapter_is_identity_at_init():
    torch.manual_seed(0)
    ref = tiny_backbone().eval()
    m = make(tuning_mode="adapter", adapter_bottleneck=4)
    m.backbone.load_state_dict(ref.state_dict(), strict=False)
    x = batch()["image"]
    with torch.no_grad():
        assert torch.allclose(ref(x).pooler_output, m.backbone(x).pooler_output, atol=1e-6)


def test_lora_and_effort_are_identity_at_init():
    ref = make(tuning_mode="frozen").eval()
    for cls, cfg in ((BiasLN, {"tuning_mode": "lora", "lora_rank": 2}), (Effort, {})):
        m = make(cls, **cfg).eval()
        m.head.load_state_dict(ref.head.state_dict())
        x = batch()["image"]
        torch.testing.assert_close(m.features({"image": x}), ref.features({"image": x}),
                                   atol=1e-4, rtol=1e-4)


def test_effort_trains_only_svd_residual():
    m = make(Effort)
    names = trainable_names(m)
    assert names and all(any(k in n for k in ("U_residual", "S_residual", "V_residual")) for n in names)
    n_lin = sum(isinstance(x, SVDResidualLinear) for x in m.backbone.modules())
    assert n_lin == 2 * 4  # layers x (q, k, v, out)
    assert not m.l2_norm


# --------------------------------------------------------------- losses
def test_weighted_ce_matches_torch():
    torch.manual_seed(0)
    logits, labels = torch.randn(9, 2), torch.randint(0, 2, (9,))
    w = torch.tensor([1.0, 2.0])
    ref = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)(logits, labels)
    torch.testing.assert_close(weighted_ce(logits, labels, w, 0.1), ref)


def test_model_loss_uses_class_weights():
    m = make()
    out = m(batch())
    loss = m.get_losses(batch(), out)
    ref = weighted_ce(out["cls"], batch()["label"], torch.tensor([1.0, 2.0]), 0.1)
    torch.testing.assert_close(loss["loss_ce"], ref)
    torch.testing.assert_close(loss["overall"], ref)


def test_aux_losses_backprop_only_into_trainable():
    for cls, cfg in ((BiasLN, {"aux_loss": "asym_supcon", "lambda_aux": 0.05, "projection_dim": 8}),
                     (GenD, {})):
        m = make(cls, **cfg).train()
        out = m(batch())
        loss = m.get_losses(batch(), out)
        assert loss["loss_aux"].item() != 0.0
        loss["overall"].backward()
        for n, p in m.backbone.named_parameters():
            assert (p.grad is not None) == p.requires_grad, n


def test_asym_supcon_prefers_separated_classes():
    lab = torch.tensor([0, 0, 1, 1])
    sep = torch.tensor([[1.0, 0], [1, 0.1], [-1, 0], [-1, -0.1]])
    mixed = torch.tensor([[1.0, 0], [-1, 0.1], [1, 0.05], [-1, -0.1]])
    loss = AsymmetricSupConLoss()
    assert loss(sep, lab) < loss(mixed, lab)


# ----------------------------------------------------------- checkpoint
def test_compact_checkpoint_round_trip():
    m = make()
    with torch.no_grad():
        for p in m.parameters():
            if p.requires_grad:
                p.add_(0.01)
    state = m.trainable_state_dict()
    n_train = sum(1 for p in m.parameters() if p.requires_grad)
    assert len(state) == n_train
    buf = io.BytesIO()
    torch.save({"state_dict": state, "trainable_only": True}, buf)
    buf.seek(0)
    fresh = make()
    missing, unexpected = fresh.load_state_dict(torch.load(buf)["state_dict"], strict=False)
    assert not unexpected
    trainable = {n for n, p in fresh.named_parameters() if p.requires_grad}
    assert not (set(missing) & trainable)
    x = batch()
    torch.testing.assert_close(fresh.eval()(x)["prob"], m.eval()(x)["prob"])


def test_param_groups_split_backbone_lr():
    m = make(tuning_mode="full", backbone_lr=1e-5)
    groups = {g["name"]: g for g in m.get_param_groups(3e-4)}
    assert groups["backbone"]["lr"] == 1e-5 and groups["head"]["lr"] == 3e-4


# ---------------------------------------------------------------- tools
def test_apply_overrides():
    cfg = {"optimizer": {"adam": {"lr": 1e-4}}}
    apply_overrides(cfg, ["optimizer.adam.lr=3e-4", "l2_norm=false", "val_dataset=[A, B]"])
    assert cfg["optimizer"]["adam"]["lr"] == pytest.approx(3e-4)
    assert cfg["l2_norm"] is False and cfg["val_dataset"] == ["A", "B"]


def test_delong_matches_sklearn_and_detects_difference():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(300), np.ones(300)].astype(int)
    good = y + rng.normal(0, 0.6, 600)
    bad = y + rng.normal(0, 1.5, 600)
    res = stats.delong_test(y, good, bad)
    assert res["auc_a"] == pytest.approx(roc_auc_score(y, good))
    assert res["auc_b"] == pytest.approx(roc_auc_score(y, bad))
    assert res["p"] < 1e-3
    assert stats.delong_test(y, good, good)["p"] == pytest.approx(1.0)


def test_sign_flip_and_holm():
    assert stats.sign_flip_test(np.array([0.01] * 5)) == pytest.approx(2 / 32)
    adj = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_param_roles_restrict_and_k_bias_is_dead():
    """param_roles keeps only the requested roles; the k-proj bias gets an
    (analytically) zero gradient because softmax is shift-invariant per query."""
    model = make(tuning_mode="bitfit", param_roles="q+fc1")
    names = trainable_names(model)
    assert names and all(n.endswith(("q_proj.bias", "fc1.bias")) for n in names)

    full = make(tuning_mode="bitfit").double()
    b = batch()
    full(dict(image=b["image"].double(), label=b["label"]))["cls"].sum().backward()
    k = [p.grad.abs().max().item() for n, p in full.backbone.named_parameters() if n.endswith("k_proj.bias")]
    q = [p.grad.abs().max().item() for n, p in full.backbone.named_parameters() if n.endswith("q_proj.bias")]
    assert k and max(k) < 1e-12 and max(q) > 1e-6


def test_real_anchor_and_prototype_losses():
    """Real-anchoring: zero at init (tuned == frozen copy), positive after the
    biases move, gradients reach the backbone, teacher stays out of state_dict."""
    m = make(tuning_mode="bitfit", aux_loss="real_proto", lambda_aux=0.1,
             lambda_anchor=1.0, anchor_on="real")
    m.train()
    b = batch()
    out = m(b)
    assert out["feat_teacher"].shape[0] == int(b["label"].eq(0).sum())
    from detectors.losses import anchor_loss
    assert anchor_loss(out["feat"][out["anchor_mask"]], out["feat_teacher"]).item() < 1e-6
    assert not any("_teacher" in k for k in m.state_dict())
    assert "loss_proto.center" in m.trainable_state_dict()
    with torch.no_grad():
        for n, p in m.backbone.named_parameters():
            if p.requires_grad:
                p.add_(0.05 * torch.randn_like(p))
    out = m(b)
    loss = m.get_losses(b, out)
    assert loss["loss_aux"].item() > 0
    loss["overall"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m.backbone.parameters() if p.requires_grad)
    m.eval()
    assert "feat_teacher" not in m(b)


def test_tune_blocks_restricts_depth():
    """tune_blocks keeps bias/LN params only in the chosen encoder blocks (+ outer LNs)."""
    from detectors.peft import parse_blocks
    assert parse_blocks("12-23") == set(range(12, 24))
    assert parse_blocks("last6") == set(range(18, 24)) and parse_blocks("first2") == {0, 1}
    assert parse_blocks("0,5,7-8") == {0, 5, 7, 8}
    full = make(tuning_mode="bitfit_ln")
    n = len(full.backbone.encoder.layers)
    m = make(tuning_mode="bitfit_ln", tune_blocks=f"{n - 1}")
    names = trainable_names(m)
    assert names and all((f"layers.{n - 1}." in x) or ("layernorm" in x) or ("layrnorm" in x) for x in names)
    assert any(f"layers.{n - 1}." in x for x in names)
    assert len(names) < len(trainable_names(full))
    m2 = make(tuning_mode="bitfit_ln", tune_blocks=f"{n - 1}", tune_outer_ln=False)
    assert all(f"layers.{n - 1}." in x for x in trainable_names(m2))


def test_ln_gamma_lr_scale_and_roles_with_depth():
    """ln_gamma_lr_scale puts LN scales in their own group; roles and depth restrictions compose."""
    m = make(tuning_mode="bitfit_ln", ln_gamma_lr_scale=0.1)
    groups = {g["name"]: g for g in m.get_param_groups(3e-4)}
    assert groups["ln_gamma"]["lr"] == pytest.approx(3e-5) and groups["backbone"]["lr"] == 3e-4
    n_all = sum(p.numel() for g in groups.values() for p in g["params"])
    assert n_all == sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert "ln_gamma" not in {g["name"] for g in make(tuning_mode="bitfit_ln").get_param_groups(3e-4)}
    n = len(m.backbone.encoder.layers)
    r = make(tuning_mode="bitfit_ln", param_roles="v+out+fc2+ln_gamma+ln_beta", tune_blocks=f"0-{n - 2}")
    names = trainable_names(r)
    assert names and not any(f"layers.{n - 1}." in x for x in names)
    assert not any(k in x for x in names for k in ("q_proj", "k_proj", "fc1", "layrnorm", "post_layernorm"))


def test_feature_layers_last_matches_pooler_and_multi_differs():
    """feature_layers=<last block> reproduces the pooled CLS; several blocks average their CLS tokens."""
    m = make(tuning_mode="bitfit_ln").eval()
    x = batch()
    n = len(m.backbone.encoder.layers)
    base = m.features(x)
    m.config["feature_layers"] = str(n)
    torch.testing.assert_close(m.features(x), base)
    m.config["feature_layers"] = f"1,{n}"
    multi = m.features(x)
    assert multi.shape == base.shape and not torch.allclose(multi, base)
    m.train()
    m.get_losses(x, m(x))["overall"].backward()
    assert m.backbone.encoder.layers[0].mlp.fc2.bias.grad is not None
