#!/usr/bin/env python
"""Feature-space figures from frame features saved by `test.py --save_feat`.

    python tools/feature_figs.py --feat_dir results_paper/features --out paper/revision/generated

Expects <feat_dir>/<model>/scores/<dataset>.npz with frame_feat, frame_label, frame_name.
`clip` (the frozen encoder, `test.py --pretrained`) is the reference for the drift figure.

fig_drift.pdf   distribution of cos(f_tuned(x), f_clip(x)) for real and fake test frames:
                how far tuning moved each class away from the pre-trained features
fig_tsne.pdf    t-SNE of video-level features (mean over frames), real vs fake
drift.json      summary numbers (median drift per class, Fisher ratio per model)
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

LABELS = {"clip": "Frozen CLIP", "bitfit": "BitFit", "ra_real": "BitFit + RA",
          "biasln": "BiasLN", "gend": "GenD", "effort": "Effort", "fullft": "Full FT"}


def load(feat_dir, model, dataset):
    p = os.path.join(feat_dir, model, "scores", f"{dataset}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    return {"feat": z["frame_feat"].astype(np.float32), "label": z["frame_label"].astype(int),
            "name": z["frame_name"]}


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(1e-8)


def video_mean(d):
    keys = np.array([os.path.dirname(n) for n in d["name"]])
    uniq, inv = np.unique(keys, return_inverse=True)
    f = np.zeros((len(uniq), d["feat"].shape[1]))
    np.add.at(f, inv, unit(d["feat"]))
    lab = np.zeros(len(uniq))
    np.add.at(lab, inv, d["label"])
    cnt = np.bincount(inv)
    return unit(f / cnt[:, None]), (lab / cnt > 0.5).astype(int)


def fisher(f, y):
    """Between-class / within-class scatter of unit features (higher = more separable)."""
    mu = f.mean(0)
    sb = sum((y == c).sum() * np.sum((f[y == c].mean(0) - mu) ** 2) for c in (0, 1))
    sw = sum(np.sum((f[y == c] - f[y == c].mean(0)) ** 2) for c in (0, 1))
    return float(sb / max(sw, 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default="Celeb-DF-v2")
    ap.add_argument("--models", nargs="+", default=["clip", "effort", "gend", "biasln", "bitfit", "ra_real"])
    ap.add_argument("--no_drift", action="store_true", help="skip the drift figure")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    os.makedirs(a.out, exist_ok=True)
    data = {m: load(a.feat_dir, m, a.dataset) for m in a.models}
    data = {m: d for m, d in data.items() if d is not None}
    summary = {}

    ref = data.get("clip")
    if ref is not None and not a.no_drift:
        drift_models = [m for m in ("bitfit", "ra_real", "biasln", "gend") if m in data]
        fig, axes = plt.subplots(1, len(drift_models), figsize=(1.75 * len(drift_models), 1.8),
                                 sharey=True, squeeze=False)
        for ax, m in zip(axes[0], drift_models):
            d = data[m]
            assert np.array_equal(d["name"], ref["name"]), f"frame order differs for {m}"
            cos = np.sum(unit(d["feat"]) * unit(ref["feat"]), axis=1)
            bins = np.linspace(min(0.0, cos.min()), 1.0, 41)
            for c, col, lab in ((0, "#4c72b0", "real"), (1, "#c44e52", "fake")):
                ax.hist(cos[d["label"] == c], bins=bins, density=True, alpha=0.55, color=col, label=lab)
            ax.set_title(LABELS.get(m, m), fontsize=8)
            ax.set_xlabel(r"$\cos(f_\theta, f_{\theta_0})$")
            summary[m] = {"median_cos_real": float(np.median(cos[d["label"] == 0])),
                          "median_cos_fake": float(np.median(cos[d["label"] == 1]))}
        axes[0][0].set_ylabel("Density")
        axes[0][0].legend(frameon=False, fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(a.out, "fig_drift.pdf"), bbox_inches="tight")
        plt.close(fig)

    show = [m for m in a.models if m in data]
    fig, axes = plt.subplots(1, len(show), figsize=(1.45 * len(show), 1.6), squeeze=False)
    for ax, m in zip(axes[0], show):
        f, y = video_mean(data[m])
        summary.setdefault(m, {})["fisher_video"] = fisher(f, y)
        emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(f)
        for c, col, lab in ((0, "#4c72b0", "real"), (1, "#c44e52", "fake")):
            ax.scatter(emb[y == c, 0], emb[y == c, 1], s=2, c=col, alpha=0.6, label=lab, linewidths=0)
        ax.set_title(LABELS.get(m, m), fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0][0].legend(frameon=False, fontsize=6, markerscale=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_tsne.pdf"), bbox_inches="tight")
    plt.close(fig)
    json.dump(summary, open(os.path.join(a.out, "drift.json"), "w"), indent=1)
    with open(os.path.join(a.out, "drift_numbers.tex"), "w") as f:
        f.write("% generated by tools/feature_figs.py -- do not edit\n")
        for m, v in summary.items():
            name = "".join(ch for ch in m.title() if ch.isalpha())
            for k, mac in (("median_cos_real", "DriftReal"), ("median_cos_fake", "DriftFake"),
                           ("fisher_video", "Fisher")):
                if k in v:
                    f.write(f"\\newcommand{{\\{mac}{name}}}{{{v[k]:.2f}}}\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
