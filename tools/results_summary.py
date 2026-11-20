#!/usr/bin/env python
"""Markdown summary (Vietnamese) of every run so far, regenerated from the run outputs.

    python tools/results_summary.py --results results_paper/results \
        --drift paper/revision/generated/drift.json --out docs/results_summary.md

Layout read: <results>/<group>/<tag>/seed<k>/metrics.json (same as tools/paper_assets.py).
Soup runs (tag *_soup) are listed separately; all other tags are merged across groups by seed.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from paper_assets import D, D_SHORT, FF, FF_SHORT, M, M_SHORT, METRIC  # noqa: E402


def load(results):
    runs, soups = defaultdict(dict), defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(results, "*", "*", "seed*", "metrics.json"))):
        parts = f.split(os.sep)
        tag, seed = parts[-3], parts[-2].replace("seed", "")
        m = json.load(open(f)).get("metrics", {})
        per = {d: v.get(METRIC) for d, v in m.items() if isinstance(v, dict)}
        (soups if tag.endswith("_soup") else runs)[tag].setdefault(seed, per)
    return runs, soups


def agg(per_seed, keys):
    vals = [float(np.mean([p[k] for k in keys])) for p in per_seed.values()
            if all(p.get(k) is not None for k in keys)]
    if not vals:
        return None
    a = np.array(vals)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else None), len(a)


def cell(s, std=True):
    if s is None:
        return "–"
    mu, sd, n = s
    return f"{mu:.4f} ± {sd:.4f}" if (std and sd is not None) else f"{mu:.4f}"


def seeds(runs, tag):
    return ", ".join(sorted(runs.get(tag, {}), key=lambda s: int(s)))


def table(runs, rows, cols, header, std_cols=()):
    rows = [r for r in rows if r[1] in runs]
    out = ["| " + " | ".join(["Cấu hình", "Seeds"] + header) + " |",
           "|" + "---|" * (len(header) + 2)]
    for label, tag in rows:
        cells = [cell(agg(runs[tag], c), std=(i in std_cols)) for i, c in enumerate(cols)]
        out.append(f"| {label} | {seeds(runs, tag)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


MAIN = [("Full FT (lr backbone 1e-5)", "fullft"), ("Linear probe", "linprobe"),
        ("Effort (tái huấn luyện)", "effort"), ("GenD (tái huấn luyện)", "gend"),
        ("BiasLN (bias + LN, bản v1)", "biasln"), ("Bias only (BitFit)", "bitfit"),
        ("**RA (bias only + neo real, λ=1) — đề xuất**", "ra_real"), ("BiasLN + RA", "ra_biasln")]
ANCHOR = [("Bias only, không RA", "bitfit"), ("+ neo real (RA, λ=1)", "ra_real"),
          ("+ neo fake", "ra_fake"), ("+ neo tất cả ảnh", "ra_all"),
          ("+ RA λ=0.3", "ra_real_l03"), ("+ RA λ=3", "ra_real_l3"),
          ("+ RA chỉ trên b_q, b_fc1", "ra_gate"), ("+ real prototype loss (cân bằng batch)", "proto"),
          ("+ RA + prototype", "ra_proto"), ("+ chỉ cân bằng batch", "bal")]
ROLES = [("Tất cả bias kể cả LN β (BitFit), 272,384", "bitfit"),
         ("b_q, b_fc1 (phụ thuộc nội dung), 122,880", "bias_gate"),
         ("b_v, b_o, b_fc2 (tịnh tiến), 73,728", "bias_shift"),
         ("LN γ, β, 51,200", "ln_l2"), ("Chỉ LN β, 25,600", "ln_beta"), ("Chỉ LN γ, 25,600", "ln_gamma"),
         ("BitFit + LN γ (BiasLN), 323,584", "biasln")]
EARLY = [("BiasLN", "biasln"), ("BiasLN + asymmetric loss", "biasln_asy"),
         ("GenD", "gend"), ("Effort", "effort")]

# First-version numbers (CDF-v2 subset used for checkpoint selection, single run)
V1 = {"CDF-v2": 0.971, "DFD": 0.977, "DFDC": 0.862, "DFDCP": 0.925}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--drift", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs, soups = load(a.results)
    dcols = [[d] for d in D] + [D]
    mcols = [[m] for m in M] + [M]
    L = []
    w = L.append
    w("# Tổng hợp kết quả thí nghiệm — Real Anchoring (bản sửa của BiasLN)\n")
    w(f"*Sinh tự động bởi `tools/results_summary.py` lúc {datetime.datetime.now():%Y-%m-%d %H:%M}. "
      "Chạy lại script để cập nhật khi có seed mới.*\n")
    w("## Protocol chung\n")
    w("- Train trên FF++ c23 (4 manipulation), 10 epoch, batch 32, bf16, 1×A100; 8 frame/video khi train, 32 frame khi test.")
    w("- **Chọn checkpoint trên FF++ val** (frame AUC), không dùng dữ liệu tập test nào.")
    w("- Chỉ số: **video AUC** (trung bình xác suất fake của các frame trong video).")
    w("- Avg(D) = trung bình 4 tập cross-dataset (CDF-v2, DFD, DFDC, DFDCP, theo protocol Effort). "
      "Avg(M) = trung bình 7 phương pháp face-swap của DF40.")
    w("- `±` là độ lệch chuẩn mẫu giữa các seed; không có `±` nghĩa là mới có 1 seed.\n")

    w("## 1. So sánh chính (cùng protocol)\n")
    w("### Cross-dataset\n")
    w(table(runs, MAIN, dcols, D_SHORT + ["**Avg(D)**"], std_cols=(4,)) + "\n")
    w("### Cross-method (DF40)\n")
    w(table(runs, MAIN, mcols, M_SHORT + ["**Avg(M)**"], std_cols=(7,)) + "\n")
    w("### In-domain (FF++ test)\n")
    w(table(runs, MAIN, [[f] for f in FF] + [FF] + [["FaceForensics++"]], FF_SHORT + ["Mean", "FF++ all"]) + "\n")

    w("## 2. Ablation Real Anchoring (trên bias only)\n")
    w(table(runs, ANCHOR, dcols + [M], D_SHORT + ["**Avg(D)**", "**Avg(M)**"], std_cols=(4, 5)) + "\n")

    w("## 3. Tune phần tham số nào (không có RA)\n")
    w(table(runs, ROLES, [D, M, ["FaceForensics++"]], ["**Avg(D)**", "**Avg(M)**", "FF++"], std_cols=(0, 1, 2)) + "\n")

    if a.drift and os.path.exists(a.drift):
        dr = json.load(open(a.drift))
        w("## 4. Độ lệch feature so với CLIP gốc (CDF-v2 test, seed 1024)\n")
        w("Median cos(f_tuned, f_CLIP) theo frame; gần 1 = giữ nguyên biểu diễn pre-trained. "
          "Fisher = độ tách real/fake của feature mức video (cao = tách tốt hơn trong không gian feature).\n")
        w("| Model | cos real | cos fake | Fisher (video) |\n|---|---|---|---|")
        names = {"clip": "CLIP đóng băng", "effort": "Effort", "gend": "GenD", "biasln": "BiasLN",
                 "bitfit": "Bias only", "ra_real": "RA"}
        for k in ["clip", "effort", "gend", "biasln", "bitfit", "ra_real"]:
            if k in dr:
                v = dr[k]
                f = lambda x: "–" if x is None else f"{x:.3f}"
                w(f"| {names[k]} | {f(v.get('median_cos_real'))} | {f(v.get('median_cos_fake'))} | {f(v.get('fisher_video'))} |")
        w("")

    w("## 5. Thí nghiệm sớm (trước khi chọn RA)\n")
    w("### Asymmetric loss và model soup\n")
    early = [r for r in EARLY if r[1] in runs]
    w(table(runs, early, [D, M, ["FaceForensics++"]], ["**Avg(D)**", "**Avg(M)**", "FF++"], std_cols=(0, 1)) + "\n")
    if soups:
        w("Model soup (trung bình trọng số các checkpoint của cùng một run):\n")
        w(table(soups, [(t.replace("_soup", "") + " soup", t) for t in sorted(soups)],
                [D, M, ["FaceForensics++"]], ["**Avg(D)**", "**Avg(M)**", "FF++"], std_cols=(0, 1)) + "\n")

    w("### Số liệu bản v1 (để đối chiếu)\n")
    w("Bản nộp đầu chọn checkpoint trên một phần CDF-v2 (cũng là tập test), 1 lần chạy:\n")
    w("| | " + " | ".join(V1) + " | Avg(D) 4 tập |\n|---|" + "---|" * (len(V1) + 1))
    w("| BiasLN v1 | " + " | ".join(f"{v:.3f}" for v in V1.values()) + f" | {np.mean(list(V1.values())):.4f} |")
    b = agg(runs.get("biasln", {}), D)
    if b:
        w(f"| BiasLN, protocol mới | " + " | ".join(cell(agg(runs['biasln'], [d]), std=False) for d in D)
          + f" | {cell(b)} |")
    w("")
    open(a.out, "w").write("\n".join(L) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
