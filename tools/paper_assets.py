#!/usr/bin/env python
"""Generate the revised manuscript's result tables (LaTeX) and figures (PDF)
from the experiment outputs, so the paper never contains hand-copied numbers.

    python tools/paper_assets.py --results results_paper/results --out paper/revision/generated

Layout read: <results>/<group>/<tag>/seed<k>/metrics.json  (group = val, roles_val, anchor_val, ...)
A tag found in several groups is merged by seed (the same seed is never counted twice).
Every table cell is mean +- std over the available seeds; the number of seeds is printed
in the table so preliminary (1-seed) rows are visible.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

D = ["Celeb-DF-v2", "DeepFakeDetection", "DFDC", "DFDCP"]
D_SHORT = ["CDF-v2", "DFD", "DFDC", "DFDCP"]
M = ["uniface_ff", "blendface_ff", "e4s_ff", "facedancer_ff", "fsgan_ff", "inswap_ff", "simswap_ff"]
M_SHORT = ["UniFace", "BlendFace", "E4S", "FaceDan.", "FSGAN", "InSwap", "SimSwap"]
FF = ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]
FF_SHORT = ["DF", "F2F", "FS", "NT"]
METRIC = "video_auc"

# label, tag, trainable params (M) of the deployed model
MAIN = [
    ("Linear probe", "linprobe", 0.002),
    ("Full FT (backbone lr $10^{-5}$)", "fullft", 303.18),
    ("Effort~\\cite{yan2025effort}", "effort", 0.199),
    ("GenD~\\cite{yermakov2025gend}", "gend", 0.104),
    ("BiasLN (bias + LN)", "biasln", 0.326),
    ("Bias only (BitFit)", "bitfit", 0.274),
    ("RA (Ours)", "ra_real", 0.274),
]
ROLES = [
    ("All biases incl.\ LN $\\beta$ (BitFit)", "bitfit", "272{,}384"),
    ("$\\mathbf{b}_q,\\mathbf{b}_{\\mathrm{fc1}}$ (content-dependent)", "bias_gate", "122{,}880"),
    ("$\\mathbf{b}_v,\\mathbf{b}_o,\\mathbf{b}_{\\mathrm{fc2}}$ (translations)", "bias_shift", "73{,}728"),
    ("LN $\\gamma,\\beta$", "ln_l2", "51{,}200"),
    ("LN $\\beta$ only", "ln_beta", "25{,}600"),
    ("LN $\\gamma$ only", "ln_gamma", "25{,}600"),
    ("BitFit + LN $\\gamma$ (BiasLN)", "biasln", "323{,}584"),
]
ANCHOR = [
    ("BitFit (no anchoring)", "bitfit"),
    ("+ anchor real only (RA)", "ra_real"),
    ("+ anchor fake only", "ra_fake"),
    ("+ anchor all images", "ra_all"),
    ("+ RA, $\\lambda=0.3$", "ra_real_l03"),
    ("+ RA, $\\lambda=3$", "ra_real_l3"),
    ("+ RA on $\\mathbf{b}_q,\\mathbf{b}_{\\mathrm{fc1}}$ only", "ra_gate"),
    ("+ real prototype loss (balanced)", "proto"),
    ("+ balanced sampling only", "bal"),
]

# Ablation in the style of the first version: tuned parameters | L_RA | anchored images | lambda.
# (label, L_RA, anchored, lambda, tag, is_ours)
ABLATION = [
    ("Full FT", "\\ding{55}", "--", "--", "fullft", False),
    ("Linear probe", "\\ding{55}", "--", "--", "linprobe", False),
    ("BiasLN (bias + LN)", "\\ding{55}", "--", "--", "biasln", False),
    ("BiasLN (bias + LN)", "\\ding{51}", "real", "1", "ra_biasln", False),
    ("Bias only", "\\ding{55}", "--", "--", "bitfit", False),
    ("Bias only", "\\ding{51}", "fake", "1", "ra_fake", False),
    ("Bias only", "\\ding{51}", "all", "1", "ra_all", False),
    ("Bias only", "\\ding{51}", "real", "0.3", "ra_real_l03", False),
    ("Bias only", "\\ding{51}", "real", "3", "ra_real_l3", False),
    ("Bias only (Ours)", "\\ding{51}", "real", "1", "ra_real", True),
]


ABL_HEAD = ["Tuned params", "$\\mathcal{L}_{\\mathrm{RA}}$", "\\textbf{Anchored}", "$\\lambda$"]
PERDATA_TAGS = ["effort", "gend", "biasln", "bitfit", "ra_real"]

# ---- BiasLN revision (the method of the submitted paper), selected with --paper biasln
MAIN_BIASLN = [
    ("Linear probe", "linprobe", 0.002),
    ("Full FT (backbone lr $10^{-5}$)", "fullft", 303.18),
    ("Effort~\\cite{yan2025effort}", "effort", 0.199),
    ("GenD~\\cite{yermakov2025gend}", "gend", 0.104),
    ("BiasLN (Ours)", "biasln", 0.326),
]
OURS_LIT_BIASLN = [("BiasLN (Ours)", "biasln", "0.33M")]
# old-paper ablation layout: PEFT | Head | L2 norm | Loss  (label, head, l2, loss, tag, is_ours)
ABLATION_BIASLN = [
    ("Full FT", "Linear", "Yes", "CE", "fullft", False),
    ("LoRA", "Linear", "No", "CE", "lora_nol2", False),
    ("LoRA", "Linear", "Yes", "CE", "lora", False),
    ("BitFit", "Linear", "No", "CE", "bitfit_nol2", False),
    ("BitFit", "Linear", "Yes", "CE", "bitfit", False),
    ("LayerNorm (LN)", "Linear", "No", "CE", "ln_nol2", False),
    ("LayerNorm (LN)", "Linear", "Yes", "CE", "ln_l2", False),
    ("BitFit + LN (BiasLN)", "Linear", "Yes", "CE", "biasln", True),
]
ABL_HEAD_BIASLN = ["PEFT Method", "\\textbf{Head}", "\\textbf{$L_2$ Norm}", "\\textbf{Loss}"]


# ---- review-style revision (paper/revision_v2), selected with --paper survey: no "(Ours)" row
MAIN_SURVEY = [
    ("Linear probe", "linprobe", 0.002),
    ("Full FT (backbone lr $10^{-5}$)", "fullft", 303.18),
    ("LoRA $r{=}8$", "lora", 0.788),
    ("Effort~\\cite{yan2025effort}", "effort", 0.199),
    ("Bias only (BitFit)", "bitfit", 0.274),
    ("BiasLN (bias + LN $\\gamma$)", "biasln", 0.326),
    ("LayerNorm only ($\\gamma,\\beta$)", "ln_l2", 0.104),
    ("GenD (LN + aux.\\ losses)~\\cite{yermakov2025gend}", "gend", 0.104),
]
ABLATION_SURVEY = [(r[0].replace(" (BiasLN)", " (BiasLN)"), *r[1:5], False) for r in ABLATION_BIASLN]
PERDATA_SURVEY = ["effort", "bitfit", "biasln", "ln_l2", "gend"]
SHORT = {}  # short figure labels (survey mode)
SHORT_SURVEY = {"linprobe": "Linear probe", "fullft": "Full FT", "lora": "LoRA", "effort": "Effort",
                "bitfit": "BitFit", "biasln": "BiasLN", "ln_l2": "LN only", "gend": "GenD"}


def highlight(vals, digits=3):
    """Old-paper convention: best value bold, second best shaded. Returns {index: style}."""
    r = sorted({round(v, digits) for v in vals if v is not None}, reverse=True)
    sty = {}
    for i, v in enumerate(vals):
        if v is None:
            continue
        if r and round(v, digits) == r[0]:
            sty[i] = "best"
        elif len(r) > 1 and round(v, digits) == r[1]:
            sty[i] = "second"
    return sty


def styled(txt, st):
    if st == "best":
        return f"\\textbf{{{txt}}}"
    if st == "second":
        return f"\\cellcolor{{gray!15}}{txt}"
    return txt


# ------------------------------------------------------------------ loading
def load(results: str):
    """tag -> seed -> {dataset: auc}"""
    runs: dict = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(results, "*", "*", "seed*", "metrics.json"))):
        parts = f.split(os.sep)
        tag, seed = parts[-3], parts[-2]
        if tag.endswith("_soup"):
            continue
        m = json.load(open(f)).get("metrics", {})
        runs[tag].setdefault(seed, {d: v.get(METRIC) for d, v in m.items() if isinstance(v, dict)})
    return runs


def stat(runs, tag, keys):
    """mean, std, n of a per-seed quantity (single dataset or average over keys)."""
    vals = []
    for per in runs.get(tag, {}).values():
        xs = [per.get(k) for k in keys]
        if all(x is not None for x in xs):
            vals.append(float(np.mean(xs)))
    if not vals:
        return None
    a = np.array(vals)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0), len(a)


def fmt(s, best=None, digits=3, std=True):
    if s is None:
        return "--"
    mu, sd, n = s
    txt = f"{mu:.{digits}f}"
    if best is not None and abs(mu - best) < 0.5 * 10 ** -digits:
        txt = f"\\textbf{{{txt}}}"
    if std and n > 1:
        txt += f"\\,{{\\scriptsize$\\pm${sd:.{digits}f}}}"
    return txt


def nseeds(runs, tag):
    return len(runs.get(tag, {}))


# ------------------------------------------------------------------ tables
def table_main(runs, out):
    rows = [(l, t, p) for l, t, p in MAIN if t in runs]
    cols = [[d] for d in D] + [D] + [[m] for m in M] + [M]
    stats = [[stat(runs, t, c) for c in cols] for _, t, _ in rows]
    styles = [highlight([st[j][0] if st[j] else None for st in stats]) for j in range(len(cols))]
    lines = [
        "\\begin{tabular}{l r c ccccc cccccccc}", "\\toprule",
        "\\multirow{2}{*}{\\textbf{Method}} & \\multirow{2}{*}{\\textbf{Params}} & \\multirow{2}{*}{$n$} & "
        "\\multicolumn{5}{c}{\\textbf{Cross-dataset}} & \\multicolumn{8}{c}{\\textbf{Cross-method}} \\\\",
        "\\cmidrule(lr){4-8} \\cmidrule(lr){9-16}",
        "& & & " + " & ".join(f"\\textbf{{{x}}}" for x in D_SHORT) + " & \\textbf{Avg(D)} & "
        + " & ".join(f"\\textbf{{{x}}}" for x in M_SHORT) + " & \\textbf{Avg(M)} \\\\", "\\midrule",
    ]
    for i, ((label, tag, p), st) in enumerate(zip(rows, stats)):
        if "(Ours)" in label:
            lines.append("\\midrule")
        cells = [styled(fmt(s_, std=(c in (D, M))), styles[j].get(i)) if s_ else "--"
                 for j, (s_, c) in enumerate(zip(st, cols))]
        lines.append(f"{label} & {p:g}M & {nseeds(runs, tag)} & " + " & ".join(cells) + " \\\\")
        if tag == "fullft":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(out, "tab_main.tex", lines)


def table_indomain(runs, out):
    """In-domain table in the layout of the first version (DF, F2F, FS, NT, Mean)."""
    rows = [(l, t) for l, t, _ in MAIN if t in runs and t not in ("fullft", "linprobe")]
    cols = [[f] for f in FF] + [FF]
    stats = [[stat(runs, t, c) for c in cols] for _, t in rows]
    styles = [highlight([100 * st[j][0] if st[j] else None for st in stats], digits=1) for j in range(len(cols))]
    lines = ["\\begin{tabular}{l cccc c}", "\\toprule",
             "\\textbf{Method} & " + " & ".join(f"\\textbf{{{x}}}" for x in FF_SHORT) + " & \\textbf{Mean} \\\\",
             "\\midrule"]
    for i, ((label, tag), st) in enumerate(zip(rows, stats)):
        if "(Ours)" in label:
            lines.append("\\midrule")
        cells = [styled("--" if s_ is None else f"{100 * s_[0]:.1f}", styles[j].get(i)) for j, s_ in enumerate(st)]
        lines.append(label + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(out, "tab_indomain.tex", lines)


def table_simple(runs, out, spec, name, with_params=False):
    rows = [r for r in spec if r[1] in runs]
    cols = [D, M, ["FaceForensics++"]]
    best = [max([s[0] for s in (stat(runs, r[1], c) for r in rows) if s] or [0]) for c in cols]
    head = "\\textbf{Tuned parameters} & \\textbf{\\#} &" if with_params else "\\textbf{Variant} &"
    lines = ["\\begin{tabular}{l " + ("r " if with_params else "") + "c ccc}", "\\toprule",
             head + " $n$ & \\textbf{Avg(D)} & \\textbf{Avg(M)} & \\textbf{FF++} \\\\", "\\midrule"]
    for r in rows:
        cells = [fmt(stat(runs, r[1], c), b) for c, b in zip(cols, best)]
        p = f" & {r[2]}" if with_params else ""
        lines.append(f"{r[0]}{p} & {nseeds(runs, r[1])} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(out, name, lines)


def table_ablation(runs, out):
    rows = [r for r in ABLATION if r[4] in runs]
    cols = [[d] for d in D] + [D, M]
    stats = [[stat(runs, r[4], c) for c in cols] for r in rows]
    styles = [highlight([st[j][0] if st[j] else None for st in stats]) for j in range(len(cols))]
    lines = ["\\begin{tabular}{l c c c c cccc c c}", "\\toprule",
             f"\\multirow{{2}}{{*}}{{\\textbf{{{ABL_HEAD[0]}}}}} & " + " & ".join(f"\\multirow{{2}}{{*}}{{{h}}}" for h in ABL_HEAD[1:]) +
             " & \\multirow{2}{*}{$n$} & "
             "\\multicolumn{4}{c}{\\textbf{Cross-dataset}} & \\multirow{2}{*}{\\textbf{Avg(D)}} & \\multirow{2}{*}{\\textbf{Avg(M)}} \\\\",
             "\\cmidrule(lr){6-9}",
             "& & & & & " + " & ".join(f"\\textbf{{{x}}}" for x in D_SHORT) + " & & \\\\", "\\midrule"]
    prev = None
    for i, (r, st) in enumerate(zip(rows, stats)):
        group = r[0].split(" (")[0]
        if r[5] or (prev is not None and group != prev):
            lines.append("\\midrule")
        prev = group
        cells = []
        for j, s_ in enumerate(st):
            txt = fmt(s_, std=(j >= len(D)))
            cells.append(styled(txt, styles[j].get(i)) if s_ else txt)
        name = f"\\textbf{{{r[0]}}}" if r[5] else r[0]
        lines.append(f"{name} & {r[1]} & {r[2]} & {r[3]} & {nseeds(runs, r[4])} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(out, "tab_ablation.tex", lines)


def table_perdataset_std(runs, out):
    """Per-dataset video AUC (x100) as mean +- std over seeds, two tables (cross-dataset, cross-method)."""
    rows = [(l, t) for l, t, _ in MAIN if t in runs]
    for name, keys, short in (("tab_perdata_D.tex", D, D_SHORT), ("tab_perdata_M.tex", M, M_SHORT)):
        cols = [[k] for k in keys] + [keys]
        stats = [[stat(runs, t, c) for c in cols] for _, t in rows]
        styles = [highlight([100 * st[j][0] if st[j] else None for st in stats], digits=2) for j in range(len(cols))]
        lines = ["\\begin{tabular}{l c " + "c" * len(cols) + "}", "\\toprule",
                 "\\textbf{Method} & $n$ & " + " & ".join(f"\\textbf{{{x}}}" for x in short) + " & \\textbf{Avg} \\\\", "\\midrule"]
        for i, ((label, tag), st) in enumerate(zip(rows, stats)):
            cells = []
            for j, s_ in enumerate(st):
                if s_ is None:
                    cells.append("--"); continue
                mu = f"{100 * s_[0]:.2f}"
                mu = styled(mu, styles[j].get(i))
                cells.append(mu + (f"\\,{{\\scriptsize$\\pm${100 * s_[1]:.2f}}}" if s_[2] > 1 else ""))
            lines.append(f"{label} & {nseeds(runs, tag)} & " + " & ".join(cells) + " \\\\")
            if tag == "fullft":
                lines.append("\\midrule")
        lines += ["\\bottomrule", "\\end{tabular}"]
        write(out, name, lines)


def write(out, name, lines):
    with open(os.path.join(out, name), "w") as f:
        f.write("% generated by tools/paper_assets.py -- do not edit\n" + "\n".join(lines) + "\n")


# Published numbers (video AUC, trained on FF++ c23). Source: Table 1 of Effort
# (Yan et al., ICML 2025, arXiv:2411.15633) for CDF-v2/DFD/DFDC/DFDCP and the 7 DF40
# methods; GenD and Forensics Adapter values as in the first manuscript version.
# Columns: CDF-v2, DFD, DFDC, DFDCP, (unused 5th value) | UniFace, BlendFace, E4S, FaceDan, FSGAN, InSwap, SimSwap
LITERATURE = [
    ("F3Net~\\cite{qian2020frequency}", "22M", [0.789, 0.844, 0.718, 0.749, 0.934], [0.809, 0.808, 0.494, 0.717, 0.845, 0.757, 0.674]),
    ("SPSL~\\cite{liu2021spsl}", "21M", [0.799, 0.871, 0.724, 0.770, 0.942], [0.747, 0.748, 0.514, 0.666, 0.812, 0.643, 0.665]),
    ("SRM~\\cite{luo2021srm}", "55M", [0.840, 0.885, 0.695, 0.728, 0.942], [0.749, 0.704, 0.704, 0.659, 0.772, 0.793, 0.694]),
    ("CORE~\\cite{ni2022core}", "22M", [0.809, 0.882, 0.721, 0.720, 0.941], [0.871, 0.843, 0.679, 0.774, 0.958, 0.855, 0.724]),
    ("RECCE~\\cite{cao2022recce}", "48M", [0.823, 0.891, 0.696, 0.734, 0.945], [0.898, 0.832, 0.683, 0.848, 0.949, 0.848, 0.768]),
    ("SLADD~\\cite{chen2022sladd}", "21M", [0.837, 0.904, 0.772, 0.756, None], [0.878, 0.882, 0.765, 0.825, 0.943, 0.879, 0.794]),
    ("SBI~\\cite{shiohara2022sbi}", "18M", [0.886, 0.827, 0.717, 0.848, None], [0.724, 0.891, 0.750, 0.594, 0.803, 0.712, 0.701]),
    ("UCF~\\cite{yan2023ucf}", "47M", [0.837, 0.867, 0.742, 0.770, None], [0.831, 0.827, 0.731, 0.862, 0.937, 0.809, 0.647]),
    ("IID~\\cite{huang2023iid}", "66M", [0.838, 0.939, 0.700, 0.689, None], [0.839, 0.789, 0.766, 0.844, 0.927, 0.789, 0.644]),
    ("LSDA$^\\dagger$~\\cite{yan2024latent}", "133M", [0.875, 0.881, 0.701, 0.812, None], [0.872, 0.875, 0.694, 0.721, 0.939, 0.855, 0.793]),
    ("ProDet$^\\dagger$~\\cite{cheng2024prodet}", "96M", [0.926, 0.901, 0.707, 0.828, None], [0.908, 0.929, 0.771, 0.747, 0.928, 0.837, 0.844]),
    ("CDFA$^\\dagger$~\\cite{lin2024cdfa}", "87M", [0.938, 0.954, 0.830, 0.881, None], [0.762, 0.756, 0.631, 0.803, 0.942, 0.772, 0.757]),
    ("Effort~\\cite{yan2025effort}", "0.19M", [0.956, 0.965, 0.843, 0.909, 0.974], [0.962, 0.873, 0.983, 0.926, 0.957, 0.936, 0.926]),
    ("Forensics Adapter~\\cite{cui2025forada}", "5.7M", [0.957, 0.972, 0.872, 0.928, 0.994], [0.950, 0.880, 0.968, 0.945, 0.975, 0.948, 0.912]),
    ("GenD~\\cite{yermakov2025gend}", "0.1M", [0.960, 0.970, 0.870, 0.917, 0.992], [0.983, 0.933, 0.988, 0.954, 0.9817, 0.965, 0.9577]),
]
D4 = D[:4]
OURS_LIT = [("RA (Ours)", "ra_real", "0.27M")]


def table_literature(runs, out):
    """Published numbers + our method, formatted like Table 2 of the first version
    (best per column bold, second best shaded)."""
    rows = []
    for name, p, d, m in LITERATURE:
        d = list(d[:4])
        rows.append((name, p, d + [sum(d) / 4] + list(m) + [sum(m) / len(m)]))
    for name, tag, p in OURS_LIT:
        if tag in runs:
            vals = [stat(runs, tag, [k]) for k in D] + [stat(runs, tag, D)] + \
                   [stat(runs, tag, [k]) for k in M] + [stat(runs, tag, M)]
            rows.append((f"{name} ($n{{=}}{nseeds(runs, tag)}$)", p, [v[0] if v else None for v in vals]))
    ncol = len(rows[0][2])
    styles = [highlight([r[2][j] for r in rows]) for j in range(ncol)]
    lines = ["\\begin{tabular}{l r ccccc cccccccc}", "\\toprule",
             "\\multirow{2}{*}{\\textbf{Method}} & \\multirow{2}{*}{\\textbf{Trainable Params}} & "
             "\\multicolumn{5}{c}{\\textbf{Cross-dataset}} & \\multicolumn{8}{c}{\\textbf{Cross-method}} \\\\",
             "\\cmidrule(lr){3-7} \\cmidrule(lr){8-15}",
             "& & " + " & ".join(f"\\textbf{{{x}}}" for x in D_SHORT) + " & \\textbf{Avg(D)} & "
             + " & ".join(f"\\textbf{{{x}}}" for x in M_SHORT) + " & \\textbf{Avg(M)} \\\\", "\\midrule"]
    for i, (name, p, vals) in enumerate(rows):
        if i == len(LITERATURE):
            lines.append("\\midrule")
        cells = [styled("--" if v is None else f"{v:.3f}", styles[j].get(i)) for j, v in enumerate(vals)]
        lines.append(f"{name} & {p} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(out, "tab_literature.tex", lines)


# ------------------------------------------------------------------ figures
def figures(runs, out, epoch_tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "serif", "pdf.fonttype": 42})
    palette = {"effort": "#8c8c8c", "gend": "#4c72b0", "bitfit": "#dd8452", "biasln": "#937860",
               "ra_real": "#c44e52", "ra_biasln": "#8c2d2d", "fullft": "#999999", "linprobe": "#bbbbbb",
               "ln_l2": "#55a868", "lora": "#8172b3"}
    names = {t: l.split("~")[0].replace("aux.\\ ", "aux. ") for l, t, _ in MAIN}
    names.update({k: v for k, v in SHORT.items() if k in names})

    # Fig A: per-dataset video AUC (cross-dataset), grouped bars with seed std
    tags = [t for t in PERDATA_TAGS if t in runs and t in names]
    if tags:
        fig, ax = plt.subplots(figsize=(3.4, 2.3) if SHORT else (7.0, 2.3))
        w = 0.8 / len(tags)
        x = np.arange(len(D) + 1)
        for i, t in enumerate(tags):
            ss = [stat(runs, t, [d]) for d in D] + [stat(runs, t, D)]
            mu = [s[0] if s else np.nan for s in ss]
            sd = [s[1] if s else 0 for s in ss]
            ax.bar(x + (i - (len(tags) - 1) / 2) * w, mu, w, yerr=sd, capsize=1.5, label=names[t],
                   color=palette.get(t), edgecolor="white", linewidth=0.4, error_kw={"lw": 0.6})
        ax.set_xticks(x, D_SHORT + ["Avg(D)"])
        ax.set_ylim(0.82, 1.0)
        ax.set_ylabel("Video AUC")
        ax.legend(ncol=len(tags), frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18),
                  fontsize=6 if SHORT else 8, handlelength=1.0, columnspacing=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_perdataset.pdf"), bbox_inches="tight")
        plt.close(fig)

    # Fig B: anchoring controls (which images are anchored)
    ctrl = [("none", "bitfit"), ("fake", "ra_fake"), ("all", "ra_all"), ("real (RA)", "ra_real")]
    ctrl = [c for c in ctrl if c[1] in runs]
    if len(ctrl) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9), sharey=False)
        for ax, keys, title in zip(axes, [D, M], ["Avg(D)", "Avg(M)"]):
            ss = [stat(runs, t, keys) for _, t in ctrl]
            ax.bar(range(len(ctrl)), [s[0] for s in ss], yerr=[s[1] for s in ss], capsize=2,
                   color=["#cccccc", "#9ab", "#79a", "#c44e52"][-len(ctrl):] if len(ctrl) == 4 else "#79a")
            lo = min(s[0] - s[1] for s in ss)
            ax.set_ylim(lo - 0.006, max(s[0] + s[1] for s in ss) + 0.004)
            ax.set_xticks(range(len(ctrl)), [c[0] for c in ctrl], rotation=25, ha="right")
            ax.set_title(title, fontsize=8)
        axes[0].set_ylabel("Video AUC")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_anchor_controls.pdf"), bbox_inches="tight")
        plt.close(fig)

    # Fig C: trainable parameters vs Avg(D) / Avg(M) (efficiency trade-off)
    def clean(l):
        l = l.split("~")[0].replace(" (backbone lr $10^{-5}$)", "").replace("aux.\\ ", "aux. ")
        return l.replace("$r{=}8$", "r=8")
    pts = [(SHORT.get(t, clean(l)), t, p) for l, t, p in MAIN if t in runs and stat(runs, t, D)]
    if pts:
        markers = "osD^vPX*h"
        fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.6)) if SHORT else plt.subplots(1, 2, figsize=(7.0, 2.6))
        for ax, keys, yl in zip(axes, [D, M], ["Avg(D) video AUC", "Avg(M) video AUC"]):
            ins = ax.inset_axes([0.33, 0.30, 0.52, 0.62])
            zoom = []
            for k, (l, t, p) in enumerate(pts):
                s_ = stat(runs, t, keys)
                if not s_:
                    continue
                kw = dict(fmt=markers[k % len(markers)], ms=4, color=palette.get(t, "#333333"), capsize=2)
                px = p * (1.12 if t == "gend" else 1.0)  # GenD and LN-only share the budget
                ax.errorbar(px, s_[0], yerr=s_[1], label=l, **kw)
                if 0.05 < p < 2:
                    ins.errorbar(px, s_[0], yerr=s_[1], **kw); zoom.append(s_)
            ax.set_xscale("log"); ins.set_xscale("log")
            if zoom:
                lo = min(z[0] - z[1] for z in zoom); hi = max(z[0] + z[1] for z in zoom)
                ins.set_ylim(lo - 0.002, hi + 0.002); ins.set_xlim(0.07, 1.3)
            ins.tick_params(labelsize=5.5)
            ax.indicate_inset_zoom(ins, edgecolor="#888888")
            ax.set_xlabel("Trainable parameters (M, log scale)")
            ax.set_ylabel(yl)
        h, l_ = axes[0].get_legend_handles_labels()
        fig.legend(h, l_, ncol=4, frameon=False, fontsize=6 if SHORT else 6.5, loc="upper center",
                   bbox_to_anchor=(0.5, 1.07 if SHORT else 1.12), handlelength=1.0, columnspacing=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_params_auc.pdf"), bbox_inches="tight")
        plt.close(fig)

    # Fig D: per-epoch FF++-val AUC vs CDF-v2 test AUC (selection inflation)
    if epoch_tables:
        fig, axes = plt.subplots(1, len(epoch_tables), figsize=(2.3 * len(epoch_tables), 1.9), squeeze=False)
        for ax, (name, et) in zip(axes[0], epoch_tables.items()):
            ep = sorted(et["per_epoch"], key=int)
            cdf = [et["per_epoch"][e].get("Celeb-DF-v2") for e in ep]
            val = [et["per_epoch"][e].get("val") for e in ep]
            ax.plot([int(e) + 1 for e in ep], cdf, "o-", ms=2.5, color="#c44e52", label="CDF-v2 test")
            if any(v is not None for v in val):
                ax.plot([int(e) + 1 for e in ep], val, "s--", ms=2.5, color="#4c72b0", label="FF++ val")
            sel = et.get("selected_epoch")
            if sel is not None:
                ax.axvline(int(sel) + 1, color="k", lw=0.6, ls=":")
            ax.set_title(name, fontsize=8)
            ax.set_xlabel("Epoch")
        axes[0][0].set_ylabel("Video AUC")
        axes[0][0].legend(frameon=False, fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig_epochs.pdf"), bbox_inches="tight")
        plt.close(fig)


def numbers(runs, out):
    """Key numbers as LaTeX macros, e.g. \\AvgDra, so the prose never drifts from the tables."""
    macros = []
    for tag in sorted(runs):
        for key, cols in (("D", D), ("M", M), ("FF", ["FaceForensics++"])):
            s = stat(runs, tag, cols)
            if s:
                name = "".join(ch if ch.isalpha() else "ABCDEFGHIJ"[int(ch)] if ch.isdigit() else ""
                               for ch in tag.title())
                macros.append(f"\\newcommand{{\\Avg{key}{name}}}{{{s[0]:.3f}}}")
                macros.append(f"\\newcommand{{\\Std{key}{name}}}{{{s[1]:.3f}}}")
                macros.append(f"\\newcommand{{\\N{key}{name}}}{{{s[2]}}}")
    write(out, "numbers.tex", macros)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--paper", choices=["biasln", "ra", "survey"], default="biasln",
                    help="biasln: the revision of the submitted BiasLN paper; ra: the Real Anchoring paper")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epoch_tables", nargs="*", default=[], help="name=path/to/epoch_table.json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.paper == "biasln":
        global MAIN, OURS_LIT, ABLATION, ABL_HEAD, PERDATA_TAGS
        MAIN, OURS_LIT, ABLATION, ABL_HEAD = MAIN_BIASLN, OURS_LIT_BIASLN, ABLATION_BIASLN, ABL_HEAD_BIASLN
        PERDATA_TAGS = ["effort", "gend", "biasln"]
    if a.paper == "survey":
        MAIN, OURS_LIT, ABLATION, ABL_HEAD = MAIN_SURVEY, [], ABLATION_SURVEY, ABL_HEAD_BIASLN
        PERDATA_TAGS = PERDATA_SURVEY
        SHORT.update(SHORT_SURVEY)
    runs = load(a.results)
    print({t: sorted(s) for t, s in runs.items()})
    table_main(runs, a.out)
    table_indomain(runs, a.out)
    table_literature(runs, a.out)
    table_simple(runs, a.out, ROLES, "tab_roles.tex", with_params=True)
    table_simple(runs, a.out, ANCHOR, "tab_anchor.tex")
    table_ablation(runs, a.out)
    table_perdataset_std(runs, a.out)
    numbers(runs, a.out)
    et = {}
    for spec in a.epoch_tables:
        k, p = spec.split("=", 1)
        if os.path.exists(p):
            et[k] = json.load(open(p))
    figures(runs, a.out, et)


if __name__ == "__main__":
    main()
