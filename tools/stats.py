#!/usr/bin/env python
"""Multi-seed aggregation and significance testing for video-level AUC.

Expected layout (produced by scripts/run_protocol.sh):

    results/<model>/seed<k>/metrics.json
    results/<model>/seed<k>/scores/<dataset>.npz     (video, score, label)

Usage:
    python tools/stats.py --results results --reference biasln \
        --models biasln gend effort --out results/summary

Produces <out>.md and <out>.json with:
  * mean +- std (and 95% t-interval) of video AUC over seeds, per dataset and
    for the cross-dataset / cross-method averages;
  * paired tests over seeds (reference vs each baseline): paired t-test and
    exact sign-flip permutation test, Holm-corrected across datasets;
  * per-seed DeLong tests on the *same* test videos, plus a DeLong test of the
    seed-ensembled scores (mean score over seeds);
  * a stratified bootstrap 95% CI of the AUC difference of the ensembled scores.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats as sps

CROSS_DATASET = ["Celeb-DF-v2", "DeepFakeDetection", "DFDC", "DFDCP", "UADFV"]
CROSS_METHOD = ["uniface_ff", "blendface_ff", "e4s_ff", "facedancer_ff",
                "fsgan_ff", "inswap_ff", "simswap_ff"]


# ----------------------------------------------------------------- DeLong
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def _fast_delong(preds_sorted: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """Sun & Xu (2014). preds_sorted: [k, n] with the m positives first."""
    n = preds_sorted.shape[1] - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.stack([_midrank(pos[r]) for r in range(k)])
    ty = np.stack([_midrank(neg[r]) for r in range(k)])
    tz = np.stack([_midrank(preds_sorted[r]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def delong_test(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> Dict[str, float]:
    """Two-sided DeLong test for two correlated ROC AUCs on the same samples."""
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")  # positives (label 1 = fake) first
    m = int(y.sum())
    preds = np.vstack([np.asarray(s1, float), np.asarray(s2, float)])[:, order]
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    diff = aucs[0] - aucs[1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    p = 2 * sps.norm.sf(abs(z)) if var > 0 else 1.0
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "diff": float(diff),
            "z": float(z), "p": float(p)}


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")
    return float(_fast_delong(np.asarray(s, float)[None, order], int(y.sum()))[0][0])


def bootstrap_diff_ci(y, s1, s2, n_boot=2000, seed=0, alpha=0.05):
    """Stratified bootstrap CI of AUC(s1) - AUC(s2) on the same videos."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        diffs[b] = auc(y[idx], s1[idx]) - auc(y[idx], s2[idx])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# ------------------------------------------------------------- seed tests
def sign_flip_test(d: np.ndarray) -> float:
    """Exact two-sided paired permutation (sign-flip) test of mean(d) = 0."""
    d = np.asarray(d, float)
    obs = abs(d.mean())
    flips = np.array(list(itertools.product([-1, 1], repeat=len(d))))
    null = np.abs((flips * d).mean(axis=1))
    return float((null >= obs - 1e-12).mean())


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, running, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def mean_ci(x: np.ndarray, alpha=0.05) -> Tuple[float, float, float]:
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()), float("nan"), float("nan")
    half = sps.t.ppf(1 - alpha / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - half), float(x.mean() + half)


# --------------------------------------------------------------- loading
def load_model_runs(root: str, model: str, metric: str):
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, model, "*"))):
        mpath = os.path.join(d, "metrics.json")
        if not os.path.isfile(mpath):
            continue
        with open(mpath) as f:
            metrics = json.load(f)["metrics"]
        scores = {}
        for npz in glob.glob(os.path.join(d, "scores", "*.npz")):
            z = np.load(npz, allow_pickle=False)
            scores[os.path.basename(npz)[:-4]] = (z["video"], z["score"], z["label"])
        runs[os.path.basename(d)] = {
            "metric": {k: v.get(metric, v.get("video_auc")) for k, v in metrics.items()},
            "scores": scores,
        }
    return runs


def aligned(scores_a, scores_b):
    va, sa, ya = scores_a
    vb, sb, yb = scores_b
    idx_b = {v: i for i, v in enumerate(vb)}
    common = [i for i, v in enumerate(va) if v in idx_b]
    j = [idx_b[va[i]] for i in common]
    if len(common) != len(va) or len(j) != len(vb):
        print(f"  [warn] video sets differ: {len(va)} vs {len(vb)}, using {len(common)} common")
    return ya[common], sa[common], sb[j]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--metric", default="video_auc_dir",
                    help="metrics.json key (video_auc_dir = per-directory video AUC)")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--out", default="results/summary")
    args = ap.parse_args()

    runs = {m: load_model_runs(args.results, m, args.metric) for m in args.models}
    for m, r in runs.items():
        print(f"{m}: {len(r)} seeds -> {sorted(r)}")
    datasets = args.datasets or sorted(
        set().union(*[set(rr["metric"]) for r in runs.values() for rr in r.values()]))
    groups = {"Avg(D)": [d for d in CROSS_DATASET if d in datasets],
              "Avg(M)": [d for d in CROSS_METHOD if d in datasets]}

    table: Dict[str, Dict[str, dict]] = {}
    per_seed: Dict[str, Dict[str, Dict[str, float]]] = {}
    for m, r in runs.items():
        table[m], per_seed[m] = {}, {}
        for d in datasets:
            vals = {s: rr["metric"][d] for s, rr in r.items() if d in rr["metric"]}
            per_seed[m][d] = vals
        for g, members in groups.items():
            if not members:
                continue
            seeds = [s for s in r if all(d in r[s]["metric"] for d in members)]
            per_seed[m][g] = {s: float(np.mean([r[s]["metric"][d] for d in members])) for s in seeds}
        for key, vals in per_seed[m].items():
            x = np.array(list(vals.values()), float)
            if len(x) == 0:
                continue
            mu, lo, hi = mean_ci(x)
            table[m][key] = {"mean": mu, "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                             "ci95": [lo, hi], "n_seeds": int(len(x))}

    tests: Dict[str, dict] = {}
    ref = args.reference
    for m in args.models:
        if m == ref:
            continue
        tests[m] = {}
        raw_p = {}
        for key in list(datasets) + [g for g in groups if groups[g]]:
            a, b = per_seed[ref].get(key, {}), per_seed[m].get(key, {})
            common = sorted(set(a) & set(b))
            entry = {"paired_seeds": len(common)}
            if len(common) >= 2:
                d = np.array([a[s] - b[s] for s in common])
                if np.allclose(d, d[0]):  # zero variance: t-test undefined
                    t_p = 1.0 if np.isclose(d[0], 0.0) else float("nan")
                else:
                    t_p = float(sps.ttest_rel([a[s] for s in common], [b[s] for s in common]).pvalue)
                entry.update({"mean_diff": float(d.mean()), "t_p": t_p,
                              "signflip_p": sign_flip_test(d)})
                if not np.isnan(t_p):
                    raw_p[key] = t_p
            if key in datasets:
                delong_ps, ens_a, ens_b, y_ref = [], [], [], None
                for s in common:
                    sa = runs[ref][s]["scores"].get(key)
                    sb = runs[m][s]["scores"].get(key)
                    if sa is None or sb is None:
                        continue
                    y, xa, xb = aligned(sa, sb)
                    delong_ps.append(delong_test(y, xa, xb)["p"])
                    if y_ref is None or len(y) == len(y_ref):
                        y_ref = y
                        ens_a.append(xa)
                        ens_b.append(xb)
                if delong_ps:
                    entry["delong_p_per_seed"] = delong_ps
                    entry["delong_significant_seeds"] = int(sum(p < 0.05 for p in delong_ps))
                    ea, eb = np.mean(ens_a, axis=0), np.mean(ens_b, axis=0)
                    entry["ensemble_delong"] = delong_test(y_ref, ea, eb)
                    entry["ensemble_boot_ci_diff"] = bootstrap_diff_ci(y_ref, ea, eb, args.n_boot)
            tests[m][key] = entry
        for key, p in holm({k: v for k, v in raw_p.items() if k in datasets}).items():
            tests[m][key]["t_p_holm"] = p

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump({"table": table, "per_seed": per_seed, "tests": tests,
                   "reference": ref, "metric": args.metric}, f, indent=2)

    cols = list(datasets) + [g for g in groups if groups[g]]
    lines = ["| Model | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for m in args.models:
        cells = []
        for c in cols:
            e = table[m].get(c)
            cells.append("-" if e is None else f"{e['mean']:.4f} ± {e['std']:.4f}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", f"Paired tests vs `{ref}` (t-test p / Holm p / sign-flip p / "
              "#seeds with DeLong p<0.05 / ensemble DeLong p / bootstrap 95% CI of ΔAUC):", ""]
    for m, per in tests.items():
        lines.append(f"**{ref} vs {m}**")
        lines.append("")
        lines.append("| Set | ΔAUC | t p | Holm p | sign-flip p | DeLong sig. seeds | ens. DeLong p | boot CI |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for key, e in per.items():
            ed = e.get("ensemble_delong", {})
            ci = e.get("ensemble_boot_ci_diff")
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                key,
                f"{e['mean_diff']:+.4f}" if "mean_diff" in e else "-",
                f"{e['t_p']:.3g}" if "t_p" in e else "-",
                f"{e['t_p_holm']:.3g}" if "t_p_holm" in e else "-",
                f"{e['signflip_p']:.3g}" if "signflip_p" in e else "-",
                f"{e['delong_significant_seeds']}/{len(e['delong_p_per_seed'])}" if "delong_p_per_seed" in e else "-",
                f"{ed['p']:.3g}" if ed else "-",
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "-"))
        lines.append("")
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
