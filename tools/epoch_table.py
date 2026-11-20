#!/usr/bin/env python
"""Per-epoch test AUC of one run and the cost/benefit of each selection rule.

    python tools/epoch_table.py logs/val/biasln_s1024 [--metric video_auc_dir]

Reads <run>/epoch_eval/epochXX/metrics.json (from scripts/eval_epochs.sh) and
<run>/run_info.json, then prints a table and writes <run>/epoch_table.json:

  selected   epoch chosen by the run's own protocol (e.g. FF++ val)
  last       final epoch
  oracle     epoch with the best *average test* AUC (not a valid protocol;
             upper bound showing how much test-set selection inflates results)
  oracle_cdf epoch with the best Celeb-DF-v2 AUC (what selecting on CDF buys)
"""

import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--metric", default="video_auc_dir")
    args = ap.parse_args()

    rows = {}
    for path in sorted(glob.glob(os.path.join(args.run, "epoch_eval", "epoch*", "metrics.json"))):
        ep = int(os.path.basename(os.path.dirname(path))[5:])
        with open(path) as f:
            m = json.load(f)["metrics"]
        rows[ep] = {d: v.get(args.metric, v.get("video_auc")) for d, v in m.items()}
    if not rows:
        raise SystemExit(f"no epoch_eval results under {args.run}")
    datasets = sorted({d for r in rows.values() for d in r})
    for r in rows.values():
        r["avg"] = float(np.mean([r[d] for d in datasets if d in r]))

    info = {}
    if os.path.exists(os.path.join(args.run, "run_info.json")):
        with open(os.path.join(args.run, "run_info.json")) as f:
            info = json.load(f)
    sel = (info.get("selected") or {}).get("epoch")
    picks = {"selected": sel, "last": max(rows),
             "oracle": max(rows, key=lambda e: rows[e]["avg"])}
    if "Celeb-DF-v2" in datasets:
        picks["oracle_cdf"] = max(rows, key=lambda e: rows[e]["Celeb-DF-v2"])

    cols = datasets + ["avg"]
    print(f"run: {args.run}  ({args.metric}; selection={info.get('selection')})")
    print("epoch | " + " | ".join(cols))
    for ep in sorted(rows):
        tags = [k for k, v in picks.items() if v == ep]
        print(f"{ep:5d} | " + " | ".join(f"{rows[ep].get(c, float('nan')):.4f}" for c in cols)
              + (f"   <- {', '.join(tags)}" if tags else ""))
    summary = {k: ({"epoch": v, **rows[v]} if v in rows else None) for k, v in picks.items()}
    if summary.get("selected") and summary.get("oracle"):
        print(f"oracle gap (avg): {summary['oracle']['avg'] - summary['selected']['avg']:+.4f}")
    with open(os.path.join(args.run, "epoch_table.json"), "w") as f:
        json.dump({"metric": args.metric, "per_epoch": rows, "picks": summary}, f, indent=2)


if __name__ == "__main__":
    main()
