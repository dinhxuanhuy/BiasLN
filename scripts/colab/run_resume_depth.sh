#!/usr/bin/env bash
# Resume the BiasLN depth study on a fresh VM: wait for setup, pull finished runs back from Drive
# (run_protocol.sh skips any (tag, seed) that already has metrics.json), then run what is left.
set -uo pipefail
until grep -q "== setup done" /content/setup.log 2>/dev/null; do sleep 60; done
export SYNC=${SYNC:-/content/drive/MyDrive/BiasLN_runs2}
mkdir -p results logs
rsync -a "$SYNC/anchor_val" results/ && rsync -a "$SYNC/grad_profile" results/ 2>/dev/null
echo "== $(date '+%F %T') resumed; finished runs: $(ls results/anchor_val/*/seed*/metrics.json | wc -l)"
bash scripts/colab/run_depth.sh
sed -i 's/^until grep -q "run_beat_gend finished".*$/: # (waiter disabled on resume)/' scripts/colab/run_fisher_depth.sh
bash scripts/colab/run_fisher_depth.sh
echo "== run_resume_depth finished"
