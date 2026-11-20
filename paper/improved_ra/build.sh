#!/usr/bin/env bash
# Build the Real Anchoring paper (improved method, separate from the BiasLN revision).
#   PY=python TECTONIC=tectonic bash build.sh
set -euo pipefail
cd "$(dirname "$0")"
RESULTS=${RESULTS:-../../results_paper/results}
FEATS=${FEATS:-../../results_paper/features}
PY=${PY:-python}
TECTONIC=${TECTONIC:-tectonic}
$PY ../../tools/paper_assets.py --paper ra --results "$RESULTS" --out generated
[ -d "$FEATS" ] && $PY ../../tools/feature_figs.py --feat_dir "$FEATS" --out generated || true
$TECTONIC -X compile RealAnchoring.tex
echo "built: RealAnchoring.pdf"
