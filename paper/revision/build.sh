#!/usr/bin/env bash
# Rebuild the revised BiasLN manuscript and the response letter from the experiment outputs.
#   PY=python TECTONIC=tectonic bash build.sh
# Set FINAL=1 to turn every remaining \pending{} into a compile error.
# (The Real Anchoring paper lives in ../improved_ra and is built separately.)
set -euo pipefail
cd "$(dirname "$0")"
RESULTS=${RESULTS:-../../results_paper/results}
FEATS=${FEATS:-../../results_paper/features}
PY=${PY:-python}
TECTONIC=${TECTONIC:-tectonic}
$PY ../../tools/paper_assets.py --paper biasln --results "$RESULTS" --out generated
[ -d "$FEATS" ] && $PY ../../tools/feature_figs.py --feat_dir "$FEATS" --out generated \
  --models clip effort gend biasln --no_drift || true
if [ "${FINAL:-0}" = 1 ]; then sed -i 's/^\\newif\\ifpending\\pendingtrue/\\newif\\ifpending\\pendingfalse/' BiasLN_revised.tex response_to_reviewers.tex; fi
$TECTONIC -X compile BiasLN_revised.tex --keep-intermediates
# copy with the revised passages in blue (for the editor), from the same source
sed 's/^\\newif\\ifrevmark\\revmarkfalse/\\newif\\ifrevmark\\revmarktrue/' BiasLN_revised.tex > BiasLN_revised_marked.tex
$TECTONIC -X compile BiasLN_revised_marked.tex
$TECTONIC -X compile response_to_reviewers.tex
echo "built: BiasLN_revised.pdf response_to_reviewers.pdf"
