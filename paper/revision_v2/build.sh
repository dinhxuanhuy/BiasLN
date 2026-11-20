#!/usr/bin/env bash
# Rebuild the revised manuscript (review + controlled study) and the response letter.
#   PY=python TECTONIC=tectonic bash build.sh
set -euo pipefail
cd "$(dirname "$0")"
RESULTS=${RESULTS:-../../results_paper/results6}
PY=${PY:-python}
TECTONIC=${TECTONIC:-tectonic}
$PY ../../tools/paper_assets.py --paper survey --results "$RESULTS" --out generated > /dev/null
$PY ../../tools/stats_table.py --results "$RESULTS" --out generated > /dev/null
$TECTONIC -X compile manuscript_revised.tex --keep-intermediates
[ -f response_to_reviewers.tex ] && $TECTONIC -X compile response_to_reviewers.tex
echo built
