#!/usr/bin/env bash
# Full revision study on one GPU VM, ordered so that a cross-model comparison
# is available as early as possible (seed-major order). Resumable.
#
#   scripts/colab/launch.sh start scripts/colab/run_all.sh
#   PHASES="main paper" SEEDS="1024 1 2 3 4" scripts/colab/run_all.sh
set -uo pipefail
PHASES=${PHASES:-"main oracle paper ablation"}
SEEDS=${SEEDS:-"1024 1 2"}
MAIN=${MAIN:-"biasln=BiasLN gend=GenD effort=Effort biasln_asy=BiasLN_Asy"}
export SYNC=${SYNC:-}
export EXTRA_OPTS=${EXTRA_OPTS:-"workers=10"}

for phase in $PHASES; do
  case $phase in
    main)    # E1: FF++-val selection (+ soup of epochs 5-9), bf16
      for s in $SEEDS; do
        ENTRIES="$MAIN" SEEDS="$s" AMP=bf16 SELECTION=val bash scripts/run_protocol.sh
      done
      python tools/stats.py --results results/val --models biasln gend effort biasln_asy \
        biasln_soup gend_soup effort_soup biasln_asy_soup --reference biasln --out results/summary_val || true
      ;;
    oracle)  # E3: per-epoch Celeb-DF-v2 AUC -> what selecting on CDF test would have given
      runs=$(ls -d logs/val/*_s* 2>/dev/null)
      TEST_SETS="Celeb-DF-v2" AMP=bf16 bash scripts/eval_epochs.sh $runs
      ;;
    paper)   # faithful fp32 reproduction of the paper configuration, one seed
      ENTRIES="biasln_fp32=BiasLN" SEEDS="1024" AMP=none SELECTION=val bash scripts/run_protocol.sh
      ;;
    ablation) # E4
      AMP=bf16 SEEDS="$SEEDS" bash scripts/run_ablation.sh
      ;;
  esac
done
echo "== run_all finished"
