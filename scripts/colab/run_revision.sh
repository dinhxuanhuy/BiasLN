#!/usr/bin/env bash
# BiasLN revision first (everything the response to the reviewers needs), Real Anchoring runs after.
#  1. main table to n=3 seeds: BitFit 1,2; GenD 2; BiasLN 2; Effort 2   (R1.4, R2.2, R2.5)
#  2. full fine-tuning (backbone lr 1e-5) and linear probe              (R1.5)
#  3. old Table-4 ablation under the new protocol: LoRA +-L2, BitFit -L2, LN -L2, LN +L2 seeds 1,2  (R1.4, R2.5)
#  4. significance tests
# Then scripts/colab/run_ra_later.sh (Real Anchoring follow-up paper).
set -uo pipefail
export EXTRA_OPTS=${EXTRA_OPTS:-"workers=10"}
export SYNC=${SYNC:-/content/drive/MyDrive/BiasLN_runs2}
B="BiasLN:tuning_mode=bitfit"
OUT=results/anchor_val; LOGS=logs/anchor_val
# seed-1/1024 baselines trained on the first VM live on Drive; bring them in so they are not re-run
for r in gend effort biasln; do for s in 1 1024; do
  src=/content/drive/MyDrive/BiasLN_runs/val/$r/seed$s
  [ -f "$src/metrics.json" ] && [ ! -e "$OUT/$r/seed$s" ] && mkdir -p $OUT/$r && cp -r "$src" $OUT/$r/
done; done
run() { ENTRIES="$1" SEEDS="$2" AMP=bf16 SELECTION=val SOUP_FROM= OUT=$OUT LOGS=$LOGS bash scripts/run_protocol.sh; }
echo "== $(date '+%F %T') revision stage 1: main-table seeds"
run "bitfit=$B" "1 2"
run "biasln=BiasLN effort=Effort gend=GenD" 2
echo "== $(date '+%F %T') revision stage 2: full FT + linear probe"
run "fullft=BiasLN:tuning_mode=full,backbone_lr=0.00001 linprobe=BiasLN:tuning_mode=frozen" 1024
echo "== $(date '+%F %T') revision stage 3: Table-4 ablation"
run "lora=BiasLN:tuning_mode=lora lora_nol2=BiasLN:tuning_mode=lora,l2_norm=false \
bitfit_nol2=$B,l2_norm=false ln_nol2=BiasLN:tuning_mode=ln,l2_norm=false" 1024
run "ln_l2=BiasLN:tuning_mode=ln" "1 2"
python tools/stats.py --results $OUT --models biasln gend effort bitfit ln_l2 --reference biasln \
  --out results/summary_revision || true
echo "== run_revision finished"
bash scripts/colab/run_ra_later.sh
