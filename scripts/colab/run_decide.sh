#!/usr/bin/env bash
# "Is tuning bias + LN actually worth it?" -- the core ablation first, with every
# row sharing BiasLN's loss / hyper-parameters / data / FF++-val selection and
# differing ONLY in which parameters are tuned. Full test suite (cross-dataset
# + DF40 + FF++), bf16, seed-major so a 1-seed answer exists after ~3 h.
# Then finishes the remaining main-comparison seeds and the rest of run_all.
set -uo pipefail
export EXTRA_OPTS=${EXTRA_OPTS:-"workers=10"}
SEEDS=${SEEDS:-"1024 1 2"}
B=BiasLN
# Which parameters carry generalisation? Rows grouped by *role* (see tools/analyze_bias.py):
#   bias_gate  = q + fc1 biases: content-dependent (attention prior, GELU gating)
#   bias_shift = v + out + fc2 biases: content-independent residual-stream translations
#   ln_beta / ln_gamma = the two halves of LN-tuning. LN beta is exactly re-parameterisable
#   as tied q/k/v (ln1) and fc1 (ln2) biases, so LN-tuning ~ constrained BitFit + gamma.
CORE="bitfit_l2=$B:tuning_mode=bitfit \
ln_l2=$B:tuning_mode=ln \
bias_gate=$B:tuning_mode=bitfit,param_roles=q+fc1 \
bias_shift=$B:tuning_mode=bitfit,param_roles=v+out+fc2 \
ln_beta=$B:tuning_mode=ln,param_roles=ln_beta+outer_beta \
ln_gamma=$B:tuning_mode=ln,param_roles=ln_gamma+outer_gamma \
biasln=$B \
linprobe=$B:tuning_mode=frozen \
lora_l2=$B:tuning_mode=lora \
fullft=$B:tuning_mode=full,backbone_lr=0.00001"
ABL=results/ablation_val
# the 'biasln' row is identical to the main-comparison run: reuse it, don't retrain
mkdir -p $ABL
for t in biasln biasln_soup; do
  for d in results/val/$t/seed*; do
    [ -d "$d" ] && [ ! -e "$ABL/$t/$(basename $d)" ] && mkdir -p $ABL/$t && cp -r "$d" $ABL/$t/
  done
done
for s in $SEEDS; do
  ENTRIES="$CORE" SEEDS="$s" AMP=bf16 SELECTION=val OUT=$ABL LOGS=logs/ablation_val bash scripts/run_protocol.sh
  python tools/stats.py --results $ABL --models biasln bitfit_l2 ln_l2 bias_gate bias_shift ln_beta ln_gamma linprobe lora_l2 fullft \
    biasln_soup bitfit_l2_soup ln_l2_soup bias_gate_soup bias_shift_soup --reference biasln --out results/summary_ablation_core || true
done
# remaining main-comparison seeds (resumable: seed 1024 is skipped), then the rest
PHASES="main oracle paper" bash scripts/colab/run_all.sh
OUT=$ABL LOGS=logs/ablation_val AMP=bf16 SEEDS="$SEEDS" SELECTION=val bash scripts/run_ablation.sh
echo "== run_decide finished"
