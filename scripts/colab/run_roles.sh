#!/usr/bin/env bash
# Role-wise tuning experiment: which parameter roles carry generalisation?
# Same loss / hyper-parameters / data / FF++-val selection as BiasLN; only the
# set of tuned parameters differs. Full test suite, bf16, no soup (speed).
set -uo pipefail
export EXTRA_OPTS=${EXTRA_OPTS:-"workers=10"}
SEEDS=${SEEDS:-"1024 1 2"}
B=BiasLN
ROLES=${ROLES:-"bias_gate=$B:tuning_mode=bitfit,param_roles=q+fc1 \
bias_shift=$B:tuning_mode=bitfit,param_roles=v+out+fc2 \
bitfit_l2=$B:tuning_mode=bitfit \
ln_l2=$B:tuning_mode=ln \
ln_beta=$B:tuning_mode=ln,param_roles=ln_beta+outer_beta \
ln_gamma=$B:tuning_mode=ln,param_roles=ln_gamma+outer_gamma"}
OUT=results/roles_val
mkdir -p $OUT
for t in biasln gend; do   # references already trained under the same protocol
  for d in results/val/$t/seed*; do
    [ -d "$d" ] && [ ! -e "$OUT/$t/$(basename $d)" ] && mkdir -p $OUT/$t && cp -r "$d" $OUT/$t/
  done
done
for s in $SEEDS; do
  ENTRIES="$ROLES" SEEDS="$s" AMP=bf16 SELECTION=val SOUP_FROM= OUT=$OUT LOGS=logs/roles_val \
    bash scripts/run_protocol.sh
  python tools/stats.py --results $OUT --models biasln gend bias_gate bias_shift bitfit_l2 ln_l2 ln_beta ln_gamma \
    --reference biasln --out results/summary_roles || true
done
echo "== run_roles finished"
