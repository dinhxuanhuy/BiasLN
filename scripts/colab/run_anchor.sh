#!/usr/bin/env bash
# Pilot: real-anchored asymmetric objective on top of bias-only tuning (BitFit).
# Same data / schedule / FF++-val selection as BiasLN; full test suite; bf16; no soup.
#   ra_real    anchor tuned CLS feature to frozen CLIP on REAL images only
#   ra_all     same anchor on all images      (control: is the asymmetry what matters?)
#   ra_fake    anchor on fakes only           (control)
#   proto      learnable real prototype, fakes pushed below cos margin, balanced batches
#   bal        balanced batches only          (control for proto)
#   ra_proto   ra_real + proto + balanced
set -uo pipefail
export EXTRA_OPTS=${EXTRA_OPTS:-"workers=10"}
SEEDS=${SEEDS:-"1024"}
B="BiasLN:tuning_mode=bitfit"
RUNS=${RUNS:-"ra_real=$B,lambda_anchor=1.0,anchor_on=real \
proto=$B,aux_loss=real_proto,lambda_aux=0.1,proto_margin=0.3,use_class_balanced_sampler=true \
ra_all=$B,lambda_anchor=1.0,anchor_on=all \
bal=$B,use_class_balanced_sampler=true \
ra_proto=$B,lambda_anchor=1.0,anchor_on=real,aux_loss=real_proto,lambda_aux=0.1,proto_margin=0.3,use_class_balanced_sampler=true \
ra_fake=$B,lambda_anchor=1.0,anchor_on=fake"}
for s in $SEEDS; do
  ENTRIES="$RUNS" SEEDS="$s" AMP=bf16 SELECTION=val SOUP_FROM= OUT=results/anchor_val LOGS=logs/anchor_val \
    bash scripts/run_protocol.sh
done
echo "== run_anchor finished"
