#!/usr/bin/env bash
# Table 4 ablation, isolated one factor at a time, every row with the same
# data, head, schedule and selection protocol (Reviewer 2: "the ablation does
# not fully isolate the contribution of each component").
#
# Factors: which parameters are tuned  x  L2 normalisation  x  class weights.
#   linprobe  head only (frozen CLIP)             -> how much does tuning add?
#   fullft    full fine-tuning, backbone LR 1e-5  -> fixes the under-trained baseline
#   lora      LoRA r=8 on q,v of all blocks
#   bitfit / ln / bitfit_ln, each with and without L2 norm
#   bitfit_ln_noW  BiasLN without class weights (1:1)
#   bitfit_ln_noLS BiasLN without label smoothing
#   asy       BiasLN + asymmetric SupCon (improved variant)
set -euo pipefail
B=BiasLN
export ENTRIES=${ENTRIES:-"\
linprobe=$B:tuning_mode=frozen \
fullft=$B:tuning_mode=full,backbone_lr=0.00001 \
lora=$B:tuning_mode=lora,l2_norm=false \
lora_l2=$B:tuning_mode=lora \
bitfit=$B:tuning_mode=bitfit,l2_norm=false \
bitfit_l2=$B:tuning_mode=bitfit \
ln=$B:tuning_mode=ln,l2_norm=false \
ln_l2=$B:tuning_mode=ln \
bitfit_ln=$B:tuning_mode=bitfit_ln,l2_norm=false \
biasln=$B \
biasln_noW=$B:weight_fake=1.0 \
biasln_noLS=$B:label_smoothing=0.0 \
asy=BiasLN_Asy"}
export SEEDS=${SEEDS:-"1024 1 2"}
export TEST_SETS=${TEST_SETS:-"Celeb-DF-v2 DeepFakeDetection DFDC DFDCP UADFV"}
export OUT=${OUT:-results/ablation_${SELECTION:-val}}
exec "$(dirname "$0")/run_protocol.sh"
