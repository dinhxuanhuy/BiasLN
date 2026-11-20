#!/usr/bin/env bash
# Training / inference cost of every adaptation scheme (synthetic inputs, no data needed).
#   bash scripts/benchmark.sh            # fp32 (paper setting) and bf16
set -euo pipefail
C=training/config/detector
OUT=${OUT:-results/efficiency}
ENTRIES=(
  --entry "LinearProbe=$C/BiasLN.yaml:tuning_mode=frozen"
  --entry "BitFit=$C/BiasLN.yaml:tuning_mode=bitfit"
  --entry "LN (GenD)=$C/GenD.yaml"
  --entry "Effort=$C/Effort.yaml"
  --entry "LoRA r8=$C/BiasLN.yaml:tuning_mode=lora"
  --entry "BiasLN=$C/BiasLN.yaml"
  --entry "BiasLN+Asy=$C/BiasLN_Asy.yaml"
  --entry "BiasLN+last2=$C/BiasLN.yaml:last_k_blocks=2"
  --entry "FullFT=$C/BiasLN.yaml:tuning_mode=full,backbone_lr=1e-5"
)
for amp in ${AMPS:-none bf16}; do
  python tools/benchmark_efficiency.py "${ENTRIES[@]}" --amp "$amp" --train_batch 32 --infer_batch 32 \
    --out "${OUT}_${amp}"
done
