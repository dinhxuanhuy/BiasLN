# Revision results

This file collects the numbers produced by the revision experiments listed in [`REBUTTAL.md`](REBUTTAL.md). Every number here was measured, not copied from the manuscript. A section stays marked *pending* until its runs finish.

## E5. Computational cost (R1.6, R2.6)

The measurements come from `bash scripts/benchmark.sh` on an NVIDIA A100-SXM4-40GB with torch 2.14.1 and synthetic 224×224 inputs. Training means forward, backward and an Adam step at batch size 32. Inference uses batch size 1 for latency and batch size 32 for throughput. "Trainable" counts the CLIP ViT-L/14 vision tower (303.18M parameters; the 123.65M-parameter text tower is not used) plus the head.

**fp32 (the paper's setting)**

| Method | Trainable | Ckpt (MB) | Train ms/step | Train img/s | Train mem (GB) | Infer ms (bs1) | Infer img/s | Infer mem (GB) | GFLOPs/img |
|---|---|---|---|---|---|---|---|---|---|
| Linear probe | 0.002M (0.001%) | 0.0 | 318 | 100.6 | 1.67 | 17.0 | 102.2 | 1.68 | 162.0 |
| BitFit | 0.274M (0.091%) | 1.1 | 687 | 46.6 | 12.16 | 17.1 | 102.1 | 1.68 | 162.0 |
| LN (GenD) | 0.104M (0.034%) | 0.4 | 680 | 47.1 | 12.16 | 17.1 | 102.2 | 1.68 | 162.0 |
| Effort | 0.199M (0.066%) | 0.8 | 768 | 41.7 | 13.22 | 23.8 | 101.4 | 1.68 | 162.2 |
| LoRA r8 | 0.788M (0.259%) | 3.0 | 698 | 45.9 | 12.87 | 18.6 | 98.9 | 1.69 | 162.4 |
| **BiasLN** | **0.326M (0.107%)** | **1.3** | 688 | 46.5 | 12.16 | **17.2** | 102.1 | 1.68 | 162.0 |
| BiasLN + Asy | 1.638M* | 6.3 | 691 | 46.3 | 12.18 | 17.1 | 102.1 | 1.69 | 162.0 |
| BiasLN + last 2 blocks | 25.491M (8.4%) | 97.3 | 712 | 45.0 | 12.62 | 17.1 | 102.1 | 1.68 | 162.0 |
| Full FT | 303.182M (100%) | 1156.7 | 975 | 32.8 | 19.72 | 17.2 | 102.1 | 1.68 | 162.0 |

\* Includes the 1.31M-parameter projection head, which is used only during training. At inference the variant has the same 0.326M trained parameters as BiasLN.

**bf16 autocast (used for the multi-seed runs)**

| Method | Train ms/step | Train img/s | Train mem (GB) | Infer ms (bs1) | Infer img/s |
|---|---|---|---|---|---|
| Linear probe | 52 | 617.7 | 1.48 | 18.6 | 685.5 |
| BitFit | 129 | 248.3 | 8.01 | 20.7 | 685.3 |
| LN (GenD) | 123 | 260.3 | 8.01 | 19.3 | 685.5 |
| Effort | 134 | 239.5 | 9.07 | 38.4 | 637.8 |
| LoRA r8 | 138 | 232.3 | 8.71 | 30.0 | 600.3 |
| **BiasLN** | 129 | 248.1 | 8.01 | 20.9 | 685.0 |
| Full FT | 172 | 186.5 | 14.26 | 21.7 | 684.8 |

**What these numbers support**

- *Inference.* BiasLN keeps the frozen backbone's architecture, so its inference cost equals plain CLIP ViT-L/14 (162 GFLOPs, same latency and memory). Effort and unmerged LoRA rebuild or add weights at every forward pass: in bf16, latency at batch 1 is 38.4 ms for Effort and 30.0 ms for LoRA, against 20.9 ms for BiasLN.
- *Training vs. full FT.* In fp32, BiasLN is 1.42× faster per step, uses 38% less peak memory (12.2 vs 19.7 GB) and stores an 890× smaller checkpoint (1.3 MB vs 1.16 GB).
- *Training vs. other PEFT.* Costs are almost the same (12.2–13.2 GB, 42–47 img/s in fp32), because activation memory for back-propagating through the frozen 24-block encoder dominates. The manuscript should **not** claim that BiasLN trains faster than GenD or LoRA. Its advantage there is zero inference overhead and a tiny storage and deployment footprint.
- The backbone itself is large: 162 GFLOPs per image and about 17 ms per image at batch 1 on an A100. Edge deployment would need a smaller backbone; list this as a limitation.

## E6. Validation/test disjointness — *pending (data setup)*

## E1. Main comparison, FF++-val selection, 5 seeds — *pending*

## E2. Last-epoch protocol — *pending*

## E3. Per-epoch curves and oracle gap — *pending*

## E4. One-factor ablation — *pending*
