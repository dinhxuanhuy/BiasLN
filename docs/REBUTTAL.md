# BiasLN: response plan for the reviewers

Manuscript: *BiasLN: Efficient Tuning for Generalizable Deepfake Detection Across Benchmarks*, EAI Endorsed Transactions on Smart Cities. Decision: revisions required by both reviewers.

Each comment below is tied to (a) the code or experiment that answers it and (b) the change the manuscript needs. Numbers from the revision runs go in [`REVISION_RESULTS.md`](REVISION_RESULTS.md).

---

## 0. Issues found in the original code while preparing the revision

The authors should know about these before writing the response, because they affect how the old numbers were produced.

| # | Finding | Where | Status |
|---|---|---|---|
| C1 | **The legacy trainer selected a separate "best" checkpoint for every test dataset.** It evaluated the *test* split of each `--test_dataset` during training and saved `test/<dataset>/ckpt_best.pth` per dataset (`save_best` saved for every key not in the FF++ pool). If a table entry came from the checkpoint that was best on that same dataset, the entry used test-set selection. This is stronger than what Reviewer 1 suspected. | `training/trainer/trainer.py` (old `save_best`) | Fixed. Only the selection key (FF++ val, or the average of the selection sets) can save `ckpt_best.pth`. |
| C2 | The manuscript says a Celeb-DF-v2 *subset disjoint from the test set* was used for validation. The training code read `dataset_info[...][mode]` with `mode='test'`, so the validation loader read **CDF-v2's test split**. Unless a separate JSON was used, validation and test overlapped. | `training/dataset/abstract_dataset.py` | Fixed. A `data_split` key is now separate from the mode, and `tools/check_splits.py` audits val/test overlap at the frame, video and identity level. |
| C3 | The repository configs did not match the reported hyper-parameters. `BiasOnly.yaml` uses lr 1e-4, weight decay 1e-4, batch size 128 and class weights 1:1. The paper reports lr 3e-4, weight decay 0, batch size 32 and w_fake:w_real = 2:1. Also, BiasOnly tunes biases only (incl. LN β) and **not** LN γ. | `training/config/detector/BiasOnly.yaml` | `BiasLN.yaml` now holds the exact paper setting, and a unit test checks the 325,634 trainable parameters. |
| C4 | Label convention. The paper says "logits for the fake and real classes, corresponding to labels (0, 1)". The code uses **0 = real, 1 = fake**. | Sec. 3.3 | Fix the text. |
| C5 | Video-level AUC grouped frames by the *last folder name* only, so two videos with the same folder name (for example a real and a fake) were merged. This does not happen in FF++ or CDF naming, but it can happen in other layouts. | `training/metrics/utils.py` | Fixed. Frames are now grouped by their full parent path. |

Before re-submitting, check how the Table 3 BiasLN numbers were obtained (which checkpoint and which JSON). Then report the revision numbers with the protocol stated precisely.

---

## Reviewer 1

### R1.1 Novelty is incremental (BitFit + LN tuning exist; GenD already tunes LN)
**Response.** Agree that the components are known, and say so plainly. Reframe the contribution as an **empirical study**: under one controlled pipeline, *which* parameters of a frozen CLIP encoder should be adapted for generalisable deepfake detection? Show that the bias and LN-affine subspace is enough and better than LoRA, SVD-residual (Effort), LN-only (GenD) and full FT. Back this with seeds, significance tests and cost measurements rather than with "SOTA" claims.

**Evidence.** `scripts/run_ablation.sh` covers every adaptation scheme on an identical pipeline, and `scripts/run_protocol.sh` compares BiasLN against GenD and Effort over 5 seeds.

**Manuscript.** Rewrite the contribution list. Drop "outperforms SOTA" wherever the gap is not significant.

### R1.2 Small gains (0.9454 vs 0.9446 ForAda)
**Response.** Report mean ± std over 5 seeds, a 95% CI and paired tests against each baseline. Where the difference is not significant, say "comparable at a lower cost" instead of "outperforms". ForAda could not be re-run under our protocol because no aligned implementation is available here, so mark its numbers as *taken from the original paper* and do not test significance against it.

**Evidence.** `tools/stats.py` (paired t-test + sign-flip across seeds with Holm correction, per-seed DeLong, bootstrap CI of ΔAUC).

### R1.3 Checkpoint selected on CDF-v2, which is also a test set (key flaw)
**Response.** Re-run every method under two target-agnostic protocols:
1. **P-val:** select on the FF++ validation split. This is in-domain and disjoint from every test set.
2. **P-last:** no selection; take the last epoch (10).

Also run **P-oracle** (select on the test sets) only to *quantify* the inflation that test-set selection causes for each method (`scripts/eval_epochs.sh`, `tools/epoch_table.py`). This turns the reviewer's concern into a measured result. It also tests the paper's own claim that in-domain validation is uninformative, which strengthens the Sec. 4.5 discussion.

**Manuscript.** Sec. 4.1 (1): replace the CDF-subset validation with P-val. Report P-last as a robustness check. Move the "importance of cross-domain validation" discussion to the limitations section, backed by the measured oracle gap.

### R1.4 No standard deviations; Table 4 "five-run average" equals Table 3 except DFDC; Table 4 omits DFDCP
**Response.** Every number is now mean ± std over seeds from `tools/stats.py`. Table 4 is regenerated from `scripts/run_ablation.sh` and includes DFDCP, with averages over all five datasets. The authors should state honestly that the earlier Table 4 BiasLN row reused single-run values.

### R1.5 Full fine-tuning baseline under-trained (DFD 0.495, below chance)
**Response.** The earlier full-FT run used the PEFT learning rate (3e-4) on all 303M parameters, which destroys the pretrained features. The new baseline uses a separate backbone lr of 1e-5 (`tuning_mode=full,backbone_lr=1e-5`) and the head at the base lr. The ablation also includes a linear probe (frozen CLIP) as a lower bound.

### R1.6 Training time / memory figures
**Evidence.** `scripts/benchmark.sh` (A100-40GB, fp32 and bf16) measures trainable params, train ms/step, img/s, peak training memory, inference latency (bs 1), throughput (bs 32), peak inference memory, GFLOPs and checkpoint size. Each run also logs time and memory per epoch (`run_info.json`).

### R1.7 Wrong citations; DF40 venue
See [Corrected references](#corrected-references).

### R1.8 Writing: grammar, missing Fig. 3, Fig. 4 not discussed, GitHub link absent
- "outperforms than" → "outperforms"; "supports to improve" → "improves"; "aproachs" → "approaches"; "These results demonstrates" → "These results demonstrate".
- Fig. 3 (train/val loss) must be referenced *before* it is placed. Number the figures in order of first mention.
- Fig. 4 (self-attention maps) sits after the references and is never cited. Either add a paragraph (attention in the last ~12 blocks stays close to the pre-trained pattern under BiasLN but drifts under full FT) or remove the figure.
- Add the code link to the abstract footnote and the conclusion: <https://github.com/dinhxuanhuy/BiasLN>.

---

## Reviewer 2

### R2.1 Limited novelty; justify why combining BitFit and LN helps
Same as R1.1. In addition:
- **Complementarity argument.** Biases shift pre-activations inside attention and MLP blocks (additive, per-feature). LN γ/β rescale and shift the normalised residual stream (multiplicative scale), which BitFit cannot express. The ablation (BitFit vs LN vs both, all with the same head) is the direct test.
- Report the trainable budget of each scheme next to its accuracy, so that "better per parameter" is visible.

### R2.2 No statistical significance vs GenD
See R1.2. GenD is re-implemented (`training/config/detector/GenD.yaml`: LN-only, L2 norm, CE + 0.1·alignment + 0.5·uniformity) and run on the same seeds, data and selection rule, so the tests are *paired*.

### R2.3 Loss equation omits the class weights
Replace Eq. (8) with the implemented loss:

$$\mathcal{L}_{\text{ce}}=\frac{\sum_{i=1}^{N}\sum_{c=1}^{C}-\,w_c\,\tilde y_{i,c}\log p_{i,c}}{\sum_{i=1}^{N} w_{y_i}},\qquad \tilde y_{i,c}=(1-\alpha)\,\mathbb 1[c=y_i]+\tfrac{\alpha}{C},$$

with $w_{\text{real}}=1$, $w_{\text{fake}}=2$, $\alpha=0.1$, $C=2$. `tests/test_biasln.py::test_weighted_ce_matches_torch` checks that this equals the PyTorch implementation used in training.

### R2.4 Model selection on CDF-v2 / evaluation independence
See R1.3 and code findings C1–C2.

### R2.5 Ablation does not isolate components
`scripts/run_ablation.sh` changes one factor at a time from BiasLN:

| Factor | Variants |
|---|---|
| Parameters tuned | frozen (linear probe), BitFit, LN, BitFit+LN, LoRA r8, full FT (backbone lr 1e-5) |
| L2 normalisation | each PEFT variant with and without |
| Class weights | 2:1 vs 1:1 |
| Label smoothing | 0.1 vs 0 |
| Auxiliary loss | none vs asymmetric SupCon (`BiasLN_Asy.yaml`) |

### R2.6 Efficiency assessed only by parameter count; inference cost; deployment
Same evidence as R1.6. State plainly that **inference cost equals that of the frozen CLIP ViT-L/14**: BiasLN changes values, not the architecture. This is an advantage over LoRA and Effort, whose unmerged forms add work at every step (measured in the benchmark), and over ForAda, which adds a parallel adapter network. Also state the limitation that the backbone itself is large (~162 GFLOPs per image). Name smaller backbones (ViT-B/16) as future work rather than claiming suitability for edge devices.

### R2.7 Real-world conditions, full videos, emerging manipulations
Add these to the limitations: the method works on frames, the evaluation uses pre-cropped faces, temporal cues are not modelled and diffusion-based face generation is not covered. Report the DF40 results as the evidence available for newer face-swap methods.

### R2.8 References: preprints, incomplete entries
See below. Replace preprints with their peer-reviewed versions where they exist (GenD → WACV 2026; Giannou et al. → COLT 2023). Keep arXiv-only entries (Wang et al. LN-tuning, DFDC, DFDCP, ValizadehAslani & Liang) and label them as preprints.

### R2.9 Writing quality, repetition, over-claiming
Remove claims of superiority that the significance tests do not support. Merge the repeated efficiency statements from the introduction, results and conclusion into one quantified paragraph that points to the efficiency table.

---

## Corrected references

Entries flagged by Reviewer 1, plus other errors found while checking (verified November 2026).

| Key | Current (wrong) | Correct |
|---|---|---|
| `sun2024lsda` | "LSDA: Large scale deepfake detection augmentation", Sun et al. | **Yan Z, Luo Y, Lyu S, Liu Q, Wu B.** Transcending forgery specificity with latent space augmentation for generalizable deepfake detection. *CVPR* 2024. (Already in the bib as `yan2024latent`; cite that key and delete `sun2024lsda`.) |
| `yang2024prodet` | "ProDet: Progressive detection...", Yang et al., AAAI 2024 | **Cheng J, Yan Z, Zhang Y, Luo Y, Wang Z, Li C.** Can we leave deepfake data behind in training deepfake detector? *NeurIPS* 2024. |
| `lin2024cdfa` | "CDFA: Cross-domain feature alignment...", Lin, Liu, Ouyang, CVPR 2024 | **Lin Y, Song W, Li B, Li Y, Ni J, Chen H, Li Q.** Fake it till you make it: Curricular dynamic forgery augmentations towards general deepfake detection. *ECCV* 2024. doi:10.1007/978-3-031-73016-0_7 |
| `cui2025forada` | "ForAda: Forgery-aware adaptive transformer...", Cui, Zhang, Yao, ... | **Cui X, Li Y, Luo A, Zhou J, Dong J.** Forensics Adapter: Adapting CLIP for generalizable face forgery detection. *CVPR* 2025, pp. 19207–19217. |
| `huang2023iid` | "IID: Instance-inconsistency detection...", Huang, Bhunia, Bhatt | **Huang B, Wang Z, Yang J, Ai J, Zou Q, Wang Q, Ye D.** Implicit identity driven deepfake face swapping detection. *CVPR* 2023. |
| `yan2024df40` | CVPRW 2024, DeepfakeBench author list | **Yan Z, Yao T, Chen S, Zhao Y, Fu X, Zhu J, Luo D, Wang C, Ding S, Wu Y, Yuan L.** DF40: Toward next-generation deepfake detection. *NeurIPS* 2024 (Datasets and Benchmarks Track). |
| `gu2022slad` | Audio-visual speech representation paper (wrong paper) | **Chen L, Zhang Y, Song Y, Liu L, Wang J.** Self-supervised learning of adversarial example: Towards good generalizations for deepfake detection. *CVPR* 2022. (This is SLADD.) |
| `luo2021srm` | Luo H, Zhang B, Liu S, ... | **Luo Y, Zhang Y, Yan J, Liu W.** Generalizing face forgery detection with high-frequency features. *CVPR* 2021. |
| `yermakov2025gend` | arXiv preprint | **Yermakov A, Cech J, Matas J, Fritz M.** Deepfake detection that generalizes across benchmarks. *WACV* 2026, pp. 773–783. |
| `wang2022ln` | "Wang Q, Ruan YP, Zuo Y, Li T" | Title and arXiv id are correct (arXiv:2211.08682, 2022). arXiv lists the first author as "Wang Qi"; check the name order against the PDF before deciding between "Wang Q" and "Qi W". Mark the entry as a preprint. |
| `giannou2023layernorm` | arXiv preprint | **Giannou A, Rajput S, Papailiopoulos D.** The expressive power of tuning only the normalization layers. *COLT* 2023. |
| `yang2022diffusion` | ACM Comput Surv 2024;62(1) | **ACM Comput Surv. 2023;56(4):1–39.** doi:10.1145/3626235 |

Also check: `bray2023testing` and `hashmi2024unmasking` are in the bibliography but never cited in the text, so remove them or cite them. `sagonas2016`, `dosovitskiy2021vit`, `hu2022lora`, `wang2020alignment`, `valizadeh2024layernorm` and `yan2023ucf` need to be checked for a citation in the text too.

Sources: [DF40 (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/34239f60eca7ce9bee5280aaf81362d8-Abstract.html), [CDFA (ECCV 2024)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/11581_ECCV_2024_paper.php), [ProDet (NeurIPS 2024)](https://papers.neurips.cc/paper_files/paper/2024/file/2718a032d15e0b80cd164b240220df89-Paper-Conference.pdf), [Forensics Adapter (CVPR 2025)](https://openaccess.thecvf.com/content/CVPR2025/html/Cui_Forensics_Adapter_Adapting_CLIP_for_Generalizable_Face_Forgery_Detection_CVPR_2025_paper.html), [GenD (WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/html/Yermakov_Deepfake_Detection_that_Generalizes_Across_Benchmarks_WACV_2026_paper.html), [LN-tuning (arXiv 2211.08682)](https://arxiv.org/abs/2211.08682).

---

## Experiment checklist

| ID | Command | Answers |
|---|---|---|
| E1 | `scripts/run_protocol.sh` (SELECTION=val, 5 seeds; BiasLN, GenD, Effort) | R1.2, R1.3, R1.4, R2.2, R2.4 |
| E2 | `SELECTION=last scripts/run_protocol.sh` | R1.3 robustness |
| E3 | `scripts/eval_epochs.sh logs/val/*` → oracle gap | R1.3, Sec. 4.5 claim |
| E4 | `scripts/run_ablation.sh` (3 seeds) | R1.4, R1.5, R2.1, R2.5 |
| E5 | `bash scripts/benchmark.sh` | R1.6, R2.6 |
| E6 | `tools/check_splits.py` (run by `setup_vm.sh`) | R1.3, R2.4 |
| E7 | `python tools/stats.py --results results/val --models biasln gend effort --reference biasln` | R1.2, R2.2 |
