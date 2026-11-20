# Tiến độ dự án BiasLN: toàn bộ huấn luyện và thí nghiệm

*Cập nhật 2026-11-18 (đã đồng bộ log từ Drive). Bảng đầy đủ từng run (sinh tự động) nằm ở [`experiment_table.md`](experiment_table.md). Chạy lại `python tools/experiment_log.py --roots results_paper/results results_drive/BiasLN_runs2 --out docs/experiment_table.md` sau khi tải thêm kết quả.*

Tài liệu liên quan: [`REBUTTAL.md`](REBUTTAL.md) (kế hoạch trả lời reviewer), [`REVISION_RESULTS.md`](REVISION_RESULTS.md) (đo chi phí), [`results_summary.md`](results_summary.md) (tổng hợp cũ của hướng Real Anchoring).

---

## 0. Trạng thái hiện tại

| Hạng mục | Trạng thái |
|---|---|
| Bản revision (sạch) | Xong: `paper/revision/BiasLN_revised.pdf`, 10 trang, không còn `[pending]` |
| Bản đánh dấu cho editor | Xong: `paper/revision/BiasLN_revised_marked.pdf` |
| Thư phản biện | Xong: `paper/revision/response_to_reviewers.pdf` |
| Kiểm định thống kê | Xong: paired t-test + TOST (±0,01) trên 3 seed. Chưa có DeLong (file điểm từng video chỉ nằm trên Drive) |
| Cần tác giả xác nhận | Câu trong thư về cách code cũ chọn checkpoint (CDF-v2) |
| Còn thiếu (tùy chọn) | LN-only và LoRA mới có 1 seed (Table 6). Cần seed 1, 2, khoảng 2 giờ A100 |
| Log và kết quả | Đã tải toàn bộ `BiasLN_runs` và `BiasLN_runs2` từ Drive về `results_drive/` ngày 18/11 (không gồm checkpoint; có 1942 file điểm `.npz`). Bảng tự động có 54 cấu hình |
| Colab | Còn 21,3 CU (18/11). Không còn phiên nào chạy |

## 1. Protocol chung (áp dụng cho mọi run dưới đây)

- **Dữ liệu train:** FF++ c23 (DF, F2F, FS, NT), split chính thức 720/140/140 video. Mặc định lấy 8 frame cách đều/video, khoảng 28,8k ảnh; thí nghiệm 32 frame dùng khoảng 115k ảnh. Mỗi frame là ảnh mặt 224×224 crop theo DeepfakeBench.
- **Augmentation:** flip, xoay ±10°, Gaussian blur, một trong brightness/contrast, FancyPCA hoặc hue/sat, JPEG 40–100 (mỗi phép p=0,5).
- **Backbone:** CLIP ViT-L/14 (303,18M tham số), head tuyến tính 2 lớp trên CLS sau L₂-norm.
- **Huấn luyện:** 10 epoch, batch 32, Adam, cosine lr (min 1e-5), bf16 trên 1×A100-40GB. BiasLN dùng lr 3e-4, label smoothing 0,1, class weight fake:real = 2:1.
- **Chọn checkpoint:** frame AUC cao nhất trên **FF++ val**. Không dùng tập test nào; đã có audit xác nhận val không trùng video hay frame với test.
- **Đánh giá:** video AUC (trung bình điểm của 32 frame).
  - Avg(D) = trung bình CDF-v2, DFD, DFDC, DFDCP (protocol Effort; UADFV vẫn được chạy nhưng không tính vào trung bình).
  - Avg(M7) = trung bình 7 tập face-swap của DF40: UniFace, BlendFace, E4S, FaceDancer, FSGAN, InSwap, SimSwap.
  - M11 = 7 tập DF40 cộng 4 tập con FF++. Chỉ dùng để xem nhanh, không đưa vào paper.
- **Seed:** 1024, 1, 2.
- **Baseline tái huấn luyện:**
  - GenD: chỉ tune LN, L₂-norm, CE + 0,1·align + 0,5·uniformity, lr 2e-4, batch 16;
  - Effort: SVD residual rank-1 trên q, k, v, o; lr 2e-4, wd 5e-4;
  - LoRA r=8 trên q, v;
  - full FT: lr backbone 1e-5, lr head 3e-4;
  - linear probe.

## 2. Kết quả chính (3 seed, chọn trên FF++ val)

| Phương pháp | Tham số train | Avg(D) | Avg(M7) | FF++ |
|---|---|---|---|---|
| Linear probe | 2k | 0,774 | 0,793 | 0,829 |
| Full FT (lr backbone 1e-5) | 303M | 0,823 | 0,819 | 0,986 |
| Effort | 199k | 0,911 ± 0,002 | 0,960 ± 0,006 | 0,991 |
| GenD | 104k | **0,925 ± 0,002** | **0,966 ± 0,002** | 0,986 |
| BitFit | 274k | 0,921 ± 0,006 | 0,960 ± 0,005 | 0,994 |
| LN-only (CE, L₂) | 104k | 0,921 (n=1) | 0,959 (n=1) | 0,987 |
| LoRA r8 (L₂) | 788k | 0,901 (n=1) | 0,961 (n=1) | 0,993 |
| **BiasLN** | 326k | 0,923 ± 0,006 | 0,957 ± 0,005 | **0,995** |

**Kiểm định ghép cặp theo seed** (BiasLN trừ đối thủ; TOST với ngưỡng ±0,01):

| So với | Avg(D) | Avg(M7) |
|---|---|---|
| GenD | −0,002, p_t = 0,57, TOST p = 0,066 (chưa chứng minh được tương đương) | −0,010, **p_t = 0,049 (GenD tốt hơn)** |
| BitFit | +0,0018 (dương ở cả 3 seed), p_t = 0,09, **TOST p = 0,002 (tương đương)** | −0,003, **TOST p = 0,019 (tương đương)** |
| Effort | +0,012, p_t = 0,10 | −0,003, TOST p = 0,019 (tương đương) |

**Kết luận:** dưới protocol sạch, các phương pháp chỉnh affine theo từng kênh (BitFit, LN, BiasLN, GenD) cho kết quả trong khoảng 0,92 ± 0,006. Loss phụ của GenD cho thêm khoảng 1 điểm trên DF40 và không đáng kể trên các dataset chéo. Số của bản gốc (0,9454) cao hơn do chọn checkpoint trên CDF-v2.

## 3. Toàn bộ thí nghiệm đã chạy

Ký hiệu: D = Avg(D), M7 = Avg(M7), M11 = 7 DF40 + 4 FF. Không ghi gì nghĩa là seed 1024.

### 3.1 Tune tham số nào (vai trò tham số), nhóm `roles_val`

| Cấu hình | Tham số | D | M7 |
|---|---|---|---|
| Bias q + fc1 ("gate", phụ thuộc nội dung) | 125k | 0,920 | 0,958 |
| Bias v + out + fc2 ("shift", tịnh tiến residual) | 76k | 0,915 | 0,960 |
| LN γ only | 53k | 0,914 | 0,961 |
| LN β only | 53k | 0,915 | 0,958 |
| LN γ+β (CE) | 104k | 0,921 | 0,959 |
| BitFit | 274k | 0,915 | 0,959 |
| BiasLN | 326k | 0,917 | 0,953 |
| roles_vof, block 0–18 (v + out + fc2 + LN) | – | 0,9185 | M11 0,973 |

Không có tập con nào vượt rõ. Mọi chênh lệch đều nhỏ hơn độ dao động giữa các seed.

### 3.2 Độ sâu tune (block nào được tune)

| Block | Tham số | D | M7 |
|---|---|---|---|
| 0–5 (first6) | 86k | 0,894 | 0,939 |
| 18–23 (last6) | 86k | 0,885 | 0,945 |
| 0–11 | 166k | 0,904 | 0,946 |
| 12–23 | 166k | 0,891 | 0,939 |
| 12–18 (theo Fisher) | 99k | 0,903 | 0,945 |
| 6–17 (mid12, 2 seed) | 166k | 0,920 | 0,964 |
| **0–18 (3 seed)** | 259k | 0,923 ± 0,004 | 0,960 ± 0,005 |
| 0–23 (BiasLN đầy đủ) | 326k | 0,923 ± 0,006 | 0,957 ± 0,005 |

**Phân tích gradient/Fisher** (`tools/grad_profile.py`):
- bias của key nhận gradient bằng 0, đúng như Proposition 1;
- block 19–23 có thể bỏ mà không giảm kết quả;
- cần tune cả nửa đầu mạng: chỉ tune 6 hoặc 12 block thì kém hẳn.

### 3.3 Tối ưu hóa và cách đọc đặc trưng (block 0–18 hoặc đầy đủ, seed 1024, chưa đồng bộ về máy)

| Cấu hình | D | M11 | Ghi chú |
|---|---|---|---|
| lr γ ×0,3 (0–18) | 0,9183 | – | kém hơn 0–18 gốc (0,9243) |
| lr γ ×0,1 (0–18) | 0,9148 | – | kém hơn |
| **lr γ ×3 (đầy đủ, 3 seed)** | 0,9205 ± 0,0044 | M7 0,9635 ± 0,0030 | So với BitFit: M7 +0,0038 (dương ở cả 3 seed, p ≈ 0,10), D −0,0005. So với BiasLN: M7 +0,0068 (p ≈ 0,06). So với GenD: M7 −0,0029, D −0,0045 |
| lr γ ×10 (đầy đủ) | 0,9040 | 0,9725 | quá đà |
| CLS từ block 12 + 18 + 24 | 0,9172 | – | không giúp |

### 3.4 Dữ liệu: 32 frame/video (khoảng 115k ảnh, như GenD), seed 1024

| Cấu hình | 8 frame D / M11 | 32 frame D / M11 |
|---|---|---|
| BiasLN 0–18 | 0,9243 / 0,969 | 0,9165 / 0,972 |
| GenD | 0,9252 / 0,974 | 0,9244 / 0,973 |
| BiasLN đầy đủ | 0,9172 / 0,969 | dừng giữa chừng |

Thêm dữ liệu không giúp, cả BiasLN lẫn GenD.

### 3.5 Trung bình trọng số (không cần train thêm)

| Cấu hình | D | M |
|---|---|---|
| Epoch soup BiasLN (epoch ≥ 5, 2 seed) | 0,921 ± 0,004 | M7 0,962 |
| Epoch soup GenD | 0,924 ± 0,004 | M7 0,964 |
| Epoch soup Effort | 0,913 | M7 0,962 |
| Seed soup BiasLN (3 seed) | 0,9204 | M11 0,955 |
| Seed soup BiasLN 0–18 | 0,9216 | M11 0,956 |
| Seed soup GenD | 0,9274 | M11 0,974 |
| WiSE-FT α = 0,5 (cả ba) | 0,82–0,86 | sụp đổ: head không còn khớp đặc trưng |

### 3.6 Biến thể đã thử rồi bỏ (có thêm loss hoặc module, trái với yêu cầu "không thêm loss")

| Cấu hình | D | M7 | Ghi chú |
|---|---|---|---|
| Real Anchoring (BitFit + neo real, λ = 1, 2 seed) | 0,927 ± 0,004 | 0,961 | Tách sang bài riêng `paper/improved_ra` |
| RA λ = 0,3 / 3 | 0,923 / 0,913 | 0,968 / 0,936 | |
| RA neo fake / neo cả hai / proto / gate | 0,923 / 0,921 / 0,918 / 0,917 | 0,962 / 0,960 / 0,960 / 0,953 | |
| Proto head / balanced sampling | 0,907 / 0,911 | 0,958 / 0,954 | |
| BiasLN + projection head bất đối xứng (Asy) | 0,911 | 0,970 | |
| LoRA không L₂ | 0,909 | 0,948 | |

### 3.7 Chi phí (A100-40GB, fp32, batch 32)

| Phương pháp | Train img/s | Train mem | Infer ms (bs 1) | Ckpt |
|---|---|---|---|---|
| GenD (LN) | 47,1 | 12,16 GB | 17,1 | 0,4 MB |
| BitFit | 46,6 | 12,16 GB | 17,1 | 1,1 MB |
| BiasLN | 46,5 | 12,16 GB | 17,2 | 1,3 MB |
| Effort | 41,7 | 13,22 GB | 23,8 | 0,8 MB |
| LoRA r8 | 45,9 | 12,87 GB | 18,6 | 3,0 MB |
| Full FT | 32,8 | 19,72 GB | 17,2 | 1157 MB |

Các phương pháp PEFT tốn gần như như nhau khi train. Khác biệt nằm ở chi phí inference (Effort, LoRA chưa merge) và dung lượng checkpoint.

## 4. Bài học rút ra (dùng cho bài survey)

1. **Quy tắc chọn checkpoint làm thay đổi kết quả nhiều hơn chính phương pháp.**
   - Chọn trên CDF-v2: 0,945. Chọn trên FF++ val: 0,923.
   - Mỗi bài công bố dùng một quy tắc khác nhau: DeepfakeBench lưu checkpoint tốt nhất cho từng tập test; GenD dùng validation từ dataset khác, có trùng với tập test.
2. **FF++ val bão hòa** (AUC ≥ 0,98), nên gần như không phân biệt được checkpoint tổng quát tốt hay kém.
3. **Các phương pháp chỉnh affine theo từng kênh (bias, LN, bias + LN) tương đương về thống kê.**
   - Có cơ sở lý thuyết: phần shift của LN chính là một cập nhật bias bị ràng buộc, và bias của key không có tác dụng.
   - Loss căn chỉnh và đồng đều của GenD cho thêm khoảng 1 điểm trên DF40.
4. **Chỉnh nhẹ là đủ:**
   - full FT 0,82, linear probe 0,77, PEFT khoảng 0,92;
   - LoRA (nhiều tham số hơn) không tốt hơn.
5. **Những thứ không giúp khi chọn checkpoint trên FF++ val:**
   - thêm frame;
   - các kiểu soup;
   - WiSE-FT;
   - CLS nhiều tầng;
   - lr riêng cho γ (chỉ cải thiện nhỏ trên DF40).

## 5. Việc còn lại

- [x] Tải log từ Drive về `results_drive/`; tắt mọi phiên Colab.
- [x] DeLong trên điểm từng video: xem [`stats_delong.md`](stats_delong.md).
- [ ] (Tùy chọn) LN-only và LoRA seed 1, 2.
- [ ] Tác giả xác nhận chi tiết protocol của bản đầu trong thư phản biện.
- [ ] Bài survey: xem [`SURVEY_PLAN.md`](SURVEY_PLAN.md).

## 2026-11-20 — Seeds 6–9 (Kaggle L4×4, kernel biasln-l4-seeds2)
- LN only, BitFit, BiasLN: seeds 6–9 done → n = 10 (1024, 1–9). GenD seeds 6–9 cut by the 11.3 h budget (GenD stays n = 6).
- Merged into results_paper/results6/anchor_val; Holm-corrected tests in docs/stats_10seed.txt.
- Avg(D)/Avg(M): LN only 0.920/0.967, BitFit 0.921/0.961, BiasLN 0.923/0.961 (n=10); GenD 0.923/0.966 (n=6).
- LN only vs BiasLN on Avg(M): +0.006, 9/10 seeds, p=0.003, p_Holm=0.048 (significant). LN vs BitFit: +0.006, 8/10, p_Holm=0.26.
- Equivalences (Holm TOST): BiasLN≡BitFit (<0.001 both), LN≡GenD (D 0.047, M 0.016), all four schemes pairwise on Avg(D).
- Bias effect per dataset (BiasLN − LN, 10 seeds): DFDCP +3.2, FF++ +0.6, DFDC −1.7, SimSwap −2.1 points.

## 2026-11-20 (tối) — Bản survey revision_v2 và nhánh additive
- Manuscript `paper/revision_v2/manuscript_revised.tex`: "A Survey of Parameter-Efficient Fine-Tuning for Generalizable Deepfake Detection", 12 trang, 0 tham chiếu lỗi.
  - Hình 1 vẽ lại dạng dọc (1 cột) trong Figma, frame `Fig_PEFT_Overview_vertical`: luồng ảnh mặt → CLIP ViT-L/14 → real/fake, thẻ màu cho 4 nhánh PEFT, không có loss.
  - Thay RQ1–3 bằng 3 khía cạnh so sánh: Budget, Location, Generalization.
  - Benchmark tách thành các mục: 5 Benchmark Setup, 6 Budget, 7 Location (gồm phần giải thích bias vs LayerNorm), 8 Generalization to Unseen Generators, 9 Training Variants and Cost.
  - Bảng kiểm định ghép cặp (Holm t-test + TOST) và bảng bias-role chuyển ra Phụ lục; bỏ biểu đồ cột per-dataset (trùng với bảng).
  - Bảng/hình được đặt lại để nằm trong trang của mục tương ứng.
- Nhánh additive (AdaptFormer-style adapter, bottleneck 16, 0.81M; VPT-deep 8 prompt/khối, 0.20M): đang chạy trên Colab L4 (biasln8), seeds 1024, 1, 2, kết quả đồng bộ về Drive `BiasLN_runs3`. Chưa có số liệu.
- ForAda: bị chặn vì cần mask FF++ (xray, patch_label, if_boundary) mà JSON dữ liệu không có.
