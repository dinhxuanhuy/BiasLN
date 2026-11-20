# Tổng hợp kết quả thí nghiệm — Real Anchoring (bản sửa của BiasLN)

*Sinh tự động bởi `tools/results_summary.py` lúc 2026-11-14 15:59. Chạy lại script để cập nhật khi có seed mới.*

## Protocol chung

- Train trên FF++ c23 (4 manipulation), 10 epoch, batch 32, bf16, 1×A100; 8 frame/video khi train, 32 frame khi test.
- **Chọn checkpoint trên FF++ val** (frame AUC), không dùng dữ liệu tập test nào.
- Chỉ số: **video AUC** (trung bình xác suất fake của các frame trong video).
- Avg(D) = trung bình 4 tập cross-dataset (CDF-v2, DFD, DFDC, DFDCP, theo protocol Effort). Avg(M) = trung bình 7 phương pháp face-swap của DF40.
- `±` là độ lệch chuẩn mẫu giữa các seed; không có `±` nghĩa là mới có 1 seed.

## 1. So sánh chính (cùng protocol)

### Cross-dataset

| Cấu hình | Seeds | CDF-v2 | DFD | DFDC | DFDCP | **Avg(D)** |
|---|---|---|---|---|---|---|
| Effort (tái huấn luyện) | 1, 1024 | 0.9459 | 0.9788 | 0.8444 | 0.8803 | 0.9124 ± 0.0008 |
| GenD (tái huấn luyện) | 1, 1024 | 0.9508 | 0.9799 | 0.8621 | 0.9004 | 0.9233 ± 0.0027 |
| BiasLN (bias + LN, bản v1) | 1, 1024 | 0.9507 | 0.9742 | 0.8528 | 0.9159 | 0.9234 ± 0.0087 |
| Bias only (BitFit) | 1024 | 0.9489 | 0.9731 | 0.8344 | 0.9033 | 0.9149 |
| **RA (bias only + neo real, λ=1) — đề xuất** | 1024 | 0.9536 | 0.9842 | 0.8832 | 0.8958 | 0.9292 |

### Cross-method (DF40)

| Cấu hình | Seeds | UniFace | BlendFace | E4S | FaceDan. | FSGAN | InSwap | SimSwap | **Avg(M)** |
|---|---|---|---|---|---|---|---|---|---|
| Effort (tái huấn luyện) | 1, 1024 | 0.9713 | 0.9163 | 0.9882 | 0.9760 | 0.9794 | 0.9530 | 0.9376 | 0.9602 ± 0.0081 |
| GenD (tái huấn luyện) | 1, 1024 | 0.9799 | 0.9129 | 0.9897 | 0.9595 | 0.9798 | 0.9663 | 0.9619 | 0.9643 ± 0.0028 |
| BiasLN (bias + LN, bản v1) | 1, 1024 | 0.9603 | 0.9239 | 0.9839 | 0.9739 | 0.9816 | 0.9505 | 0.9323 | 0.9581 ± 0.0067 |
| Bias only (BitFit) | 1024 | 0.9672 | 0.9245 | 0.9776 | 0.9683 | 0.9767 | 0.9581 | 0.9399 | 0.9589 |
| **RA (bias only + neo real, λ=1) — đề xuất** | 1024 | 0.9732 | 0.8973 | 0.9907 | 0.9723 | 0.9830 | 0.9634 | 0.9533 | 0.9619 |

### In-domain (FF++ test)

| Cấu hình | Seeds | DF | F2F | FS | NT | Mean | FF++ all |
|---|---|---|---|---|---|---|---|
| Effort (tái huấn luyện) | 1, 1024 | 0.9997 | 0.9913 | 0.9931 | 0.9780 | 0.9905 | 0.9905 |
| GenD (tái huấn luyện) | 1, 1024 | 0.9996 | 0.9892 | 0.9922 | 0.9627 | 0.9859 | 0.9859 |
| BiasLN (bias + LN, bản v1) | 1, 1024 | 0.9999 | 0.9962 | 0.9964 | 0.9876 | 0.9950 | 0.9950 |
| Bias only (BitFit) | 1024 | 0.9998 | 0.9961 | 0.9980 | 0.9871 | 0.9953 | 0.9953 |
| **RA (bias only + neo real, λ=1) — đề xuất** | 1024 | 0.9990 | 0.9949 | 0.9915 | 0.9793 | 0.9912 | 0.9912 |

## 2. Ablation Real Anchoring (trên bias only)

| Cấu hình | Seeds | CDF-v2 | DFD | DFDC | DFDCP | **Avg(D)** | **Avg(M)** |
|---|---|---|---|---|---|---|---|
| Bias only, không RA | 1024 | 0.9489 | 0.9731 | 0.8344 | 0.9033 | 0.9149 | 0.9589 |
| + neo real (RA, λ=1) | 1024 | 0.9536 | 0.9842 | 0.8832 | 0.8958 | 0.9292 | 0.9619 |
| + neo fake | 1024 | 0.9493 | 0.9773 | 0.8767 | 0.8884 | 0.9229 | 0.9619 |
| + neo tất cả ảnh | 1024 | 0.9428 | 0.9775 | 0.8705 | 0.8930 | 0.9210 | 0.9606 |
| + RA λ=0.3 | 1024 | 0.9532 | 0.9825 | 0.8643 | 0.8900 | 0.9225 | 0.9682 |
| + RA λ=3 | 1024 | 0.9238 | 0.9776 | 0.8790 | 0.8709 | 0.9128 | 0.9364 |
| + RA chỉ trên b_q, b_fc1 | 1024 | 0.9316 | 0.9796 | 0.8789 | 0.8774 | 0.9168 | 0.9530 |
| + real prototype loss (cân bằng batch) | 1024 | 0.9404 | 0.9748 | 0.8389 | 0.8750 | 0.9073 | 0.9584 |
| + RA + prototype | 1024 | 0.9500 | 0.9856 | 0.8664 | 0.8688 | 0.9177 | 0.9601 |
| + chỉ cân bằng batch | 1024 | 0.9473 | 0.9815 | 0.8292 | 0.8863 | 0.9111 | 0.9536 |

## 3. Tune phần tham số nào (không có RA)

| Cấu hình | Seeds | **Avg(D)** | **Avg(M)** | FF++ |
|---|---|---|---|---|
| Tất cả bias kể cả LN β (BitFit), 272,384 | 1024 | 0.9149 | 0.9589 | 0.9953 |
| b_q, b_fc1 (phụ thuộc nội dung), 122,880 | 1024 | 0.9202 | 0.9583 | 0.9885 |
| b_v, b_o, b_fc2 (tịnh tiến), 73,728 | 1024 | 0.9147 | 0.9600 | 0.9931 |
| LN γ, β, 51,200 | 1024 | 0.9210 | 0.9594 | 0.9871 |
| Chỉ LN β, 25,600 | 1024 | 0.9145 | 0.9584 | 0.9852 |
| Chỉ LN γ, 25,600 | 1024 | 0.9142 | 0.9605 | 0.9840 |
| BitFit + LN γ (BiasLN), 323,584 | 1, 1024 | 0.9234 ± 0.0087 | 0.9581 ± 0.0067 | 0.9950 ± 0.0007 |

## 4. Độ lệch feature so với CLIP gốc (CDF-v2 test, seed 1024)

Median cos(f_tuned, f_CLIP) theo frame; gần 1 = giữ nguyên biểu diễn pre-trained. Fisher = độ tách real/fake của feature mức video (cao = tách tốt hơn trong không gian feature).

| Model | cos real | cos fake | Fisher (video) |
|---|---|---|---|
| CLIP đóng băng | – | – | 0.017 |
| Effort | – | – | 0.065 |
| GenD | 0.468 | 0.372 | 0.038 |
| BiasLN | 0.085 | 0.155 | 0.400 |
| Bias only | 0.099 | 0.192 | 0.410 |
| RA | 0.982 | 0.927 | 0.058 |

## 5. Thí nghiệm sớm (trước khi chọn RA)

### Asymmetric loss và model soup

| Cấu hình | Seeds | **Avg(D)** | **Avg(M)** | FF++ |
|---|---|---|---|---|
| BiasLN | 1, 1024 | 0.9234 ± 0.0087 | 0.9581 ± 0.0067 | 0.9950 |
| BiasLN + asymmetric loss | 1024 | 0.9114 | 0.9695 | 0.9939 |
| GenD | 1, 1024 | 0.9233 ± 0.0027 | 0.9643 ± 0.0028 | 0.9859 |
| Effort | 1, 1024 | 0.9124 ± 0.0008 | 0.9602 ± 0.0081 | 0.9905 |

Model soup (trung bình trọng số các checkpoint của cùng một run):

| Cấu hình | Seeds | **Avg(D)** | **Avg(M)** | FF++ |
|---|---|---|---|---|
| biasln_asy soup | 1024 | 0.9092 | 0.9663 | 0.9927 |
| biasln soup | 1, 1024 | 0.9214 ± 0.0038 | 0.9620 ± 0.0041 | 0.9948 |
| effort soup | 1, 1024 | 0.9129 ± 0.0001 | 0.9622 ± 0.0035 | 0.9902 |
| gend soup | 1, 1024 | 0.9235 ± 0.0040 | 0.9636 ± 0.0025 | 0.9856 |

### Số liệu bản v1 (để đối chiếu)

Bản nộp đầu chọn checkpoint trên một phần CDF-v2 (cũng là tập test), 1 lần chạy:

| | CDF-v2 | DFD | DFDC | DFDCP | Avg(D) 4 tập |
|---|---|---|---|---|---|
| BiasLN v1 | 0.971 | 0.977 | 0.862 | 0.925 | 0.9338 |
| BiasLN, protocol mới | 0.9507 | 0.9742 | 0.8528 | 0.9159 | 0.9234 ± 0.0087 |

