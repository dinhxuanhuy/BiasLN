# Kế hoạch bài survey: PEFT cho phát hiện deepfake

*Bản nháp 2026-11-18. Danh sách tài liệu ở mục 4 cần kiểm tra lại (tên, tác giả, venue) trước khi trích dẫn; đây đúng là lỗi reviewer đã bắt ở bài BiasLN.*

## 1. Định vị

**Tên tạm:** *Parameter-Efficient Adaptation of Vision Foundation Models for Generalizable Deepfake Detection: A Survey and Controlled Benchmark*

Bài kết hợp hai phần: **survey** (phân loại và tổng hợp tài liệu) và **benchmark có kiểm soát**, dùng toàn bộ thí nghiệm trong [`PROGRESS.md`](PROGRESS.md). Phần benchmark là điểm khác biệt chính, vì các survey deepfake hiện có chỉ chép lại số liệu đã công bố. Kết quả của chúng ta cho thấy những số liệu đó không so sánh được với nhau, do mỗi bài chọn checkpoint một kiểu.

Ba thông điệp chính, đều đã có số liệu chứng minh:
1. **Quy tắc chọn checkpoint và split validation** làm lệch kết quả cross-dataset nhiều hơn mức chênh giữa các phương pháp: 0,945 so với 0,923 cho cùng một mô hình.
2. **Các phương pháp chỉnh affine theo từng kênh (BitFit, LN, BiasLN) tương đương về thống kê**, đúng như lý thuyết dự đoán. Phần cải thiện thêm chủ yếu đến từ loss phụ (GenD) hoặc module phụ.
3. **Chỉnh nhẹ là đủ:**
   - full FT và linear probe kém hẳn;
   - nhiều tham số hơn (LoRA) không tốt hơn;
   - tune block 0–18 tương đương tune cả 24 block.

## 2. Phân loại (taxonomy) đề xuất

| Nhóm | Cơ chế | Đại diện trong benchmark | Cần khảo sát thêm |
|---|---|---|---|
| Không chỉnh backbone | Linear probe trên đặc trưng frozen | linprobe | UnivFD (Ojha 2023) |
| Chọn tập con tham số | BitFit, LN-tuning, BiasLN, tune theo độ sâu | bitfit, ln_l2, biasln, depth_* | GenD (LN), BPL (bias + cặp real/fake) |
| Tái tham số hóa | LoRA, SVD residual | lora, effort | Effort, các biến thể LoRA |
| Thêm module | Adapter song song, prompt | – | ForAda (Forensics Adapter), prompt tuning trên CLIP |
| Mục tiêu trong không gian đặc trưng | Loss alignment/uniformity, neo real | gend, ra_* | GenD, các loss contrastive |
| Toàn bộ tham số | Full FT với lr nhỏ | fullft | – |
| Sau huấn luyện | Soup (epoch, seed), WiSE-FT | *_soup, seed soup, wise05 | Model soups, WiSE-FT |

Các trục phân tích chéo:
- số tham số;
- chi phí train và inference (đã đo ở mục 3.7 của PROGRESS);
- quy tắc chọn checkpoint;
- loại dữ liệu train (lượng frame, SBI hay blending);
- backbone (CLIP, DINO…, ngoài phạm vi benchmark của chúng ta).

## 3. Dàn ý

1. **Introduction:** vấn đề tổng quát hóa; vì sao dùng foundation model và PEFT; đóng góp gồm taxonomy, benchmark có kiểm soát và phân tích protocol.
2. **Background:** CLIP ViT, các họ PEFT, các benchmark (FF++, CDF-v2, DFD, DFDC, DFDCP, DF40, DeepfakeBench).
3. **Taxonomy và tổng quan tài liệu:** theo bảng mục 2. Với mỗi nhóm: cơ chế, số tham số, chi phí inference, kết quả công bố kèm quy tắc chọn checkpoint.
4. **Vấn đề đánh giá:**
   - chọn checkpoint trên tập test (DeepfakeBench lưu checkpoint tốt nhất cho từng tập test);
   - validation trùng nguồn với test;
   - trung bình trên tập dataset khác nhau giữa các bài (có UADFV hay không);
   - chỉ báo 1 seed.
   - Bảng "số công bố và số tái huấn luyện" cho Effort và GenD.
5. **Benchmark có kiểm soát:** cùng dữ liệu, augmentation, lịch huấn luyện, quy tắc chọn checkpoint và seed.
   - 5.1 So sánh chính: PROGRESS mục 2.
   - 5.2 Tham số nào (vai trò, độ sâu): mục 3.1–3.2, kèm phân tích gradient/Fisher.
   - 5.3 Tối ưu hóa và dữ liệu (lr của γ, 32 frame): mục 3.3–3.4.
   - 5.4 Trung bình trọng số: mục 3.5.
   - 5.5 Loss phụ (GenD, Real Anchoring): mục 3.6.
   - 5.6 Chi phí: mục 3.7.
   - 5.7 Kiểm định thống kê: paired t-test, TOST, DeLong.
6. **Phân tích lý thuyết:** hai proposition của BiasLN (LN shift là cập nhật bias bị ràng buộc; bias của key không có tác dụng), giải thích vì sao các phương pháp affine tương đương.
7. **Khuyến nghị thực hành:** quy tắc chọn checkpoint, số seed tối thiểu, các chỉ số cần báo cáo.
8. **Hướng mở:** validation set không trùng test (ví dụ các tập reenactment của DF40), dữ liệu tổng hợp (SBI), thông tin thời gian, backbone nhỏ cho thiết bị biên.
9. **Conclusion.**

## 4. Tài liệu cần thu thập và kiểm tra (chưa xác minh)

- **Đã trích dẫn trong BiasLN** (đã kiểm tra trong đợt revision): Effort (ICML 2025), GenD (WACV 2026), Forensics Adapter (CVPR 2025), BPL (ICML 2026), BitFit (ACL 2022), LN-tuning (Qi et al., arXiv 2022), LoRA (ICLR 2022), UnivFD (CVPR 2023), DeepfakeBench (NeurIPS D&B 2023), DF40 (NeurIPS D&B 2024), SBI (CVPR 2022), Kumar et al. (ICLR 2022), Namazifar et al. (ICASSP 2023), Giannou et al. (COLT 2023).
- **Từ đợt research 15/11, cần đọc và kiểm tra lại:**
  - MoLD (arXiv 2512.04969, đặc trưng tầng giữa);
  - "Generalized Design Choices" (arXiv 2511.21507);
  - FCG (arXiv 2404.05583, module thời gian);
  - Alpha Blending Hypothesis (arXiv 2605.10334; v2 đã bị rút);
  - Model soups và WiSE-FT (Wortsman et al. 2022).
- **Cần tìm thêm:**
  - prompt tuning và adapter trên CLIP cho deepfake;
  - các bài dùng DINOv2 hoặc các foundation model khác;
  - các survey deepfake gần đây, để chứng minh khoảng trống (chưa có survey nào chuyên về PEFT kèm benchmark có kiểm soát).

## 5. Thí nghiệm nên bổ sung cho survey (nếu có CU)

| Ưu tiên | Thí nghiệm | Chi phí (ước tính) |
|---|---|---|
| Cao | LN-only, LoRA seed 1, 2 (đủ 3 seed cho mọi nhóm) | khoảng 2 giờ A100 |
| Cao | Chọn checkpoint trên tập val reenactment của DF40 cho mọi phương pháp, để định lượng ảnh hưởng của quy tắc chọn | khoảng 1 giờ eval lại nếu còn checkpoint từng epoch |
| Trung bình | Lấy kết quả "oracle" (checkpoint tốt nhất theo từng tập test) từ các run đã có, để đo khoảng cách lạc quan | 0 giờ GPU nếu có điểm từng epoch |
| Trung bình | Prompt tuning (VPT) và một adapter đơn giản, để phủ nhóm "thêm module" | khoảng 2 giờ |
| Thấp | DeLong trên điểm từng video | 0 giờ GPU (file .npz trên Drive) |
