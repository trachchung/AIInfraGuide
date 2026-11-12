---
title: "Nhật ký cập nhật bài AIInfraGuide"
description: "Ghi lại mọi lần cập nhật nội dung của kho tri thức AIInfraGuide, để độc giả theo dõi thay đổi mới nhất"
tags: ["Thông báo", "Nhật ký cập nhật"]
sourceHash: "01537d99535d6a9a"
---

Trang này liên tục ghi lại các cập nhật nội dung kho tri thức AIInfraGuide, sắp xếp mới nhất trước, để mọi người nắm động thái mới.

---

## 2026-11-07

### Nội dung mới

**Module 4 · Tối ưu suy luận** Chương 6 «Suy luận phân tán và triển khai mô hình lớn» lên sóng. Từ bốn cách cắt song song trong suy luận đến triển khai vLLM một máy, đa nút và thí nghiệm đối chứng. Cấu hình dựa trên vLLM v0.32.0, thêm 10 hình minh họa kiểu vẽ tay:

- **Bài:** [6.1 Tổng quan chiến lược song song suy luận: chứa không nổi và không đủ nhanh là hai chuyện khác nhau](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/61-推理并行策略总览/) | Dùng sổ bộ nhớ để tách "chứa không nổi" và "không đủ nhanh", so TP, PP, DP, EP cắt gì và đồng bộ gì, đưa thứ tự chọn độ song song và script ước lượng chạy được

- **Bài:** [6.2 Suy luận song song tensor: một lớp cắt thành vài phần, mỗi lớp đồng bộ hai lần](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/62-张量并行推理/) | Suy ra lượng dữ liệu và độ trễ hai lần AllReduce mỗi lớp, phân tích sao chép KV head dưới GQA, khoảng cách băng thông NVLink với xuyên nút, ảnh hưởng của độ TP lên độ trễ và throughput, cùng Custom AllReduce của vLLM

- **Bài:** [6.3 Suy luận song song pipeline: cắt theo lớp, dùng nhiều Batch lấp đầy pipeline](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/63-流水线并行推理/) | Làm rõ nội dung và lượng giao tiếp giữa các stage, vì sao Decode cần nhiều Batch bay cùng lúc, xử lý chia lớp không đều, và dùng mô hình chi phí so tổ hợp TP×PP

- **Bài:** [6.4 Song song dữ liệu và song song chuyên gia: mở thêm bản sao, và để Token đi tìm chuyên gia](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/64-数据并行与专家并行/) | Từ nhiều bản sao mô hình Dense đến attention DP + chuyên gia EP của MoE, phủ hai lần All-to-All, đồng bộ chạy rỗng, mô phỏng tải không đều và EPLB, DeepEP cùng phương án triển khai DeepSeek-V3

- **Bài:** [6.5 Triển khai đa nút: dùng Ray hoặc đa tiến trình trải một dịch vụ lên nhiều máy](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/65-多节点部署/) | Dựng cụm Ray và bố trí rank, cách viết `--nnodes` đa tiến trình, khởi động DP + EP đa nút của MoE, cấu hình InfiniBand và GPUDirect RDMA, cùng script tự kiểm giao tiếp và bảng xử lý sự cố

- **Bài:** [6.6 Thực chiến triển khai phân tán vLLM: cùng 8 card, đo ba cách cắt cùng lúc](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/66-vllm分布式部署实战/) | Trên một máy 8 card so Llama-3.1-70B TP8, TP4×PP2, TP4×DP2: trước khởi động ước lượng dung lượng KV, sau khởi động đối chiếu log và chỉ số, dùng `vllm bench serve` quét đồng thời; đồng thời giới thiệu dùng DCP loại bỏ sao chép KV ngữ cảnh dài

### Điều chỉnh nội dung

- **Bài:** [Chương 6 Suy luận phân tán và triển khai mô hình lớn](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/第6章-分布式推理/) | Lời dẫn chương viết lại theo nội dung thực tế, bổ sung ghi chú thời hạn thông tin và liên kết từng mục

---

## 2026-11-01

### Điều chỉnh nội dung

**Module 4 · Tối ưu suy luận** Chương 3 «Đọc mã nguồn nano-vllm» — 6 bài sửa: đi mã nguồn khớp commit nhánh main nano-vllm ngày 2026-04-26, thêm 3 hình minh họa, 8 hình Mermaid cũ toàn đổi sang SVG kiểu vẽ tay, và sửa một số diễn đạt chưa chính xác:

- **Bài:** [3.1 Toàn cảnh dự án và kiến trúc bốn lớp](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/31-项目全景与四层架构/) | Bảng thông tin dự án cập nhật số dòng mã, phiên bản và phụ thuộc, số Star; hình kiến trúc bốn lớp và vòng đời yêu cầu đổi sang SVG; bổ sung ghi chú Prefill và Decode không trộn batch

- **Bài:** [3.2 Cấu hình, điểm vào và toàn bộ một lần generate](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/32-配置入口与推理主循环/) | Bổ sung ràng buộc của `tensor_parallel_size`, `ignore_eos` và tham số tương tự; ghi chú nano-vllm không hỗ trợ greedy sampling, và ví dụ README không gắn chat template; bổ sung `step` trả `num_tokens` dùng dấu dương/âm phân giai đoạn

- **Bài:** [3.3 Khung mô hình và toán tử lõi](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/33-模型与核心算子/) | Hình cắt song song tensor đổi sang SVG; thêm mục Sampler; bổ sung nguồn gốc 2 lần all_reduce mỗi lớp, và số KV head hạn chế độ TP

- **Bài:** [3.4 KV Cache và BlockManager](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/34-kvcache与blockmanager/) | Mục prefix cache tách thành chuỗi hash, đăng ký, khớp; thêm hình phân bổ khi trúng prefix, flowchart khớp đổi sang SVG; bổ sung khối cuối không tra bảng, khác biệt giữa khối đang dùng khi trúng và khối trống, cùng cắt tái dùng do độ hạt cả khối

- **Bài:** [3.5 Scheduler và continuous batching](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/35-scheduler与连续批处理/) | Hình trạng thái chuỗi và vòng lịch đổi sang SVG; giải thích lần lượt ba chỗ `break` của nhánh prefill, head-of-line blocking do FCFS chặt; bổ sung preempt đi tính lại, và điều kiện chạy của `while ... else`

- **Bài:** [3.6 ModelRunner, CUDA Graph và chuẩn hiệu năng](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/36-modelrunner引擎循环与性能/) | Thêm hình tính `slot_mapping` và hình bậc CUDA Graph; flowchart `run` và hình chuỗi gọi cả chương đổi sang SVG; bổ sung layout 6 chiều của KV Cache, quan hệ trần bậc với `max_num_seqs`; chuẩn hiệu năng bổ sung điều kiện kiểm và phạm vi áp dụng kết luận

**Module 4 · Tối ưu suy luận** Chương 4 «Lượng hóa» — 6 bài sửa: thêm 7 hình minh họa, 4 hình phẳng cũ vẽ lại kiểu vẽ tay, hình Mermaid cũ toàn đổi SVG, và đối chiếu mã nguồn vLLM v0.32.0 sửa một số diễn đạt:

- **Bài:** [4.1 Cơ bản lượng hóa](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/41-量化基础/) | Thêm hình bốn độ hạt lượng hóa, vẽ lại trục số đối xứng/không đối xứng; bổ sung phạm vi mã hóa lượng hóa đối xứng 4-bit; tài liệu tham khảo lượng hóa PyTorch đổi sang tài liệu torchao

- **Bài:** [4.2 W8A8 và SmoothQuant](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/42-w8a8与smoothquant/) | Vẽ lại hình scale cặp, flowchart lượng hóa offline đổi sang SVG

- **Bài:** [4.3 Weight-only INT4](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/43-weight-only-int4/) | Thêm hình đóng gói W4A16 và dequant trên chip, hình so GPTQ/AWQ; hình pipeline Marlin đổi sang SVG

- **Bài:** [4.4 Lượng hóa KV Cache](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/44-kv-cache量化/) | Vẽ lại hình nhóm KIVI; bổ sung vùng dư Value lượng hóa từng Token theo cửa sổ trượt rồi đẩy ra

- **Bài:** [4.5 FP8 và NVFP4/MXFP4](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/45-fp8与nvfp4-mxfp4/) | Thêm hình so phân bố thang FP8 với INT8, vẽ lại hình scale vi khối FP4; đường FP8 online bổ sung `fp8_per_channel`

- **Bài:** [4.6 Chọn lượng hóa và thực chiến vLLM](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/46-量化选型与vllm实战/) | Flowchart chọn đổi sang SVG; sửa cách nạp FP8 online (v0.32.0 trước dựng trên thiết bị meta rồi chuyển từng lớp, không nạp trọn trọng số 16-bit một lần)

**Module 4 · Tối ưu suy luận** Chương 5 «Speculative Decoding» sửa: thêm 1 hình minh họa, 4 hình Mermaid cũ toàn đổi SVG kiểu vẽ tay, và đối chiếu mã nguồn vLLM v0.32.0 sửa một số diễn đạt:

- **Bài:** [5.1 Nguyên lý lõi speculative decoding](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-投机解码核心原理/) | Flowchart một vòng Draft + Verify đổi sang SVG, theo γ = 4 đánh dấu vị trí chấp nhận, sửa và hủy; thêm hình residual sampling giữ phân bố, dùng số mục 7 minh họa hai đường cộng lại bằng p

- **Bài:** [5.2 Mô hình Draft và phương án không mô hình](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/52-draft模型与无模型方案/) | Flowchart tra N-gram đổi sang SVG, đánh dấu quy tắc vLLM lấy khớp dài nhất, sớm nhất; sửa phạm vi áp dụng `ngram_gpu` (V1 và V2 Model Runner đều dùng được); bổ sung mặc định khi Suffix không đặt trần ứng viên

- **Bài:** [5.3 Medusa, EAGLE và MTP](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/53-medusa与eagle/) | Hình Draft Tree đổi sang SVG, cùng cây với thí nghiệm mục 8, vẽ Mask dạng cây và Position ID đánh số theo độ sâu; bổ sung Block acceptance chỉ triển khai trong V2 Model Runner; sửa điều kiện MTP tránh lệch phiên bản

- **Bài:** [5.4 Biên lợi ích và giới hạn](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/54-收益边界与限制/) | Hình vòng lịch đổi sang SVG; bổ sung ví dụ các yêu cầu trong cùng Batch có độ dài bước khác nhau; sửa phạm vi hỗ trợ Adaptive Verification (không chỉ DSpark), và mô tả lợi ích N-gram/Suffix dưới QPS cao

- **Bài:** [5.5 Thực chiến speculative decoding với vLLM](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/55-vllm投机解码实战/) | Chỉ số chấp nhận từng vị trí đổi sang tên `..._per_pos_total` thực sự expose và giải thích ý nghĩa đếm; bổ sung dạng ngắn `-sc` của `--speculative-config`

---

## 2026-10-29

### Điều chỉnh nội dung

**Module 4 · Tối ưu suy luận** Chương 2 «Kỹ thuật lõi engine suy luận» sửa toàn diện: 5 bài thêm 9 hình minh họa, giá trị mặc định tham số và đi mã nguồn khớp vLLM v0.32.0, sửa một số diễn đạt lỗi thời hoặc chưa chính xác:

- **Bài:** [2.1 PagedAttention: quản KV Cache như OS quản bộ nhớ](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention/) | Thêm hình ánh xạ Block Table và hình so phân mảnh; mặc định `gpu_memory_utilization` cập nhật thành 0.92; bổ sung V1 preempt chỉ đi tính lại, không còn dùng Copy-on-Write; sửa logic vào hàng đợi trống của `free_blocks`

- **Bài:** [2.2 Continuous Batching: batch động để GPU không còn chờ không](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching/) | Thêm hình so timeline Static với Continuous Batching; dùng ví dụ slot chạy rỗng và số liệu bài Orca thay số tận dụng không nguồn; pseudocode preempt đổi thành vòng retry trong mã nguồn

- **Bài:** [2.3 Prefix Cache và RadixAttention: tái dùng KV của prefix chung](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/23-prefix-cache-与-radixattention/) | Thêm hình hash khối dạng chuỗi và hình ví dụ Radix Tree; bổ sung `cache_salt`, seed hash và đếm tham chiếu RadixAttention, LRU loại lá trước, cùng lịch theo trúng cache

- **Bài:** [2.4 Chunked Prefill và lịch Token Budget thống nhất](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/24-chunked-prefill-与统一调度/) | Thêm hình timeline chia khối và hình đóng gói ngân sách Token; giải thích độ dài khối của vLLM do ngân sách còn lại quyết định; bổ sung `long_prefill_token_threshold` và gợi ý tinh chỉnh `max_num_batched_tokens` chính thức

- **Bài:** [2.5 Backend Attention và tối ưu đồ thị](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/25-attention-后端与图优化/) | Thêm hình so khởi động Kernel với CUDA Graph, hình Piecewise CUDA Graph; bổ sung backend mặc định theo kiến trúc GPU; sửa quy tắc Padding của CUDA Graph và chế độ mặc định V1 `FULL_AND_PIECEWISE`

---

## 2026-10-26

### Điều chỉnh nội dung

**Module 4 · Tối ưu suy luận** Chương 4 «Lượng hóa» — 6 bài sửa, nội dung liên quan framework khớp vLLM v0.32.0, mỗi bài bổ sung ghi chú thời hạn thông tin, và tổ chức lại cấu trúc chương:

- **Bài:** [4.1 Cơ bản lượng hóa: Scale, độ hạt và PTQ/QAT](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/41-量化基础/) | Viết lại quantize/dequantize bằng phép so «thước có vạch hạn»; độ rộng bit, trục ma trận và độ hạt thành các mục riêng; calibration tách thành một chương, nói rõ chọn dải và calibration tĩnh/động

- **Bài:** [4.2 Lượng hóa W8A8: SmoothQuant xử lý outlier activation thế nào](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/42-w8a8与smoothquant/) | Scale tương đương bổ sung công thức và suy ra trực quan từ một tích vô hướng; mục hệ số làm mượt đổi thành cách chọn cường độ chuyển α; bổ sung luồng offline, lợi ích và điều kiện phù hợp, cùng ví dụ tích vô hướng tự làm được

- **Bài:** [4.3 Weight-only INT4: GPTQ, AWQ và Marlin](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/43-weight-only-int4/) | Bổ sung cái giá và hiệu quả GPTQ, ý tưởng AWQ dùng scale thay mixed precision; thêm chọn Kernel trong vLLM, chọn công cụ lượng hóa và thứ tự debug

- **Bài:** [4.4 Lượng hóa KV Cache: KIVI và cache FP8](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/44-kv-cache量化/) | Thêm ước lượng chiếm chỗ KV; mục FP8 KV Cache viết lại theo v0.32.0, phủ chọn định dạng, calibration Scale, bỏ qua lớp nhạy, và khác biệt độ chính xác lưu trữ với tính toán; ghi chú tham số cũ `calculate_kv_scales` đã gỡ

- **Bài:** [4.5 FP8 và NVFP4/MXFP4: số thực low-bit hoạt động thế nào](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/45-fp8与nvfp4-mxfp4/) | Bổ sung 8 biên độ của E2M1, cách viết hình dạng độ hạt vi khối và định dạng của chính Scale; so Scale hai tầng của NVFP4 với Scale một tầng của MXFP4; thêm cách dùng trong vLLM v0.32.0

- **Bài:** [4.6 Chọn lượng hóa và thực chiến vLLM](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/46-量化选型与vllm实战/) | Tách phương pháp lượng hóa, định dạng checkpoint và Kernel thành ba tầng; giải thích các điểm vào lượng hóa của vLLM; chuẩn thí nghiệm cập nhật lên v0.32.0

**Module 4 · Tối ưu suy luận** Chương 5 «Speculative Decoding» — 5 bài sửa, chuẩn cấu hình nâng từ vLLM v0.31.0 lên v0.32.0, và thêm nội dung MTP:

- **Bài:** [5.1 Nguyên lý lõi speculative decoding: đoán một đoạn, rồi để mô hình lớn nghiệm thu](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-投机解码核心原理/) | Viết lại toàn bộ một vòng Draft + Verify; thêm cách verify khi «Draft chỉ cho Token, không cho xác suất»

- **Bài:** [5.2 Mô hình Draft và phương án không mô hình: đoán rẻ mà trúng hơn thế nào](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/52-draft模型与无模型方案/) | Thêm hai mục «khớp Target quan trọng hơn điểm số tự thân cao» và Draft song song; thống nhất khẩu tính ba chỉ số tỉ lệ chấp nhận, độ dài chấp nhận trung bình và phủ ứng viên

- **Bài:** [5.3 Medusa, EAGLE và MTP: tái dùng thông tin mô hình lớn để sinh nháp tốt hơn](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/53-medusa与eagle/) | Thêm mục 7 về MTP, phủ chuyển từ mục tiêu pretrain sang tăng tốc suy luận, hiệu quả thực, hình thái trong vLLM, và khám phá một forward sinh cả khối nháp; EAGLE-2 bổ sung độ tin cậy chỉ dùng để phân ngân sách

- **Bài:** [5.4 Biên lợi ích speculative decoding: tỉ lệ chấp nhận cao, vì sao vẫn có thể chậm hơn](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/54-收益边界与限制/) | Bổ sung điều kiện tăng tốc và suy giảm lợi ích theo độ dài speculation; thêm độ dài speculation động theo tải; output có cấu trúc vào thảo luận điều kiện dừng

- **Bài:** [5.5 Thực chiến speculative decoding với vLLM: đo cùng cấu hình, tỉ lệ chấp nhận và hiệu năng](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/55-vllm投机解码实战/) | Chuẩn nâng lên vLLM v0.32.0; thêm nhóm thí nghiệm EAGLE-3 thứ tư và điều kiện gắn MTP; script thu thập đổi sang chỉ số chấp nhận từng yêu cầu, cùng thứ tự debug hiện tượng thường gặp và hướng tinh chỉnh vòng sau

---

## 2026-10-12

### Nội dung mới

**Module 4 · Tối ưu suy luận** Chương 3 «Đọc mã nguồn nano-vllm» lên sóng. Lấy engine suy luận tối giản khoảng 1200 dòng [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) làm trục, đi từng module để hiểu continuous batching, PagedAttention, FlashAttention, song song tensor và CUDA Graph được tổ chức thế nào trong mã thật:

- **Bài:** [3.1 Toàn cảnh dự án và kiến trúc bốn lớp](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/31-项目全景与四层架构/) | Nhận vị trí và cấu trúc thư mục của nano-vllm; một hình kiến trúc bốn lớp phủ hết đường dữ liệu từ yêu cầu người dùng đến token ra

- **Bài:** [3.2 Cấu hình, điểm vào và toàn bộ một lần generate](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/32-配置入口与推理主循环/) | Từ example.py tách Config/SamplingParams/LLM/LLMEngine, xem vòng chính generate điều khiển add_request, step và postprocess thế nào

- **Bài:** [3.3 Khung mô hình và toán tử lõi](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/33-模型与核心算子/) | Từng lớp Qwen3: RMSNorm, RoPE, SwiGLU, Linear song song và cắt song song tensor, Embedding/LMHead song song từ vựng, cùng ghi KV của Attention và hai đường FlashAttention

- **Bài:** [3.4 KV Cache và BlockManager](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/34-kvcache与blockmanager/) | Quản KV theo khối: cấu trúc Block, ý tưởng khối của PagedAttention, prefix cache dựa xxhash, cấp phát/thu hồi khối, cùng máy trạng thái Sequence và block_table

- **Bài:** [3.5 Scheduler và continuous batching](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/35-scheduler与连续批处理/) | Lõi lịch: hàng đợi Waiting/Running kép, lịch tách Prefill/Decode, Token Budget, chunked prefill, cùng preempt và tính lại khi thiếu bộ nhớ

- **Bài:** [3.6 ModelRunner, CUDA Graph và chuẩn hiệu năng](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/36-modelrunner引擎循环与性能/) | Tensor hóa chuỗi, tiêm Context toàn cục, cắt bộ nhớ KV Cache, bắt và phát lại CUDA Graph, cùng chuẩn chính thức so với vLLM

### Điều chỉnh nội dung

- Bài cũ *vLLM quick start* chuyển xuống Chương 1 thành [1.2 vLLM quick start](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/12-vllm快速入门/), làm cửa vào thực hành của phần cơ bản suy luận.

---

## 2026-10-08

### Nội dung mới

**Module 4 · Tối ưu suy luận** Chương 4 «Lượng hóa» — 6 bài lên sóng:

- **Bài:** [4.1 Cơ bản lượng hóa: Scale, độ hạt và PTQ/QAT](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/41-量化基础/) | Từ ánh xạ số hiểu quantize/dequantize; độ rộng bit, Scale, zero-point, độ hạt, cùng calibration và khác biệt PTQ/QAT

- **Bài:** [4.2 Lượng hóa W8A8: SmoothQuant xử lý outlier activation thế nào](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/42-w8a8与smoothquant/) | Từ Activation Outlier suy ra scale tương đương của SmoothQuant; hệ số làm mượt, INT8 GEMM và luồng triển khai offline

- **Bài:** [4.3 Weight-only INT4: GPTQ, AWQ và Marlin](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/43-weight-only-int4/) | Bù sai GPTQ, scale nhận thức activation của AWQ, và Marlin biến lưu trữ INT4 thành lợi ích suy luận thế nào

- **Bài:** [4.4 Lượng hóa KV Cache: KIVI và cache FP8](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/44-kv-cache量化/) | Ràng buộc streaming của lượng hóa KV Cache, nhóm Key/Value khác nhau của KIVI, cùng calibration và biên lợi ích FP8 KV Cache của vLLM

- **Bài:** [4.5 FP8 và NVFP4/MXFP4: số thực low-bit hoạt động thế nào](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/45-fp8与nvfp4-mxfp4/) | Phân biệt lượng hóa số nguyên và số thực; phạm vi và độ chính xác FP8; scale vi khối NVFP4/MXFP4 và điều kiện chạy phần cứng

- **Bài:** [4.6 Chọn lượng hóa và thực chiến vLLM](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/46-量化选型与vllm实战/) | Chọn ứng viên lượng hóa theo nút thắt; quan hệ định dạng lượng hóa vLLM với Kernel; chạy so throughput và chất lượng sinh FP16 với AWQ INT4

---

## 2026-10-07

### Nội dung mới

**Module 4 · Tối ưu suy luận** Chương 5 «Speculative Decoding» — 5 bài lên sóng:

- **Bài:** [5.1 Nguyên lý lõi speculative decoding: đoán một đoạn, rồi để mô hình lớn nghiệm thu](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-投机解码核心原理/) | Từ nút thắt tuần tự tự hồi quy suy ra Draft + Verify, xác suất chấp nhận và phân bố dư; chứng minh đầu ra cuối theo phân bố Target

- **Bài:** [5.2 Mô hình Draft và phương án không mô hình: đoán rẻ mà trúng hơn thế nào](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/52-draft模型与无模型方案/) | So mô hình nhỏ độc lập, N-gram Prompt Lookup và Suffix Decoding; tương thích Tokenizer, khớp lịch sử, phủ ứng viên và đánh đổi tỉ lệ chấp nhận

- **Bài:** [5.3 Medusa và EAGLE: tái dùng thông tin mô hình lớn để sinh nháp tốt hơn](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/53-medusa与eagle/) | Dự đoán đa đầu Medusa, nháp cấp đặc trưng EAGLE, cây động EAGLE-2 và cải tiến huấn luyện EAGLE-3; Tree Attention và commit đường đi

- **Bài:** [5.4 Biên lợi ích speculative decoding: tỉ lệ chấp nhận cao, vì sao vẫn có thể chậm hơn](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/54-收益边界与限制/) | Xây sổ hiệu năng từ độ dài commit trung bình và thời gian cả vòng; ràng buộc chung từ batching, ngữ cảnh dài, lượng hóa, lịch và độ trễ streaming

- **Bài:** [5.5 Thực chiến speculative decoding với vLLM: đo cùng cấu hình, tỉ lệ chấp nhận và hiệu năng](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/55-vllm投机解码实战/) | Lấy vLLM v0.31.0 làm chuẩn, lần lượt khởi Decode thường, N-gram và Draft độc lập; thu chỉ số chấp nhận, output chất lượng và độ trễ/throughput

---

## 2026-09-15

### Nội dung mới

**Module 4 · Tối ưu suy luận** thêm Chương 12 «Suy luận trên thiết bị»:

- **Bài:** [12.1 Cơ bản suy luận trên thiết bị: từ xuất mô hình đến chạy trên phần cứng dị thể](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第12章-端侧推理/121-端侧推理基础/) | Hiểu có hệ thống phần cứng, định dạng mô hình, runtime, lượng hóa, bộ nhớ và Benchmark của suy luận trên thiết bị, và đi hết chuỗi triển khai tối thiểu với ExecuTorch/XNNPACK

---

## 2026-07-09

### Nội dung mới

**Module 4 · Tối ưu suy luận** các bài Chương 1 và 2 lên sóng:

- **Bài:** [1.1 Cơ bản suy luận LLM](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/11-llm推理基础/) | Một bài nói thấu logic nền của suy luận LLM: hai giai đoạn Prefill/Decode, sổ bộ nhớ KV Cache, các chỉ số TTFT/TPOT, và dùng Roofline giải thích vì sao Decode là Memory Bound

- **Bài:** [2.1 PagedAttention](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention/) | Đi sâu kỹ thuật lõi PagedAttention của vLLM: mượn ý tưởng phân trang bộ nhớ ảo, dùng Block Table loại phân mảnh KV Cache, và hỗ trợ chia sẻ prefix cùng Copy-on-Write

- **Bài:** [2.2 Continuous Batching](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching/) | Đi sâu Continuous Batching: Iteration-level Scheduling để yêu cầu đến là ghép, xong là ra, kéo tận dụng GPU từ khoảng 30% lên trên 80%

- **Bài:** [2.3 Prefix Cache và RadixAttention](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/23-prefix-cache-与-radixattention/) | Đi sâu prefix cache: hash-based automatic prefix cache của vLLM tái dùng khối KV prefix chung thế nào, và RadixAttention của SGLang dùng Radix Tree để chia sẻ prefix hiệu quả hơn thế nào

- **Bài:** [2.4 Chunked Prefill và lịch thống nhất](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/24-chunked-prefill-与统一调度/) | Đi sâu Chunked Prefill: cắt Prefill Prompt dài thành khối để hết can thiệp Decode; vLLM V1 dùng scheduler ngân sách Token thống nhất san biên Prefill/Decode thế nào

- **Bài:** [2.5 Backend Attention và tối ưu đồ thị](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/25-attention-后端与图优化/) | Đi sâu backend Attention cắm được của vLLM (FlashAttention/FlashInfer/FlashMLA/Triton) và tối ưu đồ thị: CUDA Graph và torch.compile cắt chi phí khởi động CPU giai đoạn Decode

**Module 3 · Huấn luyện phân tán** các bài chương giao tiếp tập hợp, tối ưu hóa và song song dữ liệu lên sóng:

- **Bài:** [2.1 Nguyên thủy giao tiếp tập hợp chi tiết](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/21-集合通信原语详解/) | Ngữ nghĩa vào/ra của AllReduce/ReduceScatter/AllGather/All-to-All, thuật toán Ring AllReduce, và cách định lượng chi phí một lần giao tiếp

- **Bài:** [3.1 Tối ưu hóa: từ SGD đến AdamW và phân tích chi phí bộ nhớ](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/31-优化器原理与显存开销分析/) | Tiến hóa tối ưu hóa từ SGD đến AdamW, chi phí bộ nhớ trạng thái tối ưu hóa và vì sao cần trọng số chủ FP32, cùng LARS/LAMB cho ngữ cảnh Batch lớn

- **Bài:** [4.1 Song song dữ liệu chi tiết](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/41-数据并行详解/) | Tiến hóa DP/DDP/FSDP, đồng bộ gradient, và toán cho lượng giao tiếp / sổ bộ nhớ; khác biệt thực giữa AllReduce và sharding tham số

- **Bài:** [4.2 Song song dữ liệu PyTorch từ nguyên lý đến thực chiến](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/42-pytorch-数据并行从原理到实战/) | Toàn stack song song dữ liệu PyTorch: nguyên thủy giao tiếp, DDP, sharding tham số FSDP, và thực chiến đa máy đa card kèm mẫu mã đầy đủ

---

## 2026-07-07

### Nội dung mới

- **Bài:**  [1.1 Tổng luận huấn luyện phân tán: sổ bộ nhớ và toàn cảnh năm chiến lược song song](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/11-分布式训练总论/) | Tự tính sổ bộ nhớ huấn luyện; tổng quan năm chiến lược song song và nguyên tắc chọn

- **Bài:**  [1.2 Dựng môi trường và khởi động phân tán](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/12-环境搭建与分布式启动/) | rank/local_rank/world_size, khởi động torchrun và rendezvous, debug NCCL, script DDP tối thiểu

---

## 2026-07-06

### Nội dung mới

Cập nhật dàn ý tutorial phần huấn luyện phân tán:
| Chương | Nội dung chính |
|------|----------|
| Chương 1 Tổng luận huấn luyện phân tán | Vì sao cần huấn luyện phân tán, sổ bộ nhớ huấn luyện, toàn cảnh năm chiến lược song song, dựng môi trường và khởi động DDP |
| Chương 2 Nguyên thủy giao tiếp tập hợp | Ngữ nghĩa AllReduce/ReduceScatter/AllGather/All-to-All, thuật toán Ring, phân tích định lượng lượng giao tiếp |
| Chương 3 Tối ưu hóa | Tiến hóa SGD/Adam/AdamW, phân tích chi phí bộ nhớ trạng thái tối ưu hóa, tối ưu hóa Batch lớn LAMB/LARS |
| Chương 4 Song song dữ liệu | DataParallel, DistributedDataParallel, FSDP |
| Chương 5 Dòng ZeRO | Chiến lược cắt ZeRO-1/2/3 và chi phí giao tiếp, offload bộ nhớ dị thể ZeRO-Offload/Infinity |
| Chương 6 Song song tensor và song song chuỗi | Megatron Column/Row Parallel Linear, vị trí chèn giao tiếp, tối ưu bộ nhớ activation song song chuỗi |
| Chương 7 Song song pipeline | Bản chất vấn đề Bubble, GPipe, 1F1B, lịch Interleaved |
| Chương 8 Độ chính xác hỗn hợp và tối ưu bộ nhớ | Độ chính xác hỗn hợp FP16/BF16/FP8, tích lũy gradient, Activation Checkpointing |
| Chương 9 Huấn luyện chuỗi dài và song song ngữ cảnh | Thế khó O(s²) của Attention, Ring Attention, DeepSpeed-Ulysses, Megatron Context Parallel |
| Chương 10 Song song MoE | Cơ chế Router, giao tiếp All-to-All của Expert Parallelism, tổ hợp EP×DP×TP×PP và cân bằng tải |
| Chương 11 Song song 3D và chiến lược hỗn hợp | Thiết kế tôpô TP×PP×DP×EP×CP, ánh xạ miền giao tiếp, xếp rank, chọn cụm |


---

## 2026-06-25

### Nội dung mới

- **Bài:**  [6.1 FlashAttention V1 chi tiết](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/61-flashattention-v1详解) | Bài này phân tích sâu nguyên lý và chi tiết triển khai FlashAttention V1: Tiling + Online Softmax hạ bộ nhớ thêm của Attention từ $O(N^2)$ xuống $O(N)$, đồng thời giảm mạnh truy cập HBM, tăng tốc không mất độ chính xác

- **Bài:**  [6.2 FlashAttention V2 chi tiết](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/62-flashattention-v2详解) | Bài bài này phân tích sâu các cải tiến lõi của FlashAttention V2 so với V1: đổi thứ tự vòng trong/ngoài, tối ưu phân việc trong thread block, giảm phép không phải nhân ma trận

---

## 2026-06-06

### Nội dung mới

- **Bài:**  [5.2 Triển khai và tối ưu CUDA Online Softmax](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/52-cuda-online-softmax实现) | Từ suy ra thuật toán đến triển khai và tối ưu CUDA Kernel Online Softmax — kiến thức tiền đề lõi để hiểu FlashAttention

---

## 2026-06-05

### Nội dung mới

- **Bài:**  [5.1 Triển khai Softmax CUDA ngây thơ và tối ưu](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/51-cuda-softmax朴素实现优化) | Từ triển khai ngây thơ, lần lượt đưa Safe Softmax, song song cấp Block, Warp Shuffle, tải vector hóa và các tối ưu liên quan

---

## 2026-06-02

### Nội dung mới

- **Bài:** [4.1 Tối ưu hiệu năng toán tử CUDA GEMM](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/41-cuda-gemm算子性能优化) | Từ triển khai ngây thơ qua tiling ba cấp Block-Warp-Thread, tải vector hóa, loại Bank Conflict, double buffering… — cả phương pháp tối ưu CUDA GEMM

---

## 2026-05-18

### Nội dung mới

- **Bài:** [3.1 Tối ưu toán tử CUDA Reduce](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/31-cuda-reduce算子优化) | Reduce là một trong những toán tử GPU cơ bản nhất, cũng tốt nhất để học tư duy song song. Từ triển khai đơn giản nhất, bài lần lượt đưa nguyên thủy cấp Warp, tải vector hóa, xử lý nhiều phần tử… — mỗi bước có so hiệu năng và giải thích nguyên lý, để thật sự hiểu cách viết Kernel nhanh.

---

## 2026-05-16

### Nội dung mới

- **Module 3 · Huấn luyện phân tán** thêm [Chương 2: Tối ưu hóa](/AIInfraGuide/distributed/模块三-分布式训练/第2章-优化器) | Theo tiến hóa SGD → Momentum → Adam → AdamW và cấu trúc trạng thái nội bộ; tự tính chi phí bộ nhớ tối ưu hóa (AdamW chiếm 75% bộ nhớ huấn luyện); nắm khi nào tối ưu hóa Batch lớn LAMB/LARS phù hợp; nền cho tối ưu bộ nhớ ZeRO và huấn luyện độ chính xác hỗn hợp phía sau.

### Điều chỉnh cấu trúc

- **Module 3 · Huấn luyện phân tán** đánh số lại chương: Chương 2–7 cũ thành 3–8; Chương 2 «Tối ưu hóa» mới chèn giữa tổng luận và song song dữ liệu
- Đồng bộ lộ trình học AI Infra và mục lục chương Module 3 trong README

---

## 2026-05-13

### Nội dung mới

- **Bài:** [2.4 Đồng bộ và phép nguyên tử](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/24-同步与原子操作) | Cơ chế đồng bộ đúng là nền của chương trình song song không Bug. Bài đi `__syncthreads()` đồng bộ trong block CUDA, đồng bộ cấp Warp, Memory Fence, cùng lúc nào dùng atomic, cái giá hiệu năng và cách tối ưu — để viết mã song song hiệu năng cao mà không bỏ đúng đắn.

---

## 2026-05-12

### Nội dung mới

- **Bài:** [2.2 Tối ưu truy cập bộ nhớ](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/22-内存访问优化) | Hiệu suất truy cập bộ nhớ là đòn bẩy lớn nhất lên hiệu năng CUDA Kernel. Nguyên lý và cách kiểm Coalesced Access, nguyên nhân Bank Conflict shared memory và cách sửa bằng Padding, cùng tải vector hóa (float4/int4) để nâng tận dụng băng thông.

- **Bài:** [2.3 Occupancy và phân bổ tài nguyên](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/23-occupancy与资源分配) | Occupancy đo tỉ lệ Warp đang hoạt động trên SM so với tối đa lý thuyết — chỉ số tinh chỉnh CUDA Kernel lõi. Định nghĩa, cách tính, ba giới hạn (thanh ghi/shared memory/kích thước Block), và vì sao Occupancy không phải càng cao càng tốt — mục tiêu thật là cân bằng giữa ẩn độ trễ và dùng tài nguyên.

---

## 2026-05-11

### Nội dung mới

- **Bài:** [2.1 Warp và mô hình thực thi](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/21-warp与执行模型) | Đi sâu đơn vị thực thi lõi của GPU — Warp

---
## 2026-05-09

### Nội dung mới

- **Bài:** [Mô hình bộ nhớ CUDA](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/13-cuda内存模型) | Bộ nhớ toàn cục, shared memory, thanh ghi, bộ nhớ hằng và unified memory — đặc điểm, tình huống phù hợp và mẹo tối ưu

- **Bài:** [Kernel hữu dụng đầu tiên: cộng vector](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/14-第一个实用kernel) | Một case cộng vector đầy đủ đi hết đường lập trình CUDA

---
## 2026-05-08

### Nội dung mới

- **Bài:** [1.1 Dựng môi trường phát triển CUDA](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/11-cuda开发环境搭建)
- **Bài:** [1.2 Mô hình lập trình CUDA](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/12-cuda编程模型)

---
## 2026-05-07

### Nội dung mới

- **Bài:** [✅ Hướng dẫn nhập môn nhanh CUDA 🔥](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/cuda编程入门指南/)


---
## 2026-04-28

### Nội dung mới

- **Bài:** [🔥 Mạng giao tiếp cụm và NCCL: khung xương giao tiếp của huấn luyện phân tán](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/communication/collective-communication-primer/)

---
## 2026-04-27

### Nội dung mới

- **Bài:** [5.1 Tiến hóa kiến trúc GPU NVIDIA: từ Volta đến Blackwell](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpunvidia-gpu-evolution/)

---
## 2026-04-26

### Nội dung mới

- **Bài:** [🔥 Cơ bản GPU: từ kiến trúc phần cứng đến tính AI](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpu/gpu-basics/)

---
## 2026-04-25

### Nội dung mới

- **Bài:** [🔥 Nhập môn nhanh framework PyTorch](/AIInfraGuide/prerequisites/模块一-前置知识/pyroch/pytorch框架入门)

---
## 2026-04-24

### Nội dung mới

- **Bài:** [3.9 Tokenization và word embedding](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/39-tokenization与词嵌入)


---
## 2026-04-23

### Nội dung mới

- **Bài:** [3.8 Từ Transformer đến sinh tự hồi quy LLM đi sâu](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/38-从transformer到llm自回归生成深入理解)

---

## 2026-04-21

### Nội dung mới

- **Bài:** [3.6 LayerNorm và residual connection đi sâu](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/36-layernorm与残差连接深入理解)
- **Bài:** [3.7 Phân tích đầy đủ Transformer Decoder Block](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/37-transformer-decoder-block完整解析)

---

## 2026-04-20

### Nội dung mới

- **Bài:** [3.4 Mạng feed-forward (FFN) của Transformer đi sâu](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/34-transformer前馈网络ffn深入理解)
- **Bài:** [3.5 Mã hóa vị trí Transformer đi sâu](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/35-transformer位置编码深入理解)

---

## 2026-04-18

### Nội dung mới

- **Bài:** [3.3 Cơ chế Self-Attention đi sâu](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/33-Self-Attention机制深入理解)

---

## 2026-04-17

### Nội dung mới

- **Khu phỏng vấn**: tổng cộng **180+** buổi phỏng vấn thật, phủ **40+** công ty, tổ chức theo bậc. [Duyệt online →](https://caomaolufei.github.io/AIInfraGuide/interview)
- Các công ty gồm ByteDance, Alibaba, Tencent, Baidu, Kuaishou, Meituan, Ant Group, NVIDIA, MiniMax, NIO, XPeng, Li Auto và nữa.
- **Cổng cộng đồng**: thêm [nhóm WeChat trao đổi]((/AIInfraGuide/about/author)), chào mọi người tham gia thảo luận

### Cải thiện nội dung

- Cập nhật README và bổ sung ảnh giới thiệu dự án

---

## 2026-04-16

### Cập nhật lớn

- **Nâng cấp toàn diện khung kho tri thức**: chi tiết bốn module học, điều chỉnh bố cục trang
  - **[Module 1 · Kiến thức tiền đề](/AIInfraGuide/prerequisites)** (6 chương): [Cơ bản ngôn ngữ lập trình](/AIInfraGuide/prerequisites/模块一-前置知识/第1章-编程语言基础), [Cơ bản toán](/AIInfraGuide/prerequisites/模块一-前置知识/第2章-数学基础), [Kiến trúc Transformer chi tiết](/AIInfraGuide/prerequisites/模块一-前置知识/第3章-Transformer架构详解), [Framework PyTorch](/AIInfraGuide/prerequisites/模块一-前置知识/第4章-PyTorch框架), [Tổng quan phần cứng GPU](/AIInfraGuide/prerequisites/模块一-前置知识/第5章-GPU硬件概论), [Cơ bản giao tiếp tập hợp](/AIInfraGuide/prerequisites/模块一-前置知识/第6章-集合通信基础)
  - **[Module 2 · Lập trình CUDA và tối ưu toán tử](/AIInfraGuide/cuda)** (8 chương): [Nhập môn CUDA](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第1章-CUDA编程入门), [Cơ bản tối ưu hiệu năng](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第2章-CUDA性能优化基础), Thực chiến Reduce, Thực chiến GEMM, Thực chiến Softmax, [Toán tử Attention](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第6章-Attention算子), [Trình biên dịch AI](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第7章-AI编译器), [Chuỗi công cụ phân tích hiệu năng](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第8章-性能分析工具链)
  - **[Module 3 · Huấn luyện phân tán](/AIInfraGuide/distributed)** (7 chương): [Tổng luận huấn luyện phân tán](/AIInfraGuide/distributed/模块三-分布式训练/第1章-分布式训练总论), [Song song dữ liệu](/AIInfraGuide/distributed/模块三-分布式训练/第2章-数据并行), [Dòng ZeRO](/AIInfraGuide/distributed/模块三-分布式训练/第3章-ZeRO系列), [Song song tensor và song song chuỗi](/AIInfraGuide/distributed/模块三-分布式训练/第4章-张量并行与序列并行), [Song song pipeline](/AIInfraGuide/distributed/模块三-分布式训练/第5章-流水线并行), [Song song 3D và chiến lược huấn luyện hỗn hợp](/AIInfraGuide/distributed/模块三-分布式训练/第6章-其他显存优化技术), [Thực chiến framework huấn luyện](/AIInfraGuide/distributed/模块三-分布式训练/第7章-训练框架实战)
  - **[Module 4 · Tối ưu suy luận](/AIInfraGuide/inference)** (8 chương): Cơ bản suy luận, [Kỹ thuật lõi engine suy luận](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术), Framework suy luận, [Lượng hóa](/AIInfraGuide/inference/模块四-推理优化/第4章-量化), [Speculative Decoding](/AIInfraGuide/inference/模块四-推理优化/第5章-Speculative-Decoding), [Kiến trúc tách PD](/AIInfraGuide/inference/模块四-推理优化/第6章-PD解耦架构), [Phân tích hiệu năng và Benchmark](/AIInfraGuide/inference/模块四-推理优化/第7章-性能分析与Benchmark), [Chọn và thực chiến end-to-end](/AIInfraGuide/inference/模块四-推理优化/第8章-推理优化选型与端到端实战)

### Nội dung mới

- **Chuỗi Transformer**:
  - [Nhập môn kiến trúc Transformer](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/Transformer架构快速入门)
  - [3.1 Vì sao kỹ sư AI Infra phải hiểu Transformer](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/31-ai-infra工程师为什么必须懂transformer)
  - [3.2 Toàn cảnh Transformer và triển khai mã](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/32-transformer全貌及代码实现)
- **Tài liệu khóa học**: 150+ bài con chi tiết lên sóng, phủ mọi chương của bốn module
- **Blog**: thêm trang [Về tác giả](/AIInfraGuide/about/author)

---

## 2026-04-15

### Dự án lên sóng

- Kho tri thức AIInfraGuide chính thức lên sóng
- Phát hành tổng quan [Lộ trình học AI Infra](/AIInfraGuide/guides/AI Infra学习路线)
- Phát hành hướng dẫn nhập môn [Hiểu AI Infra từ zero](/AIInfraGuide/blog/ai-infra-introduction)

---

## 2026-04-14

### Khởi tạo dự án

- Hoàn thành dựng khung website (Astro + Starlight)
- Khởi tạo kho dự án, phát hành commit đầu
