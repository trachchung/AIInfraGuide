---
title: "Ant Group AI Infra thực tập vòng một (2)"
description: "Câu hỏi phỏng vấn Ant Group AI Infra thực tập vòng một thật, gồm tối ưu suy luận, tối ưu huấn luyện, tối ưu toán tử và các hướng liên quan"
tags: ["Tối ưu suy luận", "Tối ưu huấn luyện", "Tối ưu toán tử"]
sourceHash: "74f1cf22a8ef216b"
---

### Kiến thức nền

1. W8A8, W4A16 lần lượt nghĩa là gì? Vì sao khi weight quantize xuống 4-bit, activation ở hầu hết tình huống vẫn cần giữ độ chính xác cao hơn?
2. Uniform quantization và non-uniform quantization có đặc điểm gì? Non-uniform trả giá gì về độ phức tạp tính toán, chi phí lưu scale/zero point và thích nghi tăng tốc phần cứng?
3. Phân tích nút thắt hiệu năng trên chuỗi suy luận thế nào? Cách tối ưu hiệu quả nhất là gì? Nếu tối ưu rồi vẫn lệch mục tiêu, khoảng cách chủ yếu đến từ đâu?
4. Làm sao biết một nhóm toán tử có phù hợp fusion không? Tiêu chí nào? (độ dài chuỗi toán tử, tần suất xuất hiện, quan hệ phụ thuộc, output nút trung gian có bị nhánh khác tham chiếu không)
5. Khi profiling hiệu năng, chú ý những chỉ số nào? (tách tổng thời gian chạy, warp divergence, hiệu quả pipeline trong tình huống multi-stream)

### Kinh nghiệm dự án

6. Giới thiệu thực hành tối ưu đơn toán tử: thay bản viết tay bằng thư viện hiệu năng cao thế nào; unit test và test cả mạng thế nào?
7. Mô tả định vị và sửa vấn đề ổn định trong suy luận nhiều vòng (phân tích nguyên nhân gốc của crash ngẫu nhiên và OOM).
8. Trong static graph, sàng lọc tổ hợp toán tử phù hợp fusion thế nào?

### Bài lập trình (CUDA)

9. Triển khai toán tử Histogram: đầu vào là các số nguyên trong khoảng 0–9; đầu ra là số đếm từng bucket. Thảo luận vấn đề hiệu năng của bản atomic thuần, và phương án tối ưu reduce qua histogram cục bộ trên shared memory.
