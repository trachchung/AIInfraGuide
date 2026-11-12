---
title: "Chương 8: Triển khai và vận hành production"
description: "Đóng gói container và triển khai dịch vụ suy luận vLLM trên Kubernetes, cùng observability, autoscaling, cân bằng tải và lập kế hoạch dung lượng"
tags: ["Triển khai production", "Kubernetes", "Observability", "Autoscaling", "Lập kế hoạch dung lượng"]
sourceHash: "e7b8964e53eac356"
---

## Giới thiệu chương

Các chương trước giúp dịch vụ suy luận chạy nhanh và tiết kiệm. Chương này giải quyết việc phục vụ traffic thật một cách ổn định: đóng gói, điều phối, quan sát, mở rộng/thu hẹp và lập kế hoạch dung lượng. Đây là chặng cuối từ demo đến production, lấy triển khai vLLM làm ví dụ.

Phần container hóa và triển khai Kubernetes giải thích cách đóng gói dịch vụ vLLM thành image rồi điều phối trên K8s: yêu cầu tài nguyên GPU, readiness/liveness probe (tải model mất nhiều thời gian nên timeout của probe cần nới lỏng), chiến lược kéo image và weight, cùng các giải pháp triển khai production như vLLM Production Stack và KServe.

Observability (khả năng quan sát) là mắt của vận hành: vLLM mở endpoint Prometheus metrics (TTFT, TPOT, số request đang chạy/đang chờ, mức sử dụng KV Cache, số lần preemption, …). Kết hợp Grafana dashboard và log có cấu trúc, ta định vị được lỗi và theo dõi được hiệu năng.

Autoscaling và cân bằng tải phủ việc phân phối request khi chạy nhiều replica: HPA dựa trên QPS / độ dài hàng đợi / mức sử dụng KV Cache, cùng routing nhận biết prefix (prefix-aware) để đưa các request dùng chung System Prompt về cùng một replica nhằm tăng tỉ lệ trúng cache.

⚠️ **Cảnh báo bảo mật**: server tương thích OpenAI của vLLM mặc định không xác thực. Expose thẳng ra internet công cộng đồng nghĩa mở tính toán và dữ liệu. Triển khai production bắt buộc thêm kiểm tra API Key, TLS và cô lập mạng ở lớp gateway — chương này nêu rõ điểm này để tránh đưa dịch vụ lên mà không bảo vệ.

Lập kế hoạch dung lượng đưa ra cách suy ngược từ SLO và QPS đỉnh để ra số GPU cần: kết hợp trần throughput từ load test một replica, ràng buộc VRAM và hệ số dự phòng để ước lượng quy mô cluster cùng chi phí.

## Các bài trong chương

- 8.1 Container hóa và triển khai Kubernetes: image, lập lịch GPU, probe, Production Stack/KServe
- 8.2 Observability: Prometheus metrics, Grafana dashboard, log và cảnh báo
- 8.3 Autoscaling và cân bằng tải: chiến lược HPA, routing nhận biết prefix, xác thực và bảo mật
- 8.4 Lập kế hoạch dung lượng: suy ngược quy mô cluster và chi phí từ SLO cùng QPS đỉnh
