---
title: "Chương 6: Suy luận phân tán và triển khai mô hình lớn"
description: "Hiểu chi phí lưu trữ và giao tiếp của TP, PP, DP, EP trong suy luận, nắm triển khai vLLM một máy và nhiều nút, MoE DP+EP, DCP, rồi chọn layout bằng thí nghiệm đối chiếu"
tags: ["Suy luận phân tán", "Tensor parallelism", "Pipeline parallelism", "Expert Parallel", "Ray", "vLLM"]
sourceHash: "610ad9898850821c"
---

## Giới thiệu chương

Khi mô hình lớn đến mức một card không chứa nổi, hoặc một bản sao không đủ throughput, suy luận phải tách ra nhiều card, nhiều máy. Suy luận và huấn luyện dùng chung bộ thuật ngữ song song, nhưng mục tiêu khác: huấn luyện quan tâm đồng bộ gradient và hội tụ; suy luận quan tâm weight với KV Cache có nằm vừa không, cùng TTFT, TPOT và throughput.

Chương này đi từ "cắt thế nào" đến "triển khai và kiểm chứng thế nào":

- **Chọn phương án**: dùng sổ VRAM để tách hai bài toán "không chứa nổi" và "không đủ nhanh", so sánh TP, PP, DP, EP mỗi cái cắt gì và phải đồng bộ gì, rồi đưa thứ tự chọn độ song song.
- **TP và PP**: lượng dữ liệu và độ trễ của hai AllReduce mỗi lớp ở TP, nhân bản KV head dưới GQA, phụ thuộc vào NVLink; PP truyền gì giữa các đoạn, vì sao cần nhiều batch bay cùng lúc, và cách ghép TP×PP.
- **DP và EP**: nhiều bản sao cho mô hình Dense; với MoE là attention DP + expert EP, hai lần All-to-All, đồng bộ khi không có token (idle sync), lệch tải và EPLB, cùng layout online của DeepSeek-V3.
- **Triển khai**: hai runtime nhiều nút Ray và multi-process, layout rank, khởi động MoE nhiều nút DP + EP, cấu hình InfiniBand và GPUDirect RDMA; cuối cùng so ba layout trên một máy 8 card, và giới thiệu DCP để bỏ nhân bản KV ở ngữ cảnh dài.

> ⏰ **Độ cập nhật:** Phần nguyên lý không phụ thuộc framework; tham số, giá trị mặc định, log và chỉ số liên quan vLLM dựa trên tài liệu và mã nguồn v0.32.0 (phát hành 2026-10-18); số liệu DeepEP dựa trên v1.2.1. Tham số phân tán đổi khá nhiều ở vài phiên bản gần đây; khi đổi phiên bản hãy đối chiếu tài liệu chính thức.

## Các bài trong chương

- [**6.1 Tổng quan chiến lược song song suy luận: không chứa nổi và không đủ nhanh là hai chuyện khác nhau**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/61-推理并行策略总览/): sổ VRAM, so sánh lưu trữ và giao tiếp của bốn cách cắt, thứ tự chọn độ song song và script ước lượng
- [**6.2 Suy luận tensor parallelism: cắt một lớp thành vài phần, mỗi lớp đồng bộ hai lần**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/62-张量并行推理/): cắt theo cột và theo hàng, mô hình lượng dữ liệu và độ trễ AllReduce, nhân bản KV head, phụ thuộc NVLink và Custom AllReduce
- [**6.3 Suy luận pipeline parallelism: cắt theo lớp, dùng nhiều batch để lấp đầy pipeline**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/63-流水线并行推理/): lượng giao tiếp giữa các đoạn, hàng đợi batch ở Decode, chia lớp không đều, độ trễ một request và tổ hợp TP×PP
- [**6.4 Data parallelism và expert parallelism: mở thêm bản sao, và để token đi tìm expert**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/64-数据并行与专家并行/): Dense DP, Dispatch/Combine của MoE, attention DP + expert EP, EPLB, DeepEP và triển khai DeepSeek-V3
- [**6.5 Triển khai nhiều nút: dùng Ray hoặc multi-process trải một dịch vụ lên nhiều máy**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/65-多节点部署/): cụm Ray và Placement Group, multi-process `--nnodes`, DP + EP nhiều nút, cấu hình mạng NCCL và gỡ lỗi
- [**6.6 Thực hành triển khai phân tán vLLM: cùng 8 card, đo ba cách cắt**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/66-vllm分布式部署实战/): ước lượng VRAM, đối chiếu log và benchmark cho TP8, TP4×PP2, TP4×DP2, cùng DCP
