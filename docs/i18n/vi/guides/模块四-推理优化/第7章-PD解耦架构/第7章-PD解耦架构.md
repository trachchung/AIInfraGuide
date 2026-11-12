---
title: "Chương 7: Kiến trúc tách Prefill/Decode"
description: "Hiểu vấn đề mixed batching P/D, các phương án tách DistServe/Splitwise, truyền KV Cache, Goodput và lập lịch nhạy SLO"
tags: ["Tách P/D", "DistServe", "Splitwise", "Truyền KV Cache", "Goodput", "SLO"]
sourceHash: "9ef6708bcc011816"
---

## Giới thiệu chương

Prefill và Decode có đặc tính tính toán rất khác nhau (compute bound so với memory bound); gộp chung sẽ nhiễu nhau. Chương này đi sâu vào thiết kế và thách thức của kiến trúc tách (disaggregation), rồi hạ cánh vào khả năng Disaggregated Prefill của vLLM.

**Vấn đề mixed batching** định lượng nhiễu Prefill lên Decode: Decode P95 TPOT có thể chậm thêm 3–5 lần.

**Thiết kế kiến trúc tách** gồm DistServe (OSDI'24, lập luận có hệ thống cho tách P/D), Splitwise (ISCA'24, phân vào các GPU pool khác nhau), TaiChi (khung thống nhất cho gom và tách) và MLC Microserving (điều phối xuyên engine).

**Truyền KV Cache và Connector** đi vào thách thức kỹ thuật cốt lõi nhất của kiến trúc tách: đưa KV Cache mà nút Prefill đã tính sang nút Decode sao cho hiệu quả. Giới thiệu abstraction KV Connector của vLLM, các backend truyền như NIXL/NCCL, và ảnh hưởng của băng thông liên nút lên độ trễ tổng thể.

**Goodput và lập lịch nhạy SLO** nhấn mạnh Raw QPS không bằng trải nghiệm người dùng; Goodput (throughput còn đạt SLO) mới là mục tiêu tối ưu thật sự.

**Thách thức của kiến trúc tách** gồm độ phức tạp của scheduler, suy ra tỉ lệ tài nguyên GPU pool P/D, và khi nào nên / không nên dùng tách.

**Thực hành Disaggregated Prefill với vLLM**: dùng triển khai tách của vLLM để dựng tải hỗn hợp, định lượng mức nhiễu P/D và suy ra tỉ lệ GPU pool P/D cho một workload cho trước.

## Các bài trong chương

- **7.1 Vấn đề mixed batching**: phân tích định lượng nhiễu Prefill lên Decode
- **7.2 Thiết kế kiến trúc tách**: DistServe, Splitwise, TaiChi, Microserving
- **7.3 Truyền KV Cache và Connector**: KV Connector, NIXL/NCCL, áp lực băng thông
- **7.4 Goodput và lập lịch nhạy SLO**: lấy throughput hữu hiệu làm mục tiêu tối ưu
- **7.5 Thách thức và tỉ lệ pool**: độ phức tạp lập lịch, tỉ lệ GPU pool P/D
- **7.6 Thực hành Disaggregated Prefill với vLLM**: triển khai tách và định lượng nhiễu
