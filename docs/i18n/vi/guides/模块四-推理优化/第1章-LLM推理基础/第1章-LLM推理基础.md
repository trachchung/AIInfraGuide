---
title: "Chương 1: Cơ bản về suy luận LLM"
description: "Hai giai đoạn sinh văn bản của LLM (prefill và decode), cách KV cache hoạt động và các chỉ số quan trọng khi đo hiệu năng suy luận"
tags: ["Suy luận LLM", "Prefill", "Decode", "KV Cache", "TTFT", "TPOT", "Roofline"]
sourceHash: "06e92a77b36ae05a"
---

## Giới thiệu chương

Suy luận (inference) là bước đưa mô hình đã huấn luyện vào sử dụng thực tế. Chương này xây dựng các khái niệm cơ bản và cách phân tích hiệu năng mà mọi kỹ thuật tối ưu phía sau đều dựa vào. Xuyên suốt module, vLLM là ví dụ chính.

Phần về quá trình sinh tự hồi quy (autoregressive generation) giải thích hai giai đoạn của suy luận LLM. Prefill xử lý toàn bộ prompt trong một lần và bị giới hạn bởi tính toán (compute bound). Decode sinh từng token một và bị giới hạn bởi băng thông bộ nhớ (memory bound), nên decode dùng GPU kém hiệu quả hơn prefill rất nhiều.

KV cache là khái niệm trung tâm của tối ưu suy luận. Nó lưu lại các key và value đã tính để không phải tính lại. Chương này đi qua vòng đời của KV cache (cấp phát, ghi dữ liệu, đọc, giải phóng), cách tính dung lượng bộ nhớ GPU nó chiếm và vấn đề phân mảnh.

Phần chỉ số hiệu năng gồm TTFT (thời gian đến token đầu tiên), TPOT/ITL (thời gian cho mỗi token), throughput (thông lượng), độ trễ đuôi P50/P95/P99 và goodput, tức thông lượng vẫn đạt SLO.

Phần phân tích điểm nghẽn giới thiệu cường độ số học (arithmetic intensity) và mô hình Roofline. Từ nguyên lý cơ bản, hai công cụ này cho thấy vì sao decode là memory bound và không tận dụng hết sức tính toán của GPU, đồng thời cho ta một cách thống nhất để phân tích mọi kỹ thuật tối ưu ở các chương sau.

Phần cuối đi hết luồng xử lý một yêu cầu: tokenize, prefill, vòng lặp decode, sampling và detokenize. Với mỗi bước, ta xem nó tính toán kiểu gì, chỗ nào có thể thành điểm nghẽn, và so sánh với cách vLLM thực sự chạy.

## Các bài trong chương

- 1.1 Quá trình sinh tự hồi quy: so sánh đặc điểm tính toán của prefill và decode
- 1.2 Cơ chế KV cache: vòng đời, công thức tính bộ nhớ, phân mảnh
- 1.3 Các chỉ số hiệu năng chính: TTFT, TPOT/ITL, throughput, độ trễ đuôi, goodput
- 1.4 Phân tích điểm nghẽn: cường độ số học, mô hình Roofline, bản chất của memory bound
- 1.5 Từng bước của luồng suy luận: từ tokenize đến detokenize và cách tìm điểm nghẽn
