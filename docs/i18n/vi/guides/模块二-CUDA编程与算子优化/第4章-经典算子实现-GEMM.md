---
title: "Chương 4: Hiện thực toán tử cổ điển — GEMM"
description: "Từ nhân ma trận naive đến tiling Block-Warp-Thread, truy cập vector hóa, loại bỏ Bank Conflict và double buffering — nắm trọn phương pháp tối ưu CUDA GEMM"
tags: ["CUDA", "GEMM", "Nhân ma trận", "Tiling", "Tensor Core", "cuBLAS"]
sourceHash: "475c129fa983c763"
---

## Giới thiệu chương

GEMM (General Matrix Multiply, nhân ma trận tổng quát) là toán tử trung tâm trong deep learning — linear layer, phép chiếu QKV của Attention và tính toán FFN về bản chất đều là GEMM. Chương này hiện thực GEMM hiệu năng cao từ đầu để hiểu cầu nối giữa phần cứng và framework phía trên.

1. Bối cảnh và định nghĩa bài toán giới thiệu vị trí của GEMM trong deep learning, quy mô bài toán, đặc trưng tính toán/truy cập bộ nhớ, và mục tiêu tối ưu.

2. Phương pháp phân tích hiệu năng đưa vào cường độ số học (arithmetic intensity), mô hình Roofline và các chỉ số đánh giá, tạo khung phân tích định lượng cho từng bước tối ưu.

3. Hiện thực naive làm baseline là kernel ba vòng lặp thẳng nhất. Nó lộ điểm nghẽn Memory-Bound do truy cập global memory lặp lại.

4. Tối ưu tiling cấp Thread Block cắt ma trận thành Tile đưa vào shared memory, phân tích toán học mức giảm truy cập global memory, và hoàn thành vòng tái sử dụng dữ liệu đầu tiên.

5. Tối ưu tiling cấp Thread giữ kết quả khối con trong thanh ghi; mỗi thread tính nhiều phần tử đầu ra, nâng thêm mật độ tính toán và tỷ lệ tái sử dụng dữ liệu.

6. Tối ưu phân khối cấp Warp tổ chức cấu trúc Tile theo Warp, ánh xạ thread khớp đơn vị thực thi phần cứng, giảm tính toán thừa và chi phí đồng bộ trong Warp.

7. Truy cập vector hóa với `float4` gộp load/store, giảm số lệnh truy cập bộ nhớ và nâng tỷ lệ dùng băng thông hiệu dụng.

8. Loại bỏ Bank Conflict dùng Padding hoặc sắp xếp lại địa chỉ để phá xung đột bank khi truy cập Shared Memory, đưa thông lượng shared memory về gần đỉnh.

9. Double buffering và pipeline mở hai Buffer trong shared memory để chồng load dữ liệu lên tính toán, che độ trễ global memory.

10. Pipeline copy bất đồng bộ trên Ampere dùng `cp.async` để bất đồng bộ hóa copy từ global memory sang shared memory, dựng software pipeline nhiều cấp và ép thêm độ trễ.

11. Tối ưu cấp SASS và lập lịch thanh ghi đi sâu chỉnh lệnh: cấp phát thanh ghi, sắp xếp lệnh và Bank Conflict; đối chiếu mã SASS của cuBLAS để hiểu hiệu năng cực hạn đến từ đâu.

12. Hướng dẫn chọn tham số và tinh chỉnh thực tế tổng kết cách chọn kích thước Tile, bố cục thread, số Block, cùng hướng tinh chỉnh theo quy mô ma trận và phần cứng khác nhau.
