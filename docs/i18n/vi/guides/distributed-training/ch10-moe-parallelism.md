---
title: "Chương 10: Song song MoE"
description: "Cách song song hóa mô hình expert thưa: cơ chế Router, giao tiếp All-to-All trong Expert Parallelism, tổ hợp EP×DP×TP×PP và cân bằng tải"
tags: ["MoE", "Expert Parallelism", "All-to-All", "Cân bằng tải", "Mô hình thưa"]
sourceHash: "5f4162626e229486"
---

## Giới thiệu chương

MoE (Mixture of Experts) là kiến trúc phổ biến của các mô hình trăm tỷ/nghìn tỷ tham số hiện nay (DeepSeek-V3, Mixtral và các mô hình tương tự). Kích hoạt thưa giúp mở rộng số tham số mà không tăng đáng kể lượng tính toán. MoE cũng thêm một chiều song song mới: Expert Parallelism (EP), cùng kiểu giao tiếp All-to-All đặc trưng. Chương này đứng riêng vì kiểu giao tiếp và vấn đề cân bằng tải của nó khác bản chất so với song song trên mô hình dày đặc (dense).

---

## Các bài trong chương

### 1. Ôn lại cấu trúc mô hình MoE

- FFN dày đặc → MoE thưa: thay một FFN bằng $E$ Expert
- Router (Gating Network): chọn Top-K Expert cho mỗi token
- Kích hoạt thưa: số tham số $\times E$, nhưng lượng tính toán mỗi token chỉ tăng $K$ lần
- Capacity Factor và việc bỏ token (token dropping)

### 2. Expert Parallelism (EP)

- Ý tưởng cốt lõi: đặt các Expert khác nhau lên các GPU khác nhau; mỗi GPU chỉ lưu $\frac{E}{N}$ Expert
- Khác với TP: TP cắt một ma trận trọng số; EP quyết định Expert nào nằm ở đâu

### 3. Giao tiếp All-to-All (trọng tâm chương)

- Hai lần All-to-All: dispatch (gửi token tới GPU chứa Expert tương ứng) + combine (gửi kết quả về GPU gốc)
- Phân tích lượng giao tiếp: phụ thuộc số token, Top-K và cách phân bố Expert
- Vì sao All-to-All là nút thắt chính khi huấn luyện MoE
- Tối ưu giao tiếp: All-to-All theo nhóm, chồng tính toán lên giao tiếp, thư viện hiệu năng cao như DeepEP

### 4. Vấn đề cân bằng tải

- Router lệch: một số Expert bị chọn quá nhiều → một phần GPU quá tải, một phần nhàn rỗi
- Auxiliary Loss: mất mát phụ giúp Router phân phối token đều hơn
- Cân bằng tải không dùng aux loss của DeepSeek (aux-loss-free, điều chỉnh bias)
- Dung lượng Expert và chiến lược drop/pad

### 5. Tổ hợp EP với các kiểu song song khác

- EP × DP: chia trực giao nhóm EP và nhóm DP
- EP × TP: tiếp tục làm tensor parallelism bên trong Expert (Expert rất lớn)
- EP × PP: phân bổ lớp MoE và lớp dày đặc trong pipeline
- Ví dụ: phân tích cấu hình song song của DeepSeek-V3 / Mixtral

### 6. Thực hành

- Triển khai một lớp MoE tối thiểu + Top-2 Router, in phân bố định tuyến token
- Mô phỏng quá trình All-to-All dispatch/combine
- Quan sát mất cân bằng tải, thêm aux loss rồi so sánh

---

## Mục tiêu học tập

- Vẽ được luồng dữ liệu của lớp MoE: Router → dispatch → Expert → combine
- Giải thích được hai lần All-to-All của Expert Parallelism mỗi lần truyền gì
- Nêu được nguyên nhân mất cân bằng tải MoE và hai hướng xử lý: aux loss / aux-loss-free
- Phân tích được cách chia miền giao tiếp khi kết hợp EP với DP/TP/PP
