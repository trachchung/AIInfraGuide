---
title: "Chương 3: Optimizer"
description: "Hiểu logic tiến hóa và thành phần trạng thái nội bộ của các optimizer phổ biến, nắm cách phân tích chi phí bộ nhớ của optimizer, làm nền cho tối ưu bộ nhớ ZeRO và huấn luyện mixed precision"
tags: ["Optimizer", "Adam", "AdamW", "LAMB", "Phân tích bộ nhớ", "Mixed precision"]
sourceHash: "258c0e689a8b163e"
---

## Giới thiệu chương

Optimizer là "bánh lái" của huấn luyện—tính xong gradient chỉ giải được "đi hướng nào"; "đi thế nào, đi bao xa" là việc của nó. Chương này không sa vào suy diễn toán, chỉ tập trung hai việc kỹ sư AI Infra cần quan tâm nhất: **optimizer lưu gì bên trong** (trực tiếp quyết định bộ nhớ chiếm dụng), và **trong cảnh phân tán nó có yêu cầu đặc biệt gì** (hội tụ với Batch lớn). Hai điểm này là tiền đề trực tiếp để hiểu tối ưu bộ nhớ ZeRO và huấn luyện mixed precision phía sau.

---

## Cấu trúc chương

### 3.1 Nguyên lý optimizer và phân tích chi phí bộ nhớ

Một bài giảng hết nội dung chương. Đường chính:

- **Vị trí của optimizer trong vòng huấn luyện**: ba bước sau `optimizer.step()`, và vì sao "trạng thái nội bộ trú ngụ lâu dài trên bộ nhớ GPU" là gốc của chi phí bộ nhớ
- **Đường tiến hóa (SGD → Momentum → Adam → AdamW)**: mỗi bước thêm bao nhiêu bản trạng thái, giải bài toán gì, vì sao AdamW trở thành chuẩn huấn luyện mô hình lớn
- **Phân tích chi phí bộ nhớ (trọng tâm)**: đối chiếu số byte mỗi tham số, sổ bộ nhớ đầy đủ của mô hình 7B dưới AdamW + mixed precision, suy ra trạng thái optimizer chiếm khoảng 75\%, và vì sao phải dùng trọng số chính FP32
- **Optimizer cho Batch lớn**: khó khăn hội tụ khi song song dữ liệu phóng to Batch, tư tưởng scale thích ứng theo lớp của LARS/LAMB và khi nào nên dùng
- **Thực hành**: Dùng PyTorch đếm số phần tử trong `state` của optimizer, đo thực tế để xác nhận "Adam lưu 2 bản mỗi tham số, SGD lưu 0 bản"

---

---

## Mục tiêu học tập

- Nói rõ mỗi bước SGD → Momentum → Adam → AdamW giải bài toán gì và lưu thêm trạng thái gì
- Tính tay được bộ nhớ trạng thái optimizer của mô hình bất kỳ quy mô dưới AdamW + mixed precision, và giải thích vì sao nó chiếm khoảng 75\% bộ nhớ tĩnh
- Giải thích vì sao trong huấn luyện mixed precision optimizer vẫn phải dùng trọng số chính FP32
- Trả lời được "Batch Size phóng to 10 lần thì chỉnh learning rate thế nào", và biết LAMB/LARS chỉ cần thiết trong những cảnh nào
