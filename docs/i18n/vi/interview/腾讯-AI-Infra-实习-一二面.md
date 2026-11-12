---
title: "Tencent AI Infra thực tập vòng một và hai"
description: "Câu hỏi phỏng vấn thực tập Tencent AI Infra vòng một và hai thật, gồm tối ưu toán tử, high-performance computing và các hướng liên quan"
tags: ["Tối ưu toán tử", "High-performance computing"]
sourceHash: "4dc559481f6b96b3"
---

### Vòng một

#### Kinh nghiệm dự án

1. Giới thiệu kinh nghiệm dự án, gồm điểm khó kỹ thuật và điểm nổi bật

#### Kiến thức nền

2. Trình bày khái niệm cơ bản của MPI và cách dùng
3. Nguyên lý cơ bản của lập trình đa luồng là gì
4. Các cơ chế lock phổ biến gồm những gì
5. Điều kiện gây deadlock trong hệ điều hành và cách xử lý
6. So sánh giống và khác giữa UDP và TCP
7. TCP dùng những cơ chế nào để đảm bảo truyền tin cậy

#### Bài lập trình

8. Cho một mảng chưa sắp xếp và một mảng đã sắp xếp, chèn phần tử từ mảng chưa sắp xếp vào mảng đã sắp xếp và giữ thứ tự, phân tích độ phức tạp thời gian
9. Thuật toán miệng: bài toán xử lý big data kinh điển

---

### Vòng hai

#### Kiến thức nền

1. Hiểu kiến trúc CPU và GPU đến mức nào
2. Có kinh nghiệm viết CUDA kernel không
3. Định nghĩa và đặc tính của kiểu POD trong C++ là gì
4. Ngoài priority queue, còn STL container nào trong C++ dùng để hiện thực heap

#### High-performance computing

5. Giải thích các khái niệm: shuffle, Hyper-Q, SM, slot
6. Công dụng và cách dùng NVIDIA Nsight
7. Ý nghĩa chỉ số SOL (Speed of Light)
8. Nguyên lý cơ bản của Polyhedral Model
9. Ý tưởng cốt lõi của framework biên dịch TVM
10. Khác biệt giữa Array of Struct và Struct of Array, cùng scenario phù hợp
11. Ý tưởng thiết kế ngôn ngữ lập trình Halide
12. Cách hiện thực reduce ở mức Warp

#### Bài lập trình

13. Median cửa sổ trượt (yêu cầu dùng max-heap và min-heap)
