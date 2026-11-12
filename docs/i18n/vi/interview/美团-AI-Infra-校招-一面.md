---
title: "Meituan AI Infra tuyển mới tốt nghiệp vòng một"
description: "Câu hỏi phỏng vấn tuyển mới tốt nghiệp AI Infra tại Meituan vòng một thật, gồm tối ưu toán tử"
tags: ["Tối ưu toán tử"]
sourceHash: "218e9beebd2e9399"
---

### Vòng một

#### Lý thuyết

1. Vì sao HashMap đưa red-black tree vào?
2. Vì red-black tree truy vấn nhanh hơn linked list, vì sao không dùng red-black tree từ đầu?
3. Thay red-black tree bằng BST hoặc AVL có được không? Ưu nhược mỗi cái?
4. Tham số cốt lõi của thread pool là gì?
5. Nếu đoạn code dùng synchronized nhiều khiến hiệu năng kém, tối ưu thế nào?
6. synchronized khác lock dựa trên AQS ở điểm nào? Chênh lệch hiệu năng thể hiện ở đâu?
7. Hệ production Full GC dày, lần lỗi và xử lý thế nào?
8. CMS và G1 phù hợp kịch bản nào? Có biết ZGC không?
9. Hiểu nguyên tắc khớp tiền tố trái nhất thế nào?
10. Lập joint index trên a, b, c; điều kiện `a > 1 AND c = 1` thì index được dùng ra sao?
11. Production gặp slow query, định vị và lần lỗi thế nào?

#### Đào sâu dự án

12. Giới thiệu chi tiết tối ưu hiệu năng trong kỳ thực tập, và cách ghi metadata toán tử không xâm nhập code nghiệp vụ.

#### Lập trình tại chỗ

13. Xóa nút trùng trong linked list đã sắp II.

### Vòng hai

#### Lý thuyết

1. Đoạn Java sau chạy ra gì? Xử lý vấn đề thế nào?
   ```java
   int a = 3;
   Integer b = null;
   int c = a + b;
   long d = 1L + 0x7fffffff;
   int e = a + (int)d;
   ```
2. `Integer a = 567; Integer b = 567; System.out.println(a == b);` in ra gì? Sửa thế nào cho khớp kỳ vọng?
3. Giới thiệu kiến thức concurrent programming Java bạn biết.
4. Java có những lock phổ biến nào? Khác nhau thế nào?
5. Thread pool hoạt động thế nào?
6. Từ lúc bấm nút nguồn tới mở trình duyệt vào link phỏng vấn, máy tính tầng dưới lần lượt trải qua những gì? (POST, boot sector, khởi động OS, tạo process/thread, render hiển thị, toàn chuỗi network request)
7. Vì sao họp video tầng dưới dùng UDP chứ không TCP?
8. Từ góc ứng dụng, giới thiệu kiến thức cốt lõi MySQL.
9. Làm sao biết câu SQL có đi index không? EXPLAIN thường nhìn field nào?
10. Mô tả thiết kế field và index của một bảng DB bạn từng dùng thật.
11. Thứ tự thực thi của SELECT, FROM, WHERE, GROUP BY, HAVING, v.v. thế nào?

#### Đào sâu dự án

12. Chọn một trải nghiệm thực tập trình bày chi tiết, nhấn điểm khó kỹ thuật và thách thức.

#### Lập trình tại chỗ

13. Trong ma trận 2D hàng đã sắp, cột đã sắp, tìm một giá trị đích.
