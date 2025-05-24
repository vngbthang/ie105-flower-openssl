# Xác Thực Mã Hóa TLS/SSL trong Giao Tiếp Học Liên Hợp

Tài liệu này giải thích phương pháp và công cụ được sử dụng để xác minh rằng dữ liệu truyền giữa client và server Flower thực sự được mã hóa bằng TLS/SSL.

## Tổng Quan

Trong hệ thống Học Liên Hợp (Federated Learning), đảm bảo thông tin liên lạc giữa client và server được mã hóa an toàn là rất quan trọng để bảo vệ các tham số mô hình nhạy cảm và dữ liệu huấn luyện. Triển khai của chúng tôi sử dụng TLS/SSL với xác thực hai chiều (mTLS) để bảo mật giao tiếp này.

Quy trình xác thực này:
1. Kiểm tra mã hóa thực tế của dữ liệu trong quá trình huấn luyện mô hình
2. Ghi lại và phân tích các gói tin mạng trong quá trình giao tiếp an toàn
3. So sánh giao tiếp đã mã hóa và chưa mã hóa (tùy chọn)
4. Cung cấp bằng chứng rằng các tham số mô hình được bảo vệ đúng cách

## Tại Sao Việc Xác Thực Mã Hóa Quan Trọng

Học Liên Hợp liên quan đến việc huấn luyện các mô hình học máy trên nhiều thiết bị phi tập trung lưu trữ các mẫu dữ liệu cục bộ. Giao tiếp giữa các thiết bị này (client) và máy chủ trung tâm bao gồm:

- Tham số mô hình (trọng số và độ lệch)
- Độ dốc (gradients)
- Metadata huấn luyện

Nếu không có mã hóa phù hợp:
- Các tham số mô hình nhạy cảm có thể bị rò rỉ
- Kẻ tấn công có thể thực hiện đầu độc mô hình bằng cách chặn và sửa đổi tham số
- Quyền riêng tư của dữ liệu huấn luyện có thể bị xâm phạm thông qua tấn công đảo ngược mô hình
- Sở hữu trí tuệ liên quan đến kiến trúc và trọng số mô hình có thể bị đánh cắp

## Phương Pháp

### Thành Phần Kiểm Tra

Quy trình xác thực mã hóa của chúng tôi bao gồm một số thành phần:

1. **Khối Lượng ML Thực**: Sử dụng tập dữ liệu MNIST để huấn luyện một mô hình CNN đơn giản trong môi trường học liên hợp.
2. **Ghi Lại Gói Tin**: Sử dụng `tcpdump` để ghi lại lưu lượng mạng trong quá trình huấn luyện.
3. **Phân Tích Lưu Lượng**: Sử dụng `tshark` (Wireshark CLI) để phân tích các gói tin đã ghi lại.
4. **Trích Xuất Văn Bản Thuần**: Cố gắng tìm bất kỳ văn bản thuần liên quan đến mô hình trong lưu lượng mạng.
5. **So Sánh**: So sánh các mẫu giao tiếp an toàn (TLS/SSL) với không an toàn.

### Quy Trình Kiểm Tra

Quy trình xác thực tuân theo các bước sau:

1. Bắt đầu ghi lại gói tin với `tcpdump` trên giao diện loopback
2. Khởi chạy server Flower với TLS/SSL được bật
3. Kết nối client Flower với chứng chỉ phù hợp và huấn luyện mô hình
4. Phân tích lưu lượng đã ghi lại để tìm bằng chứng về mã hóa
5. Lặp lại với kết nối không an toàn (tùy chọn)
6. So sánh kết quả giữa lưu lượng an toàn và không an toàn

### Chi Tiết Mô Hình ML

Bài kiểm tra sử dụng kiến trúc Mạng Nơ-ron Tích Chập (CNN) đơn giản cho MNIST:

```python
class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
```

Mô hình này chứa các tham số quan trọng có thể bị lộ nếu được truyền không mã hóa:
- Trọng số lớp tích chập
- Trọng số lớp kết nối đầy đủ
- Độ lệch cho mỗi lớp

Các tham số được trao đổi trong quá trình Học Liên Hợp dưới dạng mảng NumPy đã tuần tự hóa, nên được bảo vệ bằng mã hóa TLS/SSL.

## Công Cụ Sử Dụng

- **tcpdump**: Công cụ phân tích gói tin mạng được sử dụng để ghi lại lưu lượng
- **tshark**: Giao diện dòng lệnh của Wireshark được sử dụng để phân tích gói tin
- **PyTorch**: Được sử dụng để tạo và huấn luyện mô hình ML
- **Flower**: Framework học liên hợp
- **OpenSSL**: Được sử dụng để tạo và quản lý chứng chỉ
- **strings**: Được sử dụng để cố gắng trích xuất văn bản thuần từ dữ liệu nhị phân

## Các Chỉ Số Bảo Mật

Các chỉ số sau đây được sử dụng để xác minh mã hóa đúng cách:

1. **Sự Hiện Diện của TLS Handshake**: Các gói tin phải hiển thị bằng chứng về thông điệp TLS handshake (content type 22)
2. **Dữ Liệu Ứng Dụng**: Dữ liệu ứng dụng đã mã hóa phải hiện diện (content type 23)
3. **Ẩn Header gRPC**: Trong giao tiếp an toàn, các header gRPC không được nhìn thấy dưới dạng văn bản thuần
4. **Không Có Tham Số Mô Hình Dạng Văn Bản Thuần**: Lưu lượng ghi lại không được chứa bất kỳ tham số mô hình nào có thể đọc được
5. **Chi Phí Mã Hóa**: Lưu lượng đã mã hóa phải lớn hơn một chút so với lưu lượng không mã hóa do chi phí TLS

## Chạy Quy Trình Xác Thực

Quy trình xác thực có thể được chạy bằng lệnh:

```bash
sudo ./run_encryption_test.sh
```

Quyền sudo là cần thiết để ghi lại gói tin. Script sẽ:
- Kiểm tra các phụ thuộc cần thiết
- Chạy bài kiểm tra mã hóa
- Phân tích và hiển thị kết quả
- Lưu kết quả chi tiết vào tệp nhật ký

Bài kiểm tra tự động tạo môi trường Học Liên Hợp thực sử dụng:
- Kiến trúc mô hình CNN đơn giản cho phân loại MNIST
- Tham số mô hình thực tế được trao đổi giữa client và server
- Mã hóa TLS/SSL thực tế với chứng chỉ trong thư mục `certs`

## Kết Quả Mong Đợi

Khi mã hóa hoạt động đúng cách, bạn sẽ thấy:

1. Lưu lượng SSL/TLS được xác nhận trong dữ liệu ghi lại
2. Không có tham số mô hình dạng văn bản thuần nào có thể trích xuất từ lưu lượng an toàn
3. Thông điệp TLS handshake xuất hiện trong lưu lượng an toàn (content type 22)
4. Dữ liệu ứng dụng đã mã hóa hiện diện (content type 23)
5. Thông tin phiên bản TLS (lý tưởng là TLS 1.2 hoặc 1.3)
6. Bộ mã hóa mạnh đang được sử dụng

Trong phần so sánh tùy chọn với lưu lượng không mã hóa, bạn nên thấy:
1. Header gRPC ở dạng văn bản thuần trong lưu lượng không an toàn nhưng không xuất hiện trong lưu lượng an toàn
2. Thông tin mô hình có thể trích xuất được từ lưu lượng không an toàn
3. Sự khác biệt về kích thước giữa lưu lượng đã mã hóa và chưa mã hóa (do chi phí mã hóa)
4. Mẫu có cấu trúc trong lưu lượng không mã hóa so với mẫu có vẻ ngẫu nhiên trong lưu lượng đã mã hóa

## Giải Thích Kết Quả

Bài kiểm tra tạo ra tệp nhật ký chi tiết (`encryption_test_results.log`) với các thông tin sau:

| Loại Kết Quả | Giải Thích |
|------------|----------------|
| TEST PASSED | Bằng chứng mạnh mẽ rằng dữ liệu được mã hóa đúng cách |
| PARTIAL TEST PASSED | Có bằng chứng về mã hóa, nhưng xác minh còn hạn chế |
| TEST FAILED | Bằng chứng cho thấy dữ liệu có thể không được mã hóa đúng cách |

### Các Chỉ Số Chính

1. **Thông Điệp SSL/TLS Handshake**: Chỉ báo quá trình đàm phán TLS đúng giữa client và server.
2. **Thông Điệp Dữ Liệu Ứng Dụng SSL/TLS**: Đây là dữ liệu trao đổi đã mã hóa thực tế.
3. **Phiên Bản TLS**: Phiên bản cao hơn (1.2, 1.3) cung cấp bảo mật tốt hơn phiên bản cũ.
4. **Trích Xuất Văn Bản Thuần**: Cố gắng tìm chuỗi có thể đọc được liên quan đến tham số mô hình.
5. **Header gRPC**: Không được nhìn thấy dưới dạng văn bản thuần trong giao tiếp an toàn.

## Xử Lý Sự Cố

Nếu bài kiểm tra không phát hiện mã hóa:

1. Đảm bảo chứng chỉ được cấu hình đúng và có thể truy cập
2. Xác minh server đang sử dụng chứng chỉ TLS đúng cách
3. Kiểm tra client được cấu hình đúng để xác thực chứng chỉ của server
4. Kiểm tra kết quả để biết thông báo lỗi cụ thể
5. Thử chạy Wireshark GUI để phân tích gói tin chi tiết hơn

## Sơ Đồ Quy Trình Xác Thực Mã Hóa

```
+------------------------------------------------------+
|                 XÁC THỰC MÃ HÓA                      |
+------------------------------------------------------+

+---------------+   Kết nối TLS  +---------------+
|               |<-------------->|               |
|    Client     |    An toàn     |    Server     |
|   (Mô hình)   |               |  (Tổng hợp)   |
+---------------+               +---------------+
      ^                               ^
      |                               |
+---------------+               +---------------+
|   tcpdump     |-------------->|   Phân tích   |
| Thu gói tin   |   Lưu lượng   | (tshark/grep) |
+---------------+    đã thu     +---------------+
                                      |
                                      v
                              +---------------+
                              |   Kết quả     |
                              |   Xác thực    |
                              +---------------+

+-----------------------------------------------------------------------+
|  LƯỢNG AN TOÀN                      |  LƯỢNG KHÔNG AN TOÀN (TÙY CHỌN) |
|---------------------------------------|-------------------------------|
|  - TLS Handshake Messages             |  - Không TLS Handshake        |
|  - Dữ liệu ứng dụng đã mã hóa        |  - Dữ liệu ứng dụng văn bản   |
|  - Không thấy tham số mô hình        |  - Nhìn thấy tham số mô hình   |
|  - gRPC Headers bị ẩn                |  - gRPC Headers hiện rõ        |
+-----------------------------------------------------------------------+
```

## Kết Luận

Quy trình xác thực này cung cấp bằng chứng thực nghiệm rằng hệ thống Học Liên Hợp của chúng tôi đang mã hóa đúng cách việc giao tiếp giữa client và server. Bằng cách thu thập và phân tích lưu lượng mạng thực tế trong quá trình huấn luyện mô hình, chúng tôi có thể xác nhận rằng các tham số mô hình nhạy cảm được bảo vệ khỏi bị nghe trộm.

Sự kết hợp của TLS/SSL với xác thực hai chiều (mTLS) đảm bảo:
1. **Bảo Mật**: Tham số mô hình được mã hóa trong quá trình truyền
2. **Toàn Vẹn**: Bất kỳ sự giả mạo nào với dữ liệu truyền đều sẽ bị phát hiện
3. **Xác Thực**: Cả client và server đều xác minh danh tính của nhau
4. **Không Thể Phủ Nhận**: Người tham gia không thể phủ nhận sự tham gia của họ trong giao tiếp

Khung kiểm tra này có thể được mở rộng cho các phân tích bảo mật phức tạp hơn, chẳng hạn như kiểm tra khả năng chống chịu của mã hóa đối với các cuộc tấn công khác nhau hoặc kiểm tra với các bộ mã hóa khác nhau.
