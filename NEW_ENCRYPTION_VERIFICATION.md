# Xác Thực Mã Hóa TLS/SSL trong Giao Tiếp Học Liên Hợp

## Tổng Quan

Tài liệu này trình bày phương pháp phân tích và xác minh bảo mật của kết nối TLS/SSL trong hệ thống Học Liên Hợp (Federated Learning) sử dụng framework Flower. Mục tiêu chính là kiểm chứng rằng dữ liệu truyền giữa client và server thực sự được mã hóa, đảm bảo tính bảo mật và toàn vẹn của các tham số mô hình trong quá trình truyền tải.

## Tầm Quan Trọng của Mã Hóa trong Học Liên Hợp

Trong Học Liên Hợp, việc bảo vệ các tham số mô hình trong quá trình truyền tải là vô cùng quan trọng vì:

1. **Bảo vệ sở hữu trí tuệ**: Các tham số mô hình có thể chứa thông tin giá trị về kiến trúc và kiến thức học máy
2. **Ngăn chặn tấn công đầu độc mô hình**: Ngăn chặn kẻ tấn công thay đổi tham số mô hình trong quá trình truyền
3. **Bảo vệ quyền riêng tư**: Mặc dù Federated Learning đã giúp dữ liệu gốc không bị chia sẻ, nhưng tham số mô hình vẫn có thể tiết lộ thông tin về dữ liệu huấn luyện

## Phương Pháp Phân Tích Bảo Mật

### Công cụ sử dụng

- **Wireshark**: Công cụ phân tích gói tin mạng để kiểm tra lưu lượng mạng giữa client và server
- **OpenSSL**: Công cụ để kiểm tra và xác minh cấu hình TLS/SSL
- **Framework Flower**: Hỗ trợ giao tiếp an toàn với TLS/SSL
- **PyTorch**: Thư viện học máy để huấn luyện mô hình MNIST đơn giản

### Các bước phân tích

1. **Thiết lập môi trường thử nghiệm**:
   - Cài đặt server Flower với TLS/SSL
   - Cấu hình client Flower để kết nối bảo mật
   - Chuẩn bị dữ liệu MNIST để huấn luyện

2. **Bắt gói tin mạng**:
   - Sử dụng Wireshark để bắt gói tin trên giao diện mạng
   - Lọc gói tin dựa trên cổng server (8443)
   - Bắt đầu quá trình huấn luyện mô hình

3. **Phân tích gói tin**:
   - Xác định các gói tin TLS Handshake
   - Kiểm tra các thông số bảo mật của phiên TLS
   - Xác minh mã hóa trong dữ liệu Application Data

4. **So sánh với kết nối không bảo mật**:
   - Thực hiện lại quá trình huấn luyện mà không sử dụng TLS/SSL
   - Bắt gói tin và phân tích dữ liệu không được mã hóa
   - So sánh sự khác biệt giữa hai trường hợp

## Hướng Dẫn Phân Tích Bảo Mật với Wireshark

### 1. Khởi động bắt gói tin

```bash
# Khởi động Wireshark
wireshark
```

Sau khi Wireshark mở, chọn interface mạng (thường là loopback cho kết nối local) và thiết lập bộ lọc:
```
tcp.port == 8443
```

### 2. Khởi động server và client Flower

Trong terminal thứ nhất:
```bash
python mnist_federated_learning.py
# Chọn 1 để chạy server
# Chọn Y để sử dụng TLS/SSL
```

Trong terminal thứ hai:
```bash
python mnist_federated_learning.py
# Chọn 2 để chạy client
# Chọn Y để sử dụng TLS/SSL
```

### 3. Phân tích gói tin TLS

Trong Wireshark, bạn sẽ thấy các gói tin sau:

#### 3.1. TLS Handshake

- **Client Hello**: Client gửi version TLS được hỗ trợ và cipher suites
- **Server Hello**: Server chọn version TLS và cipher suite
- **Certificate**: Server gửi chứng chỉ của mình
- **Server Key Exchange**: Server gửi tham số cho tạo khóa phiên
- **Certificate Request**: Server yêu cầu chứng chỉ từ client (trong mTLS)
- **Client Certificate**: Client gửi chứng chỉ của mình
- **Client Key Exchange**: Client gửi tham số cho tạo khóa phiên
- **Finished**: Hoàn thành quá trình bắt tay TLS

#### 3.2. Dữ liệu ứng dụng

- **Application Data**: Các gói tin này chứa dữ liệu gRPC đã được mã hóa
- Đây là nơi các tham số mô hình được truyền tải một cách an toàn

### 4. Xác minh mã hóa

Kiểm tra các gói tin Application Data để xác minh rằng chúng thực sự đã được mã hóa:

1. Chọn một gói tin Application Data trong Wireshark
2. Mở rộng phần TLS để xem chi tiết
3. Xác nhận rằng nội dung được mã hóa và không thể đọc được ở dạng văn bản thuần
4. Kiểm tra cipher suite được sử dụng (ví dụ: TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384)

### 5. So sánh với kết nối không bảo mật

Để so sánh, thực hiện lại quá trình huấn luyện mà không sử dụng TLS:

```bash
python mnist_federated_learning.py
# Chọn 1 để chạy server
# Chọn N để không sử dụng TLS/SSL
```

```bash
python mnist_federated_learning.py
# Chọn 2 để chạy client
# Chọn N để không sử dụng TLS/SSL
```

Trong trường hợp này, bạn sẽ thấy:
1. Không có quá trình bắt tay TLS
2. Dữ liệu gRPC được truyền ở dạng không mã hóa
3. Tham số mô hình có thể bị chặn và đọc bằng các công cụ phân tích gói tin

## Kết luận

Qua phân tích với Wireshark, có thể xác nhận rằng:

1. **Kết nối TLS/SSL hoạt động chính xác**: Quá trình bắt tay TLS diễn ra đúng quy trình
2. **Dữ liệu được mã hóa**: Các tham số mô hình được mã hóa trong quá trình truyền tải
3. **mTLS thiết lập thành công**: Cả server và client đều được xác thực
4. **Bảo mật đảm bảo**: Không thể đọc được nội dung dữ liệu truyền tải nếu không có khóa giải mã

Việc sử dụng TLS/SSL trong Flower đã cung cấp một lớp bảo mật mạnh mẽ, bảo vệ tham số mô hình và đảm bảo tính toàn vẹn của quá trình Học Liên Hợp.

## Tài liệu tham khảo

1. [Flower Framework Documentation](https://flower.dev/docs/)
2. [TLS 1.3 Specification (RFC 8446)](https://datatracker.ietf.org/doc/html/rfc8446)
3. [Wireshark User's Guide](https://www.wireshark.org/docs/wsug_html_chunked/)
4. [OpenSSL Documentation](https://www.openssl.org/docs/)
