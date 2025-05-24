# Triển khai và Phân tích Mã hóa TLS/SSL trong Hệ thống Học Liên Hợp Flower

## Slide 1: Giới thiệu

- **Đề tài**: Triển khai và Phân tích Mã hóa TLS/SSL trong Hệ thống Học Liên Hợp Flower
- **Môn học**: Nhập môn An toàn Thông tin - IE105
- **Mục tiêu**: Tìm hiểu và phân tích bảo mật trong hệ thống học máy phân tán sử dụng TLS/SSL

## Slide 2: Khái niệm cơ bản

- **Federated Learning (Học Liên Hợp)**: Mô hình học máy phân tán không cần chia sẻ dữ liệu trực tiếp
- **Flower**: Framework cho Học Liên Hợp
- **TLS/SSL**: Transport Layer Security / Secure Sockets Layer
- **mTLS**: Mutual TLS (xác thực hai chiều)

## Slide 3: Kiến trúc hệ thống

```
+---------------+        Encrypted channel         +---------------+
|               |<------------------------------>  |               |
|  Flower       |  - Authenticated with certs      |  Flower       |
|  Server       |  - Encrypted with TLS            |  Client       |
|               |  - Integrity protected           |               |
+---------------+                                  +---------------+
                \                                 /
                 \                               /
                  v                             v
           +------------------------------------+
           |         Certificate Authority      |
           |         (Trust Anchor)            |
           +------------------------------------+
```

## Slide 4: Cơ sở hạ tầng khóa công khai (PKI)

- **Certificate Authority (CA)**: Tự tạo, ký chứng chỉ cho server và client
- **Server Certificate**: Xác thực server với client
- **Client Certificate**: Xác thực client với server (mTLS)
- **Public/Private Key**: Cặp khóa cho mã hóa bất đối xứng

## Slide 5: Mô hình học máy

- **Mô hình**: CNN đơn giản cho phân loại MNIST
  - 2 lớp tích chập (32/64 filters)
  - Lớp pooling và các lớp fully-connected
  - Đầu ra 10 classes (chữ số 0-9)
- **Dữ liệu**: MNIST dataset (60K training, 10K test)
- **Huấn luyện**: Sử dụng Federated Learning với Flower

## Slide 6: Cài đặt TLS/SSL trong Flower

### Server:
```python
# Đọc chứng chỉ và khóa
cert_chain = open("certs/server/server.pem", "rb").read()
private_key = open("certs/server/server.key", "rb").read()
root_certificate = open("certs/ca/ca.pem", "rb").read()

# Khởi tạo server với TLS
server = fl.server.start_server(
    server_address="0.0.0.0:8443",
    certificates=(
        cert_chain,     # Certificate chain
        private_key,    # Private key
        root_certificate # CA certificate
    ),
    config={"num_rounds": 3},
)
```

### Client:
```python
# Đọc chứng chỉ CA
with open("certs/ca/ca.pem", "rb") as f:
    ca_cert = f.read()

# Khởi động client với TLS
fl.client.start_client(
    server_address="localhost:8443",
    client=client,
    root_certificates=ca_cert
)
```

## Slide 7: Phân tích bảo mật TLS/SSL

- **Bắt tay TLS**: Quá trình trao đổi chứng chỉ và thiết lập khóa phiên
- **Xác thực mTLS**: Server và Client xác thực lẫn nhau
- **Mã hóa dữ liệu**: Bảo vệ tham số mô hình trong quá trình truyền
- **Bảo vệ khỏi MITM**: Ngăn chặn tấn công Man-in-the-Middle

## Slide 8: Phương pháp phân tích mã hóa

1. **Cài đặt môi trường**: Server và client Flower với TLS/SSL
2. **Bắt gói tin**: Sử dụng Wireshark để bắt gói tin trong quá trình huấn luyện
3. **Phân tích bắt tay TLS**: Xác minh quá trình thiết lập kết nối bảo mật
4. **Kiểm tra dữ liệu**: Xác minh dữ liệu được truyền đi đã mã hóa
5. **So sánh**: Đối chiếu với kết nối không bảo mật

## Slide 9: Kết quả phân tích trong Wireshark

- **TLS Handshake**: Client Hello, Server Hello, Certificate, Key Exchange...
- **Cipher Suite**: Bộ mã hóa sử dụng (ví dụ: TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384)
- **Application Data**: Gói tin đã được mã hóa, không thể đọc nội dung trực tiếp
- **Độ bảo mật**: Xác minh rằng dữ liệu thực sự được mã hóa và bảo vệ

## Slide 10: Lỗ hổng bảo mật tiềm ẩn và giảm thiểu

1. **Quản lý Private Key**: Cần bảo vệ khóa riêng tư
   - Giải pháp: Cấp quyền phù hợp, HSM, xoay khóa định kỳ

2. **Self-signed CA**: Không có xác thực bên thứ ba
   - Giải pháp: Trong môi trường sản xuất, sử dụng CA tin cậy

3. **Downgrade Attack**: Hạ cấp TLS version hoặc cipher suite
   - Giải pháp: Chỉ cho phép TLS 1.2+, tắt cipher suite yếu

4. **Phân tích băng thông**: Suy ra thông tin từ pattern giao tiếp
   - Giải pháp: Padding, thêm nhiễu

## Slide 11: Tích hợp với bảo mật học máy

- **Differential Privacy**: Thêm nhiễu vào dữ liệu để bảo vệ quyền riêng tư
- **Secure Aggregation**: Tổng hợp tham số mô hình mà không tiết lộ giá trị từng client
- **Robust Aggregation**: Phát hiện và loại bỏ cập nhật độc hại
- **TLS/SSL**: Bảo vệ trong quá trình truyền tải

## Slide 12: Kết luận

- TLS/SSL cung cấp lớp bảo mật quan trọng cho hệ thống Học Liên Hợp
- mTLS đảm bảo xác thực hai chiều
- Dữ liệu truyền tải (tham số mô hình) được bảo vệ khỏi nghe trộm và thay đổi
- Kết hợp TLS/SSL với các kỹ thuật bảo mật học máy tạo ra một hệ thống toàn diện

## Slide 13: Triển khai thực tế

- **Cài đặt**: `pip install flwr torch torchvision`
- **Tạo chứng chỉ**: `./generate_certs.sh`
- **Chạy server**: `python mnist_federated_learning.py` (chọn 1)
- **Chạy client**: `python mnist_federated_learning.py` (chọn 2)
- **Phân tích**: Sử dụng Wireshark trên port 8443

## Slide 14: Câu hỏi và thảo luận

- Vai trò của TLS/SSL trong bảo vệ mô hình học máy
- So sánh hiệu năng giữa kết nối có và không có TLS
- Các cải tiến có thể thực hiện trong tương lai
- Giới hạn của hệ thống hiện tại

## Slide 15: Tham khảo

1. Flower Framework Documentation: [flower.dev](https://flower.dev/docs/)
2. TLS 1.3 Specification: RFC 8446
3. Federated Learning: A Survey on Enabling Technologies, Protocols, and Applications
4. TLS in Practice: Cryptanalysis of TLS Implementation
