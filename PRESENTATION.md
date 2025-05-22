# Triển khai và Đánh giá Giao tiếp An toàn dựa trên TLS/SSL cho Hệ thống Học Li### Client (Recommended Method):

```python
# Sử dụng trực tiếp trong mã Python
import flwr as fl
import os

# Đọc chứng chỉ CA
with open("certs/ca/ca.pem", "rb") as f:
    ca_cert = f.read()

# Khởi động client với chứng chỉ CA
fl.client.start_client(
    server_address="localhost:8443",
    client=client,
    root_certificates=ca_cert
)
```ower sử dụng OpenSSL

## Slide 1: Giới thiệu

- **Đề tài**: Triển khai và Đánh giá Giao tiếp An toàn dựa trên TLS/SSL cho Hệ thống Học Liên Hợp Flower sử dụng OpenSSL
- **Môn học**: Nhập môn An toàn Thông tin - IE105
- **Mục tiêu**: Triển khai giao tiếp bảo mật giữa server và client trong hệ thống học liên hợp Flower

## Slide 2: Khái niệm cơ bản

- **Flower**: Framework cho học liên hợp (Federated Learning)
- **TLS/SSL**: Transport Layer Security / Secure Sockets Layer
- **mTLS**: Mutual TLS (xác thực hai chiều)
- **OpenSSL**: Thư viện mã nguồn mở để làm việc với TLS/SSL

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

- **Certificate Authority (CA)**: Tự tạo (self-signed) để ký chứng chỉ
- **Server Certificate**: Ký bởi CA, xác thực server với client
- **Client Certificate**: Ký bởi CA, xác thực client với server
- **Cấu trúc chứng chỉ**: X.509 format

## Slide 5: Triển khai TLS/SSL trong Flower

### Server (Traditional Method):

```python
# Read certificate files
with open(server_cert_file, 'rb') as f:
    server_cert = f.read()
with open(server_key_file, 'rb') as f:
    server_key = f.read()
with open(ca_cert_file, 'rb') as f:
    ca_cert = f.read()

# Pass certificates to server
certificates = (server_cert, server_key, ca_cert)
fl.server.start_server(..., certificates=certificates)
```

### Server (Recommended Method):

```bash
flower-superlink \
    --ssl-certfile=certs/server/server.pem \
    --ssl-keyfile=certs/server/server.key \
    --ssl-ca-certfile=certs/ca/ca.pem \
    --fleet-api-address=[::]:8443
```

### Client (Traditional Method):

```python
# Read CA certificate
with open(ca_cert_file, 'rb') as f:
    ca_cert = f.read()

# Connect to server with CA certificate
fl.client.start_numpy_client(..., root_certificates=ca_cert)
```

### Client (Recommended Method):

```bash
flower-supernode \
    --superlink='localhost:8443' \
    --ssl-ca-certfile=certs/./benchmark_tls.shca/ca.pem \
    --ssl-certfile=certs/client/client.pem \
    --ssl-keyfile=certs/client/client.key
```

## Slide 6: Quá trình xác thực mTLS

1. **Bước 1**: Client kết nối đến server
2. **Bước 2**: Server gửi certificate cho client
3. **Bước 3**: Client xác thực certificate của server bằng CA certificate
4. **Bước 4**: Server yêu cầu client certificate
5. **Bước 5**: Client gửi certificate cho server
6. **Bước 6**: Server xác thực certificate của client bằng CA certificate
7. **Bước 7**: Cả hai bên trao đổi khóa phiên
8. **Bước 8**: Dữ liệu được mã hóa bảo mật trong quá trình truyền tải

## Slide 7: Các tính chất bảo mật đạt được

- **Tính bảo mật (Confidentiality)**: Dữ liệu được mã hóa end-to-end
- **Tính toàn vẹn (Integrity)**: Phát hiện và ngăn chặn sửa đổi dữ liệu
- **Tính xác thực (Authentication)**: Xác thực hai chiều server-client
- **Không chối bỏ (Non-repudiation)**: Giao dịch được xác thực bằng chứng chỉ
- **Chống Man-in-the-Middle**: Bảo vệ chống lại tấn công trung gian

## Slide 8: Kiểm thử an toàn

- **Test không chứng chỉ**: Client không thể kết nối nếu không cung cấp chứng chỉ
- **Test chứng chỉ không hợp lệ**: Client không thể kết nối với chứng chỉ không được CA ký
- **Test hết hạn chứng chỉ**: Chứng chỉ hết hạn không được chấp nhận
- **Test revoked certificate**: Chứng chỉ đã thu hồi không được chấp nhận

## Slide 9: Đánh giá hiệu năng

| Metric                | Không TLS    | Có TLS       | Chênh lệch    |
|-----------------------|--------------|--------------|---------------|
| Thời gian thiết lập   | Nhanh hơn    | Chậm hơn     | +10-15%       |
| Thông lượng dữ liệu   | Cao hơn      | Thấp hơn     | -5-10%        |
| CPU usage             | Thấp hơn     | Cao hơn      | +10-20%       |
| Độ trễ mạng           | Thấp hơn     | Cao hơn      | +5-15%        |

## Slide 10: Lỗ hổng và biện pháp giảm thiểu

- **Quản lý khóa riêng**: Phân quyền hợp lý, sử dụng HSM
- **Self-signed CA**: Trong môi trường sản xuất, nên dùng CA tin cậy
- **Downgrade attack**: Chỉ hỗ trợ TLS 1.2+, vô hiệu hóa ciphersuites yếu
- **Side-channel attack**: Sử dụng thư viện mã hóa an toàn, cập nhật thường xuyên

## Slide 11: Kết luận

- Đã triển khai thành công TLS/SSL trong hệ thống Flower
- Đảm bảo giao tiếp an toàn giữa server và client
- Thực hiện xác thực hai chiều (mTLS)
- Cân bằng giữa an toàn và hiệu năng

## Slide 12: Hướng phát triển

- Tích hợp với PKI thực tế
- Triển khai certificate rotation và revocation
- Tối ưu hóa hiệu năng TLS
- Áp dụng bảo mật nhiều lớp kết hợp với TLS
