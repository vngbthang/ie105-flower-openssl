# Chi tiết Kỹ thuật về TLS/SSL trong Flower

## 1. Kiến trúc TLS trong Flower

Flower sử dụng gRPC làm giao thức giao tiếp giữa server và client. gRPC hỗ trợ TLS/SSL để bảo mật kết nối, và Flower mở rộng khả năng này để hỗ trợ mTLS (mutual TLS).

```
+---------------+        Encrypted channel         +---------------+
|               |<------------------------------>  |               |
|  Flower       |  - Authenticated with certs      |  Flower       |
|  Server       |  - Encrypted with TLS            |  Client       |
|               |  - Integrity protected           |               |
+---------------+                                  +---------------+
```

## 2. Thiết lập PKI (Public Key Infrastructure)

### 2.1. Certificate Authority (CA)

CA trong dự án này là self-signed CA được tạo bởi OpenSSL:

```bash
# Tạo CA key
openssl genrsa -out certs/ca/ca.key 4096

# Tạo CA certificate
openssl req -x509 -new -nodes -key certs/ca/ca.key -sha256 -days 3650 -out certs/ca/ca.pem -subj "/CN=Flower CA"
```

CA này được sử dụng để ký chứng chỉ cho cả server và client.

### 2.2. Server Certificate

Server certificate được ký bởi CA:

```bash
# Tạo server key
openssl genrsa -out certs/server/server.key 4096

# Tạo Certificate Signing Request (CSR)
openssl req -new -key certs/server/server.key -out certs/server/server.csr -subj "/CN=localhost"

# Ký CSR bởi CA
openssl x509 -req -in certs/server/server.csr -CA certs/ca/ca.pem -CAkey certs/ca/ca.key -CAcreateserial -out certs/server/server.pem -days 3650 -sha256
```

### 2.3. Client Certificate

Client certificate cũng được ký bởi CA:

```bash
# Tạo client key
openssl genrsa -out certs/client/client.key 4096

# Tạo Certificate Signing Request (CSR)
openssl req -new -key certs/client/client.key -out certs/client/client.csr -subj "/CN=flower-client"

# Ký CSR bởi CA
openssl x509 -req -in certs/client/client.csr -CA certs/ca/ca.pem -CAkey certs/ca/ca.key -CAcreateserial -out certs/client/client.pem -days 3650 -sha256
```

## 3. Cài đặt TLS trong Flower

### 3.1. Server-side TLS

```python
def run_server(use_secure=False):
    """Khởi động Flower server."""
    if use_secure:
        # Đọc chứng chỉ và khóa
        cert_chain = open(CERT_DIR / "server" / "server.pem", "rb").read()
        private_key = open(CERT_DIR / "server" / "server.key", "rb").read()
        root_certificate = open(CERT_DIR / "ca" / "ca.pem", "rb").read()

        # Khởi tạo server có bật TLS
        server = fl.server.start_server(
            server_address=f"0.0.0.0:{SERVER_PORT}",
            config=fl.server.ServerConfig(num_rounds=3),
            certificates=(cert_chain, private_key, root_certificate),
        )
    else:
        # Khởi tạo server không sử dụng TLS
        server = fl.server.start_server(
            server_address=f"0.0.0.0:{SERVER_PORT}",
            config=fl.server.ServerConfig(num_rounds=3),
        )
```

### 3.2. Client-side TLS

```python
def run_client(use_secure=False):
    """Khởi động Flower client."""
    model = MnistNet().to(DEVICE)
    trainloader, testloader = load_data()
    
    if use_secure:
        # Đọc chứng chỉ CA
        with open(CERT_DIR / "ca" / "ca.pem", "rb") as f:
            ca_cert = f.read()

        # Khởi động client với TLS
        fl.client.start_client(
            server_address=f"127.0.0.1:{SERVER_PORT}",
            client=MnistClient(model, trainloader, testloader),
            root_certificates=ca_cert,
        )
    else:
        # Khởi động client không sử dụng TLS
        fl.client.start_client(
            server_address=f"127.0.0.1:{SERVER_PORT}",
            client=MnistClient(model, trainloader, testloader),
        )
```

## 4. Bắt tay TLS (TLS Handshake)

Bắt tay TLS xảy ra khi client kết nối đến server và thiết lập kênh giao tiếp an toàn:

1. **ClientHello**: Client gửi version TLS hỗ trợ, random number, và cipher suites hỗ trợ
2. **ServerHello**: Server chọn version TLS và cipher suite, gửi random number
3. **Certificate**: Server gửi chứng chỉ của mình
4. **ServerKeyExchange**: Nếu cần, server gửi thông tin bổ sung
5. **CertificateRequest**: Server yêu cầu client gửi chứng chỉ (trong mTLS)
6. **ServerHelloDone**: Server kết thúc phần của mình
7. **Certificate**: Client gửi chứng chỉ của mình (trong mTLS)
8. **ClientKeyExchange**: Client gửi pre-master secret đã mã hóa bằng public key của server
9. **CertificateVerify**: Client chứng minh sở hữu private key tương ứng với chứng chỉ
10. **ChangeCipherSpec**: Cả hai bên chuyển sang sử dụng khóa đã thỏa thuận
11. **Finished**: Cả hai bên xác nhận bắt tay thành công

## 5. Kiểm tra chứng chỉ trong mTLS

Trong mTLS, server và client đều kiểm tra chứng chỉ của đối tác:

1. **Kiểm tra chữ ký**: Xác minh chứng chỉ được ký bởi CA tin cậy
2. **Kiểm tra thời hạn**: Đảm bảo chứng chỉ chưa hết hạn
3. **Kiểm tra trạng thái**: Đảm bảo chứng chỉ không bị thu hồi (trong triển khai cao cấp hơn)
4. **Kiểm tra tên**: Đảm bảo tên trong chứng chỉ phù hợp với host kết nối

## 6. Bộ mã hóa (Cipher Suites)

Flower sử dụng các bộ mã hóa mạnh được hỗ trợ bởi OpenSSL. Ví dụ về bộ mã hóa thông dụng:

- `TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384`
  - Key Exchange: ECDHE (Elliptic Curve Diffie-Hellman Ephemeral)
  - Authentication: RSA
  - Bulk Encryption: AES 256 bit in GCM mode
  - Message Authentication: SHA384

## 7. Các ưu điểm của TLS trong Flower

- **Bảo mật**: Dữ liệu được mã hóa trong quá trình truyền
- **Tính toàn vẹn**: Phát hiện bất kỳ thay đổi nào đối với dữ liệu truyền
- **Xác thực**: Đảm bảo danh tính của các bên tham gia
- **Ngăn chặn tấn công Man-in-the-Middle**: Đảm bảo kẻ tấn công không thể đánh chặn hoặc thay đổi thông tin

## 8. Phân tích hiệu năng

Việc sử dụng TLS có một số tác động đến hiệu năng:

1. **Chi phí bắt tay TLS**: Phát sinh khi thiết lập kết nối
2. **Chi phí mã hóa/giải mã**: Ảnh hưởng đến thông lượng dữ liệu
3. **Overhead giao thức**: TLS thêm các header và trailer vào dữ liệu

Tuy nhiên, với tài nguyên phần cứng hiện đại, những chi phí này thường không đáng kể so với lợi ích bảo mật mà TLS mang lại.

## 9. Thực hành tốt nhất

1. **Sử dụng TLS 1.2 hoặc cao hơn**: Tránh các phiên bản cũ với lỗ hổng đã biết
2. **Sử dụng các bộ mã hóa mạnh**: Ưu tiên ECDHE cho key exchange và AES-GCM cho mã hóa
3. **Bảo vệ khóa riêng tư**: Đảm bảo khóa riêng tư không bị lộ
4. **Xoay khóa định kỳ**: Tạo lại key và chứng chỉ định kỳ để tăng bảo mật
5. **Kiểm tra thu hồi chứng chỉ**: Triển khai CRL hoặc OCSP trong môi trường sản xuất

## 10. Tóm tắt

TLS/SSL trong Flower cung cấp một lớp bảo mật quan trọng cho việc truyền tải tham số mô hình và dữ liệu học máy. Việc triển khai mTLS đảm bảo cả client và server đều được xác thực, giúp bảo vệ chống lại các cuộc tấn công man-in-the-middle và đảm bảo tính riêng tư trong quá trình học liên hợp.
