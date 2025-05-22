# TLS/SSL trong Flower: Chi tiết Kỹ thuật

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

CA trong đồ án này là self-signed CA được tạo bởi OpenSSL:

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

# Tạo CSR
openssl req -new -key certs/client/client.key -out certs/client/client.csr -subj "/CN=client"

# Ký CSR bởi CA
openssl x509 -req -in certs/client/client.csr -CA certs/ca/ca.pem -CAkey certs/ca/ca.key -CAcreateserial -out certs/client/client.pem -days 3650 -sha256
```

## 3. Thiết lập TLS trong Flower

### 3.1. Flower Server

Server cấu hình TLS theo hai cách:

#### 3.1.1. Sử dụng start_server API:

```python
import flwr as fl
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server_cert_file = os.path.join(base_dir, "certs/server/server.pem")
    server_key_file = os.path.join(base_dir, "certs/server/server.key")
    ca_cert_file = os.path.join(base_dir, "certs/ca/ca.pem")
    
    with open(server_cert_file, 'rb') as f:
        server_cert = f.read()
    with open(server_key_file, 'rb') as f:
        server_key = f.read()
    with open(ca_cert_file, 'rb') as f:
        ca_cert = f.read()
    
    certificates = (server_cert, server_key, ca_cert)
    
    fl.server.start_server(
        server_address="[::]:8443",
        config=fl.server.ServerConfig(num_rounds=1),
        certificates=certificates,
    )
```

#### 3.1.2. Sử dụng flower-superlink CLI:

```bash
flower-superlink \
    --ssl-certfile=certs/server/server.pem \
    --ssl-keyfile=certs/server/server.key \
    --ssl-ca-certfile=certs/ca/ca.pem \
    --fleet-api-address=[::]:8443
```

### 3.2. Flower Client

Client cấu hình TLS theo ba cách:

#### 3.2.1. Sử dụng triển khai trực tiếp (khuyến nghị, từ Flower v1.5+):

```python
# Trong file client_supernode.py
import flwr as fl
from flwr.client import NumPyClient
import os

class DummyClient(NumPyClient):
    def get_parameters(self, config): return []
    def fit(self, parameters, config): return [], 0, {}
    def evaluate(self, parameters, config): return 0.0, 0, {}

# Client chính để sử dụng với flower-supernode
if __name__ == "__main__":
    # Lấy đường dẫn tuyệt đối đến chứng chỉ CA
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ca_cert_path = os.path.join(base_dir, "certs/ca/ca.pem")
    
    # Đọc chứng chỉ CA
    with open(ca_cert_path, "rb") as f:
        ca_cert = f.read()
    
    # Khởi động client với chứng chỉ CA
    fl.client.start_client(
        server_address="localhost:8443",
        client=DummyClient(),
        root_certificates=ca_cert
    )
```

Chạy bằng lệnh:

```bash
python client/client_supernode.py
```

#### 3.2.2. Sử dụng root_certificates (deprecated):

```python
import flwr as fl
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    with open(os.path.join(base_dir, "certs/ca/ca.pem"), 'rb') as f:
        ca_cert = f.read()
    
    fl.client.start_numpy_client(
        server_address="localhost:8443",
        client=DummyClient(),
        root_certificates=ca_cert
    )
```

#### 3.2.3. Sử dụng SSL Context (deprecated):

```python
import flwr as fl
import ssl
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    context = ssl.create_default_context(
        ssl.Purpose.SERVER_AUTH,
        cafile=os.path.join(base_dir, "certs/ca/ca.pem")
    )
    context.load_cert_chain(
        certfile=os.path.join(base_dir, "certs/client/client.pem"),
        keyfile=os.path.join(base_dir, "certs/client/client.key")
    )
```

## 4. Cơ chế xác thực TLS

### 4.1. Xác thực Server (Server Authentication)

1. Client kết nối đến server
2. Server gửi certificate của mình cho client
3. Client kiểm tra chữ ký của certificate sử dụng CA certificate
4. Nếu hợp lệ, client tin tưởng server và thiết lập kết nối

### 4.2. Xác thực Client (Client Authentication - mTLS)

1. Sau khi server đã được xác thực, server yêu cầu client certificate
2. Client gửi certificate của mình cho server
3. Server kiểm tra chữ ký của certificate sử dụng CA certificate
4. Nếu hợp lệ, server chấp nhận kết nối từ client

### 4.3. Thiết lập khóa phiên (Session Key)

1. Client và server thực hiện trao đổi khóa để tạo khóa phiên an toàn
2. Khóa phiên được sử dụng để mã hóa/giải mã dữ liệu trong suốt phiên làm việc

## 5. Ưu điểm của hệ thống

1. **Tính bảo mật**: Dữ liệu được mã hóa end-to-end
2. **Tính toàn vẹn**: Bất kỳ thay đổi nào đối với dữ liệu trong quá trình truyền tải đều được phát hiện
3. **Tính xác thực hai chiều**: Cả server và client đều được xác thực
4. **Không cần tin tưởng mạng**: Kẻ tấn công không thể nghe lén hoặc thay đổi dữ liệu kể cả khi kiểm soát mạng

## 6. Hạn chế của hệ thống

1. **Quản lý chứng chỉ**: Cần có cơ chế quản lý chứng chỉ một cách an toàn
2. **Chi phí tính toán**: TLS/SSL tăng chi phí tính toán và độ trễ
3. **CA tự ký**: Trong môi trường thực tế, nên sử dụng CA được tin cậy thay vì CA tự ký

## 7. So sánh với không sử dụng TLS

| Khía cạnh               | Có TLS                     | Không có TLS                  |
|-------------------------|----------------------------|-------------------------------|
| Bảo mật dữ liệu         | Được mã hóa                | Không được mã hóa             |
| Xác thực                | Xác thực hai chiều         | Không có xác thực             |
| Man-in-the-middle       | Được bảo vệ                | Dễ bị tấn công                |
| Chi phí tính toán       | Cao hơn                    | Thấp hơn                      |
| Độ trễ mạng             | Cao hơn                    | Thấp hơn                      |
| Cấu hình                | Phức tạp hơn               | Đơn giản hơn                  |

## 8. Kiến trúc SuperNode/SuperLink và Phương pháp Gốc

Flower nâng cấp kiến trúc từ phiên bản 1.5 với việc đưa vào các khái niệm mới: SuperNode và SuperLink. Phần này sẽ so sánh phương pháp gốc và phương pháp mới.

### 8.1. Phương pháp gốc (Client/Server)

Kiến trúc ban đầu của Flower sử dụng mô hình Client/Server truyền thống:

```
+---------------+                               +---------------+
|               |                               |               |
|  Flower       |<-----------------------------→|  Flower       |
|  Server       |       Kết nối gRPC trực tiếp  |  Client       |
|               |                               |               |
+---------------+                               +---------------+
```

**Đặc điểm:**
- Server điều phối toàn bộ quá trình huấn luyện
- Client kết nối trực tiếp đến server
- API `start_server()` và `start_client()` được sử dụng
- Cấu hình TLS được truyền trực tiếp qua các tham số API

**Triển khai với TLS:**
```python
# Server
fl.server.start_server(
    server_address="[::]:8443",
    config=fl.server.ServerConfig(num_rounds=1),
    certificates=(server_cert, server_key, ca_cert),
)

# Client
fl.client.start_client(
    server_address="localhost:8443",
    client=client,
    root_certificates=ca_cert
)
```

### 8.2. Kiến trúc SuperNode/SuperLink

Kiến trúc mới được giới thiệu từ Flower 1.5+ cung cấp nhiều lợi ích hơn:

```
+---------------+                           +---------------+
|               |                           |               |
|  SuperLink    |<------------------------→ |  SuperNode    |
|  (Server)     |    Kết nối bảo mật        |  (Client)     |
|               |                           |               |
+---------------+                           +---------------+
       ↑                                            ↑
       |                                            |
       |  Quản lý                                   |  Cung cấp  
       |  chiến lược                                |  tài nguyên
       |  huấn luyện                                |  tính toán
       ↓                                            ↓
+---------------+                           +---------------+
|  Driver       |                           |  Python       |
|  Code         |                           |  Client       |
|  (Chiến lược) |                           |  Code         |
+---------------+                           +---------------+
```

**Đặc điểm:**
- SuperLink thay thế cho Server truyền thống, hoạt động như một điều phối viên
- SuperNode thay thế cho Client truyền thống, cung cấp tài nguyên tính toán
- Hỗ trợ cấu hình qua dòng lệnh (CLI) thay vì API, thuận tiện hơn cho quản lý
- Cách ly mã ứng dụng khỏi cơ sở hạ tầng (Driver code không cần biết chi tiết về kết nối)
- Hỗ trợ xác thực nâng cao và khả năng mở rộng tốt hơn

**Triển khai với TLS:**
```bash
# SuperLink (Server)
flower-superlink \
    --ssl-certfile=certs/server/server.pem \
    --ssl-keyfile=certs/server/server.key \
    --ssl-ca-certfile=certs/ca/ca.pem \
    --fleet-api-address=[::]:8443

# SuperNode (Client)
flower-supernode \
    --superlink='localhost:8443' \
    --root-certificates='certs/ca/ca.pem'
```

### 8.3. So sánh giữa hai phương pháp

| Khía cạnh               | Phương pháp gốc (Client/Server) | Kiến trúc SuperNode/SuperLink       |
|-------------------------|--------------------------------|-------------------------------------|
| API                     | start_server(), start_client() | CLI: flower-superlink, flower-supernode |
| Cấu hình TLS           | Thông qua tham số API          | Thông qua tham số dòng lệnh         |
| Tính module hóa        | Thấp                           | Cao                                 |
| Khả năng mở rộng       | Hạn chế                        | Tốt hơn                             |
| Bảo trì                | Phức tạp hơn                   | Đơn giản hơn                        |
| Tách biệt mối quan tâm  | Ít                             | Nhiều                               |
| Phù hợp với             | Thử nghiệm đơn giản            | Triển khai quy mô lớn               |
| Cập nhật codebase       | Cần sửa mã nguồn               | Chỉ thay đổi cấu hình dòng lệnh     |

### 8.4. Lưu ý khi chuyển đổi giữa hai phương pháp

1. **API thay đổi**:
   - Từ `start_server()` sang `flower-superlink`
   - Từ `start_client()` sang `flower-supernode`

2. **Cấu hình TLS**:
   - Phương pháp gốc: `certificates=(server_cert, server_key, ca_cert)` và `root_certificates=ca_cert`
   - Kiến trúc mới: `--ssl-certfile`, `--ssl-keyfile`, `--ssl-ca-certfile` và `--root-certificates`

3. **Phát triển Client**:
   - Phương pháp gốc: Trực tiếp khởi động client trong mã
   - Kiến trúc mới: Định nghĩa hàm `get_client_fn()` để SuperNode gọi khi cần

4. **Đường dẫn chứng chỉ**:
   - Phương pháp gốc: Thường đọc chứng chỉ vào biến
   - Kiến trúc mới: Cung cấp đường dẫn trực tiếp đến tệp chứng chỉ

Việc sử dụng kiến trúc SuperNode/SuperLink được khuyến nghị cho các dự án mới và quy mô lớn vì tính module hóa và dễ bảo trì.
