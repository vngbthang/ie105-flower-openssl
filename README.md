# Triển khai và Đánh giá Giao tiếp An toàn dựa trên TLS/SSL cho Hệ thống Học Liên Hợp Flower sử dụng OpenSSL

Đồ án Nhập môn An toàn Thông tin - IE105

## Tổng quan

Đồ án này triển khai hệ thống Học Liên Hợp (Federated Learning) sử dụng framework Flower với giao tiếp an toàn giữa client và server thông qua TLS/SSL. Mục đích chính là thiết lập môi trường với mã hóa end-to-end và xác thực hai chiều (mTLS - mutual TLS) để đảm bảo tính bảo mật, toàn vẹn và xác thực trong quá trình truyền dữ liệu.

## Môi trường cài đặt

- Python 3.6+
- Flower (flwr) 1.5+
- OpenSSL
- Các thư viện Python liên quan

## Cài đặt

### 1. Cài đặt thư viện

```bash
pip install "flwr>=1.5.0"
```

### 2. Tạo chứng chỉ SSL với OpenSSL

Sử dụng script `generate_certs.sh` để tạo Certificate Authority (CA), chứng chỉ server và client:

```bash
chmod +x generate_certs.sh
./generate_certs.sh
```

Script này tạo ra:
- CA key và certificate
- Server key và certificate được ký bởi CA
- Client key và certificate được ký bởi CA

### 3. Khởi động Server

```bash
./start_server_superlink.sh
```

Hoặc sử dụng phương pháp thay thế:

```bash
python server/server.py
```

### 4. Khởi động Client

**Phương pháp được khuyến nghị (Flower v1.5+):**
```bash
./start_client_supernode.sh
```

Hoặc chạy client trực tiếp:
```bash
python client/client_supernode.py
```

**Phương pháp truyền thống (đã deprecated nhưng vẫn hoạt động):**
```bash
./start_client.sh
```

Hoặc:
```bash
python client/client_with_context.py
```

## Cấu trúc dự án

```
generate_certs.sh           # Script tạo chứng chỉ SSL
README.md                   # Tài liệu hướng dẫn
SECURITY_ANALYSIS.md        # Phân tích bảo mật
TLS_TECHNICAL_DETAILS.md    # Chi tiết kỹ thuật về TLS
PRESENTATION.md             # Slide trình bày
start_client.sh             # Script khởi động client (deprecated)
start_client_supernode.sh   # Script khởi động client với supernode (khuyến nghị)
start_server_superlink.sh   # Script khởi động server với superlink
benchmark_tls.sh            # Script đánh giá hiệu năng TLS
test_security.py            # Script kiểm tra bảo mật
certs/                      # Thư mục chứa chứng chỉ
├── ca/                     # Certificate Authority
│   ├── ca.key              # CA private key
│   ├── ca.pem              # CA certificate
│   └── ca.srl              # Serial number file
├── client/                 # Client certificates
│   ├── client.csr          # Certificate signing request
│   ├── client.key          # Private key
│   └── client.pem          # Certificate
└── server/                 # Server certificates
    ├── server.csr          # Certificate signing request
    ├── server.key          # Private key
    └── server.pem          # Certificate
client/                     # Client code
├── client_with_context.py  # Client sử dụng SSL context (deprecated)
├── client.py               # Client sử dụng root_certificates (deprecated)
└── client_supernode.py     # Client sử dụng flower-supernode (recommended)
server/                     # Server code
├── server_superlink.py     # Cấu hình server với superlink
└── server.py               # Server triển khai thông thường
```

## Cơ chế bảo mật

### 1. Mutual TLS (mTLS)

Hệ thống sử dụng mTLS để:
- Server xác thực client (yêu cầu client cung cấp chứng chỉ hợp lệ)
- Client xác thực server (ngăn man-in-the-middle attack)

### 2. Mã hóa dữ liệu

- Dữ liệu truyền tải giữa server và client được mã hóa bởi TLS
- Sử dụng OpenSSL để quản lý khóa và chứng chỉ

### 3. Certificate Authority (CA)

- CA tự tạo (self-signed) được sử dụng để ký chứng chỉ server và client
- Server và client tin cậy lẫn nhau thông qua sự tin tưởng chung vào CA này

## Phân tích an toàn

Hệ thống đảm bảo các tính chất bảo mật cơ bản:
- **Tính bảo mật (Confidentiality)**: Dữ liệu được mã hóa trong quá trình truyền tải
- **Tính toàn vẹn (Integrity)**: TLS đảm bảo dữ liệu không bị thay đổi trong quá trình truyền
- **Tính xác thực (Authentication)**: mTLS đảm bảo cả client và server đều được xác thực

## Benchmark và Testing

### Kiểm tra bảo mật TLS/SSL

Dự án cung cấp các script để kiểm tra tính bảo mật của cài đặt TLS/SSL:

```bash
# Script kiểm tra bảo mật TLS cơ bản
python test_security.py

# Script kiểm tra bảo mật cho kiến trúc SuperNode/SuperLink
python test_security_supernode.py
```

Các bài kiểm tra bảo mật bao gồm:
1. **Kết nối không có chứng chỉ**: Thử kết nối đến server mà không cung cấp bất kỳ chứng chỉ nào
2. **Kết nối với CA không hợp lệ**: Thử kết nối với một chứng chỉ CA giả mạo
3. **Kết nối SuperNode không có chứng chỉ**: Kiểm tra bảo mật cho kiến trúc SuperNode/SuperLink

Các bài kiểm tra này đảm bảo rằng hệ thống từ chối các kết nối không được xác thực đúng cách, bảo vệ chống lại các cuộc tấn công man-in-the-middle và nghe trộm.

### Phân tích kết quả kiểm tra bảo mật

Khi chạy `test_security.py` hoặc `test_security_supernode.py`, kết quả sẽ hiển thị ngay trên terminal. Một kết quả thành công sẽ hiển thị:

```
===== TÓM TẮT KẾT QUẢ KIỂM TRA =====
✓ Tất cả kiểm tra đều thành công! Cài đặt TLS/SSL của bạn an toàn.
Số kiểm tra thành công: 2/2
```

Đối với mỗi bài kiểm tra, nếu hệ thống hoạt động đúng:
- **Kiểm tra không chứng chỉ**: Sẽ thất bại với lỗi kết nối
- **Kiểm tra CA giả mạo**: Sẽ thất bại với lỗi xác thực chứng chỉ

Những lỗi phổ biến bạn có thể gặp:
- `SSL_ERROR_SSL: error:1000007d:SSL routines:OPENSSL_internal:CERTIFICATE_VERIFY_FAILED` - Chứng chỉ không được tin cậy
- `StatusCode.UNAVAILABLE` - Không thể thiết lập kết nối TLS

Nếu có bất kỳ bài kiểm tra nào thất bại, hãy kiểm tra:
1. Đường dẫn đến chứng chỉ CA
2. Quyền truy cập tệp chứng chỉ
3. Tính hợp lệ của chứng chỉ

### Benchmark hiệu năng

Để đánh giá ảnh hưởng của TLS/SSL đến hiệu năng, dự án cung cấp các script benchmark:

```bash
# Benchmark cho kiến trúc SuperNode/SuperLink
./benchmark_supernode.sh
```

Các script benchmark so sánh:
1. **Thời gian hoàn thành**: Đo thời gian để hoàn thành một vòng huấn luyện
2. **Overhead TLS**: So sánh hiệu năng giữa kết nối có TLS và không có TLS
3. **Thông lượng dữ liệu**: Đánh giá khả năng truyền tải dữ liệu với các cài đặt khác nhau

Kết quả benchmark được lưu trong các tệp log để phân tích chi tiết:
- `secure_superlink_log.txt` / `secure_supernode_log.txt`: Log khi sử dụng TLS/SSL
- `insecure_superlink_log.txt` / `insecure_supernode_log.txt`: Log khi không sử dụng TLS/SSL

### Ví dụ kết quả benchmark

Dưới đây là ví dụ kết quả khi chạy benchmark với kiến trúc SuperNode/SuperLink:

```
===== TLS/SSL Performance Benchmark for Flower (SuperNode/SuperLink) =====
This benchmark compares the performance of Flower SuperNode/SuperLink with and without TLS

--- Running benchmark ---
1. Testing with TLS/SSL enabled (secure)
Starting secure superlink server...
Starting secure supernode client...

real    0m2.543s
user    0m1.326s
sys     0m0.247s
Stopping secure superlink server...

2. Testing without TLS/SSL (insecure)
Creating insecure supernode/superlink configurations...
Starting insecure superlink server...
Starting insecure supernode client...

real    0m2.201s
user    0m1.182s
sys     0m0.188s
Stopping insecure superlink server...

--- Results Summary ---
Secure connection (SuperNode/SuperLink): SUCCESS
Insecure connection: SUCCESS

Check logs for detailed timings:
- secure_superlink_log.txt / secure_supernode_log.txt
- insecure_superlink_log.txt / insecure_supernode_log.txt
```

Trong ví dụ này, ta có thể thấy:
- Kết nối bảo mật với TLS mất khoảng 2.543 giây
- Kết nối không bảo mật mất khoảng 2.201 giây
- Overhead của TLS là khoảng 0.342 giây (~15.5%)

Thông tin chi tiết hơn có thể được tìm thấy trong tệp log. Ví dụ từ `secure_superlink_log.txt`:

```
INFO:      Starting Flower SuperLink
INFO:      Flower ECE: Starting Fleet API (gRPC-rere) on [::]:8443
INFO:      [Fleet.CreateNode] Request ping_interval=30.0
INFO:      [Fleet.CreateNode] Created node_id=5263325793613200899
INFO:      [Fleet.PullMessages] node_id=5263325793613200899
```

Overhead này là chi phí hợp lý cho việc đảm bảo bảo mật, và trong hầu hết các tình huống thực tế, ưu điểm về bảo mật của TLS/SSL vượt trội hơn đáng kể so với nhược điểm về hiệu năng.

### Cách đọc kết quả benchmark

Các tệp log chứa thông tin chi tiết về quá trình giao tiếp giữa client và server. Thông số quan trọng cần chú ý:

1. **Thời gian thiết lập kết nối**: Thời gian cần thiết để thiết lập kết nối TLS (bao gồm bắt tay TLS)
2. **Thời gian tổng thể**: Tổng thời gian để hoàn thành nhiệm vụ
3. **Số lượng thông điệp**: Số lượng thông điệp được trao đổi

Khi TLS được bật, bạn sẽ thấy overhead khoảng 5-15% so với kết nối không bảo mật, đây là chi phí cho việc mã hóa và xác thực dữ liệu.

## Kết luận

Đồ án này đã triển khai thành công một hệ thống Học Liên Hợp Flower với giao tiếp an toàn sử dụng TLS/SSL. Điều này giúp bảo vệ dữ liệu và mô hình trong quá trình truyền tải, đảm bảo tính riêng tư và an toàn thông tin trong hệ thống học máy phân tán.