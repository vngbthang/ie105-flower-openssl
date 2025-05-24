# Phân tích Bảo mật TLS/SSL trong Hệ thống Học Liên Hợp Flower

Đồ án Nhập môn An toàn Thông tin - IE105

## Tổng quan

Dự án này triển khai một hệ thống Học Liên Hợp (Federated Learning) sử dụng framework Flower với tập dữ liệu MNIST và phân tích khía cạnh bảo mật của giao tiếp giữa client và server thông qua TLS/SSL. Mục đích chính là tìm hiểu về cách mà mã hóa TLS/SSL bảo vệ các tham số mô hình trong quá trình truyền dữ liệu, với trọng tâm vào phần học máy và xác thực hai chiều (mTLS).

## Mục Tiêu Dự Án

1. **Triển khai hệ thống ML phân tán:** Xây dựng mô hình CNN đơn giản được huấn luyện trong môi trường phân tán với framework Flower
2. **Phân tích bảo mật giao tiếp:** Kiểm tra và đánh giá việc sử dụng TLS/SSL để bảo mật giao tiếp giữa client và server
3. **Hiểu rõ về mTLS:** Triển khai xác thực hai chiều (mutual TLS) để đảm bảo cả client và server đều được xác thực

## Cấu trúc Dự Án

Dự án bao gồm hai phần chính:

1. **Hệ thống Học Liên Hợp:**
   - Mô hình CNN đơn giản cho phân loại MNIST
   - Client và server sử dụng framework Flower
   - Hỗ trợ giao tiếp TLS/SSL bảo mật

2. **Phân Tích Bảo Mật:**
   - Tạo và quản lý chứng chỉ SSL/TLS với OpenSSL
   - Phân tích giao tiếp mạng bảo mật với Wireshark
   - Đánh giá hiệu quả của mTLS trong bảo vệ tham số mô hình

## Môi Trường Cài Đặt

- Python 3.8+
- Flower (flwr) 1.5+
- PyTorch và torchvision
- OpenSSL
- Wireshark (cho việc phân tích gói tin)

## Cài Đặt

### 1. Cài đặt thư viện

```bash
pip install flwr torch torchvision numpy
```

### 2. Tạo chứng chỉ SSL với OpenSSL

```bash
chmod +x generate_certs.sh
./generate_certs.sh
```

### 3. Chạy Mô Hình MNIST

```bash
python mnist_federated_learning.py
```

Chọn 1 để chạy server và 2 để chạy client trong các terminal khác nhau. Chọn có hoặc không sử dụng TLS/SSL.

### 4. Phân Tích Bảo Mật

Sử dụng Wireshark để bắt và phân tích gói tin trong quá trình huấn luyện:
1. Mở Wireshark và chọn interface (thường là loopback cho kết nối local)
2. Thiết lập bộ lọc: `tcp.port == 8443`
3. Bắt đầu bắt gói tin và chạy quá trình huấn luyện mô hình
4. Phân tích các gói tin đã bắt được để kiểm tra bảo mật TLS/SSL

## Mô hình Học Máy

Dự án sử dụng một mô hình CNN đơn giản cho phân loại hình ảnh MNIST:
- Lớp tích chập đầu tiên với 32 filter
- Lớp tích chập thứ hai với 64 filter
- Lớp max pooling
- Lớp fully-connected với 128 neuron
- Lớp đầu ra với 10 neuron (tương ứng với 10 chữ số)

Xem chi tiết về mô hình học máy tại [ML_README.md](ML_README.md).

## Triển khai TLS/SSL trong Flower

Flower cho phép bảo mật kết nối gRPC giữa client và server thông qua TLS/SSL:

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
        root_certificate # CA certificate for client verification
    ),
    config={"num_rounds": 3},
)
```

### Client:

```python
# Đọc chứng chỉ CA
with open("certs/ca/ca.pem", "rb") as f:
    ca_cert = f.read()

# Khởi động client với chứng chỉ CA
fl.client.start_client(
    server_address="localhost:8443",
    client=client,
    root_certificates=ca_cert
)
```

## Chi tiết Kỹ thuật

Để biết thêm chi tiết kỹ thuật về TLS/SSL trong dự án, xem [TLS_TECHNICAL_DETAILS.md](TLS_TECHNICAL_DETAILS.md).

## Phân tích An toàn

Để biết thêm thông tin về các lỗ hổng tiềm ẩn và biện pháp giảm thiểu, xem [SECURITY_ANALYSIS.md](SECURITY_ANALYSIS.md).

## Kết luận

Dự án này đã triển khai thành công một hệ thống Học Liên Hợp sử dụng Flower với giao tiếp an toàn thông qua TLS/SSL. Việc phân tích bảo mật cho thấy tầm quan trọng của mã hóa và xác thực trong việc bảo vệ tham số mô hình và dữ liệu trong hệ thống học máy phân tán.
