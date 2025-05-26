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

#### 3.1. Chạy với `run_easy.sh` (Cách đơn giản nhất)

Script này cung cấp giao diện menu đơn giản để chạy Federated Learning:

```bash
chmod +x run_easy.sh  # Cấp quyền thực thi nếu chưa có
./run_easy.sh
```

Chọn các tùy chọn từ menu:
- 1: Chạy server bảo mật (SSL/TLS) 
- 2: Chạy server không bảo mật
- 3: Chạy client bảo mật
- 4: Chạy client không bảo mật
- 5: Chạy chế độ mô phỏng

**Xem hướng dẫn chi tiết**: [RUNNING_GUIDE.md](RUNNING_GUIDE.md)

#### 3.2. Chạy với `mnist_federated_learning.py` (Cách cũ)

```bash
python mnist_federated_learning.py
```

Chọn 1 để chạy server và 2 để chạy client trong các terminal khác nhau. Chọn có hoặc không sử dụng TLS/SSL.

#### 3.3. Chạy với `run_mnist_flower_datasets.sh` (Cách trực tiếp với nhiều tùy chọn)

Script này sử dụng `flwr-datasets` để tải và quản lý dữ liệu MNIST, đồng thời cung cấp nhiều tùy chọn hơn để chạy federated learning.

**Cấp quyền thực thi cho script (nếu chưa có):**

```bash
chmod +x run_mnist_flower_datasets.sh
```

**Chạy script với các tùy chọn mặc định (chế độ `direct`, có TLS/SSL, 3 rounds):**

```bash
./run_mnist_flower_datasets.sh
```

**Các tùy chọn có sẵn:**

Bạn có thể xem tất cả các tùy chọn bằng cách chạy:

```bash
./run_mnist_flower_datasets.sh --help
```

Dưới đây là một số ví dụ:

- **Chạy ở chế độ `direct` không bảo mật (không TLS/SSL):**
  ```bash
  ./run_mnist_flower_datasets.sh --insecure
  ```

- **Chạy server riêng biệt với 5 rounds:**
  ```bash
  ./run_mnist_flower_datasets.sh --mode server --rounds 5
  ```

- **Chạy client riêng biệt (client ID 0) kết nối đến server (yêu cầu server đang chạy):**
  ```bash
  ./run_mnist_flower_datasets.sh --mode client --client-id 0
  ```
  (Lưu ý: `--client-id` chỉ có tác dụng khi `--mode client`)

- **Chạy mô phỏng (simulation) với 2 rounds:**
  ```bash
  ./run_mnist_flower_datasets.sh --mode simulation --rounds 2
  ```

- **Chạy với thời gian chờ tối đa là 10 phút (600 giây):**
  ```bash
  ./run_mnist_flower_datasets.sh --timeout 600
  ```

### 4. Phân Tích Bảo Mật với Wireshark

Sử dụng Wireshark để bắt và phân tích gói tin trong quá trình huấn luyện:

#### 4.1. Thiết lập Wireshark
1. Mở Wireshark với quyền root: `sudo wireshark`
2. Chọn interface (thường là `lo` cho kết nối local hoặc `eth0`/`wlan0` cho kết nối mạng)
3. Bắt đầu bắt gói tin bằng cách nhấn vào biểu tượng "Start Capturing Packets"

#### 4.2. Thiết lập bộ lọc:
- **Kết nối bảo mật**: `tcp.port == 18443 && tls`
- **Kết nối không bảo mật**: `tcp.port == 18080`

#### 4.3. Phân tích gói tin:
- Với kết nối bảo mật, quan sát quá trình bắt tay TLS và các gói "Application Data" được mã hóa
- Với kết nối không bảo mật, xem dữ liệu gRPC được truyền

#### 4.4. So sánh hiệu suất:
- Sử dụng "Statistics" > "I/O Graph" để so sánh lưu lượng
- Sử dụng "Statistics" > "TCP Stream Graphs" > "Round Trip Time" để so sánh độ trễ

#### 4.5. Hướng dẫn chi tiết:
Xem hướng dẫn phân tích đầy đủ trong [RUNNING_GUIDE.md](RUNNING_GUIDE.md)

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
