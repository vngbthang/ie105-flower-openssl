# Hướng dẫn chạy Học Liên Hợp với TLS/SSL

## Giới thiệu
Tài liệu này hướng dẫn cách chạy hệ thống Học Liên Hợp (Federated Learning) sử dụng MNIST và framework Flower với bảo mật TLS/SSL.

## Vấn đề SSL trong phiên bản cũ của Flower

Trong phiên bản mới của Flower, phương thức `fl.server.start_server()` đã bị deprecated và được thay thế bằng `flower-superlink` CLI. Điều này ảnh hưởng đến cách thiết lập kết nối bảo mật TLS/SSL.

## Cách chạy hệ thống

### 1. Tạo chứng chỉ SSL

Đầu tiên, đảm bảo chứng chỉ SSL được tạo đúng cách:

```bash
bash generate_certs.sh
```

### 2. Chạy server

Có hai cách để chạy server:

**Cách 1**: Sử dụng script shell (khuyến nghị)
```bash
./run_mnist_server.sh
```

**Cách 2**: Sử dụng Python script
```bash
python mnist_federated_learning.py --server --ssl
```

hoặc chạy tương tác:
```bash
python mnist_federated_learning.py
# Chọn 1 cho server và y cho TLS/SSL
```

### 3. Chạy client

Tương tự, có hai cách để chạy client:

**Cách 1**: Sử dụng script shell
```bash
./run_mnist_client.sh
```

**Cách 2**: Sử dụng Python script
```bash
python mnist_federated_learning.py --client --ssl
```

hoặc chạy tương tác:
```bash
python mnist_federated_learning.py
# Chọn 2 cho client và y cho TLS/SSL
```

## Khắc phục lỗi

Nếu gặp lỗi liên quan đến chứng chỉ SSL:

1. Đảm bảo chứng chỉ đã được tạo đúng cách
   ```bash
   bash generate_certs.sh
   ```

2. Kiểm tra quyền truy cập của các file chứng chỉ
   ```bash
   chmod 600 certs/server/server.key certs/ca/ca.key
   chmod 644 certs/server/server.pem certs/ca/ca.pem
   ```

3. Đảm bảo đã cài đặt Flower với phiên bản mới nhất
   ```bash
   pip install -U flwr
   ```

4. Nếu vẫn gặp lỗi, có thể chạy không bảo mật cho mục đích thử nghiệm
   ```bash
   ./run_mnist_server.sh --insecure  # cho server
   python mnist_federated_learning.py --client --no-ssl  # cho client
   ```

## Phân tích giao tiếp mạng

Để phân tích gói tin mạng và xác minh mã hóa TLS/SSL, hãy sử dụng Wireshark:

1. Mở Wireshark
2. Lọc gói tin theo cổng: `tcp.port == 8443`
3. Bắt đầu bắt gói tin
4. Chạy server và client
5. Phân tích gói tin TLS Handshake và Application Data

Chi tiết hơn về phân tích mạng có thể được tìm thấy trong `ENCRYPTION_VERIFICATION.md`.
