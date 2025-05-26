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
   - Sử dụng các script tự động hóa: `run_easy.sh`, `start_server_superlink.sh`, `start_client_supernode.sh`

2. **Phân Tích Bảo Mật:**
   - Tạo và quản lý chứng chỉ SSL/TLS với OpenSSL
   - Phân tích giao tiếp mạng bảo mật với Wireshark
   - Đánh giá hiệu quả của mTLS trong bảo vệ tham số mô hình

## Môi Trường Cài Đặt

- Python 3.8+
- Flower (flwr) >=1.18.0
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

### 3. Chạy hệ thống với script tự động

#### 3.1. Chạy với `run_easy.sh` (Khuyến nghị)

Script này cung cấp giao diện menu đơn giản để chạy Federated Learning và phân tích bảo mật:

```bash
chmod +x run_easy.sh
./run_easy.sh
```
- Chọn các tùy chọn để chạy server, client, mô phỏng, sửa chứng chỉ, hoặc bắt gói tin Wireshark.
- Đảm bảo chọn đúng chế độ bảo mật (SSL/TLS) để kiểm tra bảo mật thực tế.

#### 3.2. Chạy thủ công từng thành phần (nâng cao)

- **Server:**
  ```bash
  ./start_server_superlink.sh 18443
  ```
- **Client:**
  ```bash
  ./start_client_supernode.sh localhost 18443 0
  ```

## Phân tích bảo mật với Wireshark

- Có thể chọn trực tiếp trong menu của `run_easy.sh` (tùy chọn 8) để tự động mở Wireshark với bộ lọc phù hợp.
- Xem hướng dẫn chi tiết trong file [`wireshark_analysis.md`](wireshark_analysis.md).

## Kết luận

- Hệ thống đảm bảo bảo mật khi sử dụng TLS/SSL (OpenSSL) cho giao tiếp federated learning.
- Có thể kiểm tra, phân tích và xác thực bảo mật bằng Wireshark.
- Đảm bảo sử dụng đúng script để kiểm tra bảo mật thực tế, không chỉ simulation mode.

## Tài liệu tham khảo

1. **Flower Framework**
   - [Flower - A Friendly Federated Learning Framework](https://flower.dev/)
   - [Flower Documentation](https://flower.dev/docs/)
   - [Flower GitHub Repository](https://github.com/adap/flower)

2. **TLS/SSL và OpenSSL**
   - [OpenSSL Official Documentation](https://www.openssl.org/docs/)
   - [TLS 1.3 Specification - RFC 8446](https://datatracker.ietf.org/doc/html/rfc8446)
   - [mTLS (Mutual TLS) Explained](https://www.cloudflare.com/learning/access-management/what-is-mutual-tls/)

3. **Federated Learning**
   - McMahan, H. B., et al. (2017). "Communication-efficient learning of deep networks from decentralized data." AISTATS.
   - [Federated Learning: Collaborative Machine Learning without Centralized Training Data](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
   - Yang, Q., et al. (2019). "Federated Learning: Challenges, Methods, and Future Directions." IEEE Signal Processing Magazine.

4. **Wireshark và Phân tích bảo mật**
   - [Wireshark User's Guide](https://www.wireshark.org/docs/wsug_html/)
   - [Analyzing TLS with Wireshark](https://wiki.wireshark.org/TLS)

5. **MNIST Dataset**
   - LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition." Proceedings of the IEEE.
   - [MNIST Database - Yann LeCun](http://yann.lecun.com/exdb/mnist/)
