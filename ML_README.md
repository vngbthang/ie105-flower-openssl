# Học Liên Hợp (Federated Learning) với MNIST và Framework Flower

## Giới thiệu

Dự án này triển khai hệ thống Học Liên Hợp (Federated Learning) sử dụng framework Flower với bộ dữ liệu MNIST. Hệ thống bao gồm một mô hình phân loại chữ số viết tay đơn giản được huấn luyện trong môi trường phân tán, trong đó dữ liệu được giữ cục bộ tại các client và chỉ chia sẻ tham số mô hình với server trung tâm. Hệ thống hỗ trợ bảo mật giao tiếp bằng TLS/SSL (OpenSSL).

## Cách sử dụng (Khuyến nghị)

### 1. Chạy script tự động

```bash
chmod +x run_easy.sh
./run_easy.sh
```
- Chọn các tùy chọn để chạy server, client, mô phỏng, sửa chứng chỉ, hoặc bắt gói tin Wireshark.
- Đảm bảo chọn đúng chế độ bảo mật (SSL/TLS) để kiểm tra bảo mật thực tế.

### 2. Chạy thủ công từng thành phần

- **Server:**
  ```bash
  ./start_server_superlink.sh 18443
  ```
- **Client:**
  ```bash
  ./start_client_supernode.sh localhost 18443 0
  ```

### 3. Phân tích bảo mật

- Có thể chọn bắt gói tin Wireshark trực tiếp trong menu của `run_easy.sh` (tùy chọn 8)
- Xem hướng dẫn chi tiết trong file [`wireshark_analysis.md`](wireshark_analysis.md)

## Mô hình ML được sử dụng

Mô hình là một mạng nơ-ron tích chập (CNN) đơn giản cho nhiệm vụ phân loại chữ số viết tay MNIST:
- Lớp tích chập đầu tiên với 32 filter
- Lớp tích chập thứ hai với 64 filter
- Lớp max pooling
- Lớp fully-connected với 128 neuron
- Lớp đầu ra với 10 neuron (tương ứng với 10 chữ số 0-9)

## Bộ dữ liệu

Bộ dữ liệu MNIST được sử dụng trong dự án này là bộ dữ liệu tiêu chuẩn và phổ biến cho bài toán nhận dạng chữ số viết tay. Bộ dữ liệu này bao gồm:
- 60,000 ảnh huấn luyện
- 10,000 ảnh kiểm tra
- Mỗi ảnh có kích thước 28x28 pixel, grayscale
- 10 lớp (chữ số từ 0-9)

Bộ dữ liệu được tải xuống tự động thông qua thư viện torchvision khi chạy script, được lưu trong thư mục `./data/MNIST`.

## Cài đặt thư viện cần thiết

```bash
pip install flwr torch torchvision numpy
```

## Cấu trúc của mã nguồn ML

1. **Định nghĩa mô hình (`MnistNet`)**:
   - CNN đơn giản cho phân loại MNIST
   - Các lớp tích chập, fully connected và hàm kích hoạt

2. **Client Học Liên Hợp (`MnistClient`)**:
   - Quản lý dữ liệu huấn luyện và kiểm tra cục bộ
   - Thực hiện các hàm `fit` và `evaluate` theo yêu cầu của server
   - Trả về tham số mô hình sau khi huấn luyện cục bộ

3. **Hàm tiện ích**:
   - `set_parameters`: Cập nhật tham số mô hình từ mảng NumPy
   - `train`: Huấn luyện mô hình với dữ liệu cục bộ
   - `test`: Đánh giá mô hình với dữ liệu kiểm tra
   - `load_data`: Tải bộ dữ liệu MNIST

4. **Chức năng của Server và Client**:
   - `run_server`: Khởi động Flower server
   - `run_client`: Khởi động Flower client với mô hình MNIST
   - Hỗ trợ TLS/SSL cho bảo mật trong truyền thông

## Lưu ý

- Để kiểm tra bảo mật thực tế, luôn chạy server và client riêng biệt với TLS/SSL
- Simulation mode chỉ dùng để kiểm thử logic, không tạo ra lưu lượng mạng thực

## Tài liệu tham khảo

1. **Flower Framework**
   - [Flower Documentation](https://flower.dev/docs/)
   - [Flower GitHub Repository](https://github.com/adap/flower)
   - [Flower API Reference](https://flower.dev/docs/apiref.html)

2. **PyTorch và Deep Learning**
   - [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
   - [PyTorch Tutorials](https://pytorch.org/tutorials/)
   - Goodfellow, I., et al. (2016). "Deep Learning." MIT Press.

3. **Federated Learning**
   - McMahan, H. B., et al. (2017). "Communication-efficient learning of deep networks from decentralized data." AISTATS.
   - Li, T., et al. (2020). "Federated Learning: Challenges, Methods, and Future Directions." IEEE Signal Processing Magazine.
   - [Google's Federated Learning: Collaborative Machine Learning without Centralized Training Data](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)

4. **MNIST Dataset**
   - LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition." Proceedings of the IEEE.
   - [MNIST Database - Yann LeCun](http://yann.lecun.com/exdb/mnist/)

5. **Convolutional Neural Networks**
   - LeCun, Y., et al. (2015). "Deep learning." Nature.
   - Zhang, W., et al. (2018). "A comprehensive survey on cross-modal retrieval." arXiv preprint.
