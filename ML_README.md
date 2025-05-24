# Học Liên Hợp (Federated Learning) với MNIST và Framework Flower

## Giới thiệu

Dự án này triển khai hệ thống Học Liên Hợp (Federated Learning) sử dụng framework Flower với bộ dữ liệu MNIST. Hệ thống bao gồm một mô hình phân loại chữ số viết tay đơn giản được huấn luyện trong môi trường phân tán, trong đó dữ liệu được giữ cục bộ tại các client và chỉ chia sẻ tham số mô hình với server trung tâm.

## Mô hình ML được sử dụng

Mô hình được sử dụng là một mạng nơ-ron tích chập (CNN) đơn giản cho nhiệm vụ phân loại chữ số viết tay MNIST:

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

## Cách sử dụng

### 1. Chạy server

```bash
python mnist_federated_learning.py
```

Khi được hỏi, chọn "1" để chạy server, và "y" hoặc "n" để sử dụng hoặc không sử dụng TLS/SSL.

### 2. Chạy client

Mở một terminal khác và chạy:

```bash
python mnist_federated_learning.py
```

Khi được hỏi, chọn "2" để chạy client, và chọn cùng một tùy chọn TLS/SSL như server.

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

## Luồng dữ liệu trong Học Liên Hợp

1. Server khởi tạo mô hình toàn cục và bắt đầu quá trình Federated Learning.
2. Các client nhận tham số mô hình từ server.
3. Mỗi client huấn luyện mô hình trên dữ liệu cục bộ của mình.
4. Các client chỉ gửi các cập nhật tham số (không phải dữ liệu) về server.
5. Server tổng hợp các cập nhật từ tất cả các client để cải thiện mô hình toàn cục.
6. Quy trình lặp lại cho đến khi đạt được số vòng huấn luyện đã định.

## Lợi ích của việc sử dụng Học Liên Hợp

1. **Bảo mật dữ liệu**: Dữ liệu vẫn được lưu giữ tại client, chỉ có tham số mô hình được chia sẻ.
2. **Hiệu quả về băng thông**: Truyền tải các tham số mô hình chiếm ít băng thông hơn so với việc truyền toàn bộ dữ liệu.
3. **Tính riêng tư**: Phù hợp với các quy định về bảo vệ dữ liệu như GDPR.
4. **Khả năng mở rộng**: Có thể dễ dàng mở rộng đến nhiều client phân tán trên mạng.

## Mở rộng

Dự án này có thể được mở rộng theo nhiều cách:
- Thêm nhiều client để mô phỏng môi trường thực tế hơn
- Sử dụng các bộ dữ liệu phức tạp hơn (CIFAR-10, ImageNet)
- Triển khai các chiến lược tổng hợp khác nhau (FedAvg, FedProx, etc.)
- Tích hợp với các hệ thống đảm bảo quyền riêng tư khác (Differential Privacy, Secure Aggregation)
