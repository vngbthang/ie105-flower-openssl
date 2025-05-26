# Hướng Dẫn Phân Tích Gói Tin TLS/SSL Trong Flower với Wireshark

## Chuẩn Bị

1. **Cài đặt Wireshark**:
   ```bash
   sudo apt update
   sudo apt install wireshark
   ```

2. **Cấu hình quyền bắt gói tin** (nếu cần):
   ```bash
   sudo usermod -a -G wireshark $USER
   # Đăng xuất và đăng nhập lại để áp dụng thay đổi
   ```

## Bắt Gói Tin TLS/SSL trong Flower

### Phương Pháp 1: Sử dụng run_easy.sh

1. Chạy script `run_easy.sh` và chọn tùy chọn 8 (Chạy bắt gói tin Wireshark)
2. Nhập cổng mà server Flower sẽ lắng nghe (mặc định: 18443)
3. Wireshark sẽ tự động mở với bộ lọc `tcp port 18443 and tls`
4. Mở terminal mới để chạy server (`run_easy.sh` và chọn tùy chọn 1)
5. Mở terminal khác để chạy client (`run_easy.sh` và chọn tùy chọn 3)

### Phương Pháp 2: Chạy Wireshark Thủ Công

1. Mở Wireshark với quyền root:
   ```bash
   sudo wireshark
   ```

2. Chọn giao diện mạng để bắt gói tin (thường là `lo` cho localhost)

3. Thiết lập bộ lọc:
   ```
   tcp port 18443 && tls
   ```

4. Nhấn "Start Capturing Packets"

5. Chạy server và client Flower trong các terminal riêng biệt

## Phân Tích Kết Quả

### 1. TLS Handshake

Tìm các gói tin trao đổi handshake TLS ban đầu:

- **Client Hello**: Client gửi thông tin về phiên bản TLS hỗ trợ
- **Server Hello**: Server phản hồi và chọn phiên bản TLS và cipher suite
- **Certificate**: Server gửi chứng chỉ SSL/TLS
- **Key Exchange**: Quá trình trao đổi khóa

### 2. Xác Thực Dữ Liệu Được Mã Hóa

Sau khi handshake TLS hoàn tất, bạn sẽ thấy các gói tin "Application Data":
- Dữ liệu này được mã hóa hoàn toàn
- Không thể đọc được nội dung thô của gói tin
- Đảm bảo rằng tham số mô hình được bảo vệ trong quá trình truyền

### 3. So Sánh Với Kết Nối Không Bảo Mật

Để so sánh, bạn có thể bắt gói tin không có TLS (cổng 18080):
```
tcp port 18080
```

Với kết nối không bảo mật:
- Không có quá trình TLS handshake
- Dữ liệu gRPC được truyền ở dạng không mã hóa
- Tham số mô hình có thể bị chặn và đọc

## Ghi Chú Quan Trọng

- Ngay cả khi server đang sử dụng "simulation mode" trong Flower, giao tiếp giữa client và server vẫn được mã hóa nếu TLS/SSL được bật
- Để phân tích sâu hơn, bạn có thể lưu lại file `.pcap` từ Wireshark để phân tích sau
- Phân tích kỹ các gói tin Application Data để đảm bảo không có dữ liệu nào bị lộ

## Kết luận

Nếu kết nối TLS/SSL hoạt động chính xác, bạn sẽ thấy:
- Quá trình bắt tay TLS diễn ra đúng quy trình
- Tất cả dữ liệu được mã hóa trong quá trình truyền tải
- Không thể đọc được nội dung dữ liệu từ bên ngoài
