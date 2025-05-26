# Hướng Dẫn Phân Tích Gói Tin TLS/SSL Trong Flower với Wireshark

## Tổng quan

Tài liệu này hướng dẫn cách bắt và phân tích gói tin TLS/SSL trong hệ thống Flower sử dụng các script tự động của dự án.

## 1. Chạy bắt gói tin tự động (Khuyến nghị)

1. Chạy script menu:
   ```bash
   ./run_easy.sh
   ```
2. Chọn tùy chọn 8 (Chạy bắt gói tin Wireshark)
3. Nhập cổng server (mặc định: 18443)
4. Wireshark sẽ tự động mở với bộ lọc phù hợp
5. Mở terminal mới để chạy server (tùy chọn 1) và client (tùy chọn 3)

## 2. Chạy Wireshark thủ công

1. Mở Wireshark với quyền root:
   ```bash
   sudo wireshark
   ```
2. Chọn giao diện mạng (thường là `lo` cho localhost)
3. Thiết lập bộ lọc:
   ```
   tcp port 18443 && tls
   ```
4. Nhấn "Start Capturing Packets"
5. Chạy server và client trong các terminal riêng biệt

## 3. Phân tích kết quả

- **TLS Handshake**: Tìm các gói Client Hello, Server Hello, Certificate...
- **Application Data**: Dữ liệu sau handshake sẽ được mã hóa hoàn toàn
- **So sánh với kết nối không bảo mật**: Bắt gói trên cổng 18080 để thấy dữ liệu không mã hóa

## 4. Lưu ý

- Simulation mode không tạo ra lưu lượng mạng thực, không thể bắt gói tin TLS
- Để kiểm tra bảo mật thực tế, luôn chạy server và client riêng biệt với TLS/SSL
- Có thể lưu file .pcap để phân tích lại

## 5. Kết luận

- Nếu thấy handshake TLS và Application Data được mã hóa, hệ thống đã bảo mật đúng
- Nếu chạy không bảo mật, dữ liệu có thể bị đọc trực tiếp
- Luôn kiểm tra bằng Wireshark để xác thực bảo mật thực tế

## Tài liệu tham khảo

1. **Wireshark và phân tích gói tin**
   - [Wireshark User's Guide](https://www.wireshark.org/docs/wsug_html/)
   - [Display Filter Reference](https://www.wireshark.org/docs/dfref/)
   - Sanders, C. (2017). "Practical Packet Analysis: Using Wireshark to Solve Real-World Network Problems." No Starch Press.

2. **TLS Protocol**
   - [Wireshark Wiki: TLS](https://wiki.wireshark.org/TLS)
   - [TLS 1.3 Specification - RFC 8446](https://datatracker.ietf.org/doc/html/rfc8446)
   - Rescorla, E. (2000). "SSL and TLS: Designing and Building Secure Systems." Addison-Wesley.

3. **gRPC Protocol**
   - [gRPC Official Documentation](https://grpc.io/docs/)
   - [gRPC Protocol Security](https://grpc.io/docs/guides/auth/)

4. **Bảo mật trong Flower Framework**
   - [Flower Security Documentation](https://flower.dev/docs/framework/how-to-use-ssl-tls.html)
   - [Secure Aggregation in Federated Learning](https://arxiv.org/abs/1902.08927)

5. **Công cụ phân tích bảo mật**
   - [tshark - Wireshark CLI](https://www.wireshark.org/docs/man-pages/tshark.html)
   - [ssldump - SSL/TLS network protocol analyzer](https://github.com/adulau/ssldump)
   - [OpenSSL s_client](https://www.openssl.org/docs/man1.1.1/man1/s_client.html)
