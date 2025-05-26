# Xác Thực Mã Hóa TLS/SSL trong Giao Tiếp Học Liên Hợp

## Tổng Quan

Tài liệu này trình bày phương pháp phân tích và xác minh bảo mật của kết nối TLS/SSL trong hệ thống Học Liên Hợp (Federated Learning) sử dụng framework Flower. Mục tiêu chính là kiểm chứng rằng dữ liệu truyền giữa client và server thực sự được mã hóa, đảm bảo tính bảo mật và toàn vẹn của các tham số mô hình trong quá trình truyền tải.

## Tầm Quan Trọng của Mã Hóa trong Học Liên Hợp

- Bảo vệ tham số mô hình, quyền riêng tư, chống tấn công trung gian
- Đảm bảo dữ liệu truyền giữa client và server luôn được mã hóa

## Hướng dẫn nhanh kiểm tra bảo mật với script tự động

1. **Chạy script menu tự động:**
   ```bash
   ./run_easy.sh
   ```
   - Chọn các tùy chọn để chạy server, client, hoặc bắt gói tin Wireshark (tùy chọn 8)
   - Có thể xem hướng dẫn chi tiết trong file [`wireshark_analysis.md`](wireshark_analysis.md)

2. **Chạy server và client thực sự (không phải simulation):**
   - Server: `./start_server_superlink.sh 18443`
   - Client: `./start_client_supernode.sh localhost 18443 0`

3. **Bắt gói tin TLS/SSL:**
   - Có thể chọn trực tiếp trong menu hoặc tự mở Wireshark:
   ```bash
   sudo wireshark
   ```
   - Bộ lọc: `tcp.port == 18443 && tls`

4. **Phân tích:**
   - Xác nhận có handshake TLS, các gói Application Data được mã hóa
   - So sánh với chế độ không bảo mật (cổng 18080)

## Lưu ý

- Simulation mode không tạo ra lưu lượng mạng thực, không thể bắt gói tin TLS
- Để kiểm tra bảo mật thực tế, luôn chạy server và client riêng biệt với TLS/SSL
- Có thể lưu file .pcap để phân tích lại

## Kết luận

- Nếu thấy handshake TLS và Application Data được mã hóa, hệ thống đã bảo mật đúng
- Nếu chạy không bảo mật, dữ liệu có thể bị đọc trực tiếp
- Luôn kiểm tra bằng Wireshark để xác thực bảo mật thực tế

## Tài liệu tham khảo

1. **TLS/SSL Protocols**
   - [TLS 1.3 Specification - RFC 8446](https://datatracker.ietf.org/doc/html/rfc8446)
   - [TLS 1.2 Specification - RFC 5246](https://datatracker.ietf.org/doc/html/rfc5246)
   - Rescorla, E. (2018). "The Transport Layer Security (TLS) Protocol Version 1.3."

2. **OpenSSL**
   - [OpenSSL Official Documentation](https://www.openssl.org/docs/)
   - [OpenSSL Command-Line HOWTO](https://www.madboa.com/geek/openssl/)
   - Viega, J., et al. (2002). "Network Security with OpenSSL." O'Reilly Media.

3. **Wireshark và Phân tích mạng**
   - [Wireshark User's Guide](https://www.wireshark.org/docs/wsug_html/)
   - [Analyzing TLS with Wireshark](https://wiki.wireshark.org/TLS)
   - Sanders, C. (2017). "Practical Packet Analysis: Using Wireshark to Solve Real-World Network Problems." No Starch Press.

4. **bảo mật trong Federated Learning**
   - Bonawitz, K., et al. (2017). "Practical secure aggregation for privacy-preserving machine learning." CCS.
   - Truex, S., et al. (2019). "A hybrid approach to privacy-preserving federated learning." AISec.
   - Kairouz, P., et al. (2021). "Advances and open problems in federated learning." Foundations and Trends in Machine Learning.

5. **mTLS (Mutual TLS)**
   - [Mutual TLS Authentication - IETF](https://datatracker.ietf.org/doc/html/rfc8446#section-4.6.2)
   - [Understanding mutual TLS Authentication](https://www.cloudflare.com/learning/access-management/what-is-mutual-tls/)
