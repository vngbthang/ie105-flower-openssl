# Hướng Dẫn Chạy Hệ Thống Federated Learning và Phân Tích Kết Quả

## 1. Cách Chạy Hệ Thống

### 1.1. Sử dụng Script `run_easy.sh`

Script `run_easy.sh` là cách đơn giản nhất để chạy hệ thống Federated Learning:

```bash
./run_easy.sh
```

Script sẽ hiển thị menu với các tùy chọn:
- **1:** Chạy server bảo mật (SSL/TLS) - cổng 18443
- **2:** Chạy server không bảo mật - cổng 18080
- **3:** Chạy client bảo mật
- **4:** Chạy client không bảo mật
- **5:** Chạy chế độ mô phỏng (simulation)
- **6:** Sửa chữa chứng chỉ SSL/TLS
- **7:** Chạy kiểm tra kết nối
- **8:** Thoát

### 1.2. Các Bước Chạy Hệ Thống

#### Cách 1: Chạy Server và Client Riêng Biệt (khuyến nghị)

1. **Khởi động Server:**
   - Mở terminal thứ nhất
   - Chạy `./run_easy.sh`
   - Chọn 1 cho server bảo mật (SSL/TLS) hoặc 2 cho server không bảo mật
   - Đợi server khởi động hoàn tất (thấy thông báo "Starting Fleet API")

2. **Khởi động Client:**
   - Mở terminal thứ hai
   - Chạy `./run_easy.sh`
   - Chọn 3 cho client bảo mật hoặc 4 cho client không bảo mật
   - Nhập các thông số như client ID, cổng server, địa chỉ host (hoặc sử dụng giá trị mặc định)

3. **Khởi động Nhiều Client:**
   - Lặp lại bước 2 trong nhiều terminal khác nhau, nhưng hãy đảm bảo sử dụng ID client khác nhau

#### Cách 2: Chạy Chế Độ Mô Phỏng

1. **Chạy Mô Phỏng:**
   - Chạy `./run_easy.sh`
   - Chọn 5 để chạy chế độ mô phỏng
   - Nhập số lượng clients mong muốn (mặc định: 3)
   - Hệ thống sẽ tự động chạy cả server và clients trong cùng một tiến trình

## 2. Theo Dõi Quá Trình Huấn Luyện

### 2.1. Hiểu Các Thông Báo Trạng Thái

- `ChannelConnectivity.IDLE`: Client đã khởi tạo nhưng chưa bắt đầu kết nối
- `ChannelConnectivity.CONNECTING`: Client đang kết nối đến server
- `ChannelConnectivity.READY`: Client đã kết nối thành công và sẵn sàng giao tiếp

### 2.2. Xác Nhận Quá Trình Huấn Luyện Đang Hoạt Động

Khi quá trình huấn luyện thực sự bắt đầu, bạn sẽ thấy các thông báo sau trên client:

```
INFO - Các thông báo về fit round và evaluation round
INFO - Thông báo về quá trình huấn luyện như số epoch, loss, accuracy
```

Nếu không thấy các thông báo trên trong 1-2 phút sau khi kết nối, có thể có vấn đề trong việc bắt đầu quá trình huấn luyện.

### 2.3. Khắc Phục Sự Cố

Nếu client kết nối thành công (thấy `ChannelConnectivity.READY`) nhưng không bắt đầu huấn luyện:

1. **Kiểm tra server:**
   - Server có thể chưa sẵn sàng cung cấp nhiệm vụ huấn luyện
   - Đảm bảo server đang chạy và hiển thị thông báo "Starting Fleet API"

2. **Khởi động lại hệ thống:**
   - Dừng tất cả các tiến trình client và server
   - Khởi động lại server trước, sau đó là client

3. **Sử dụng chế độ mô phỏng:**
   - Chọn tùy chọn 5 trong menu để chạy ở chế độ mô phỏng
   - Điều này giúp loại bỏ các vấn đề về kết nối mạng

## 3. Phân Tích Với Wireshark

### 3.1. Thiết Lập Wireshark

1. **Cài Đặt Wireshark:**
   ```bash
   sudo apt-get update
   sudo apt-get install wireshark
   # Hoặc trên Fedora:
   sudo dnf install wireshark
   ```

2. **Khởi Động Wireshark:**
   ```bash
   sudo wireshark
   ```

3. **Chọn Giao Diện Mạng:**
   - Chọn `lo` (loopback) nếu server và client đang chạy trên cùng máy
   - Chọn giao diện mạng chính (như `eth0` hoặc `wlan0`) nếu chạy qua mạng

4. **Bắt Đầu Bắt Gói Tin:**
   - Nhấn vào nút "Start capturing packets" (biểu tượng xanh)

### 3.2. Thiết Lập Bộ Lọc Trong Wireshark

1. **Lọc Theo Cổng:**
   - Cho kết nối bảo mật: `tcp.port == 18443`
   - Cho kết nối không bảo mật: `tcp.port == 18080`

2. **Lọc Giao Thức TLS:**
   - Chỉ xem các kết nối TLS: `tls`

3. **Lọc Kết Hợp:**
   - Xem kết nối TLS trên cổng cụ thể: `tcp.port == 18443 && tls`
   - Xem kết nối gRPC không mã hóa: `tcp.port == 18080 && grpc`

### 3.3. Phân Tích Kết Quả

#### 3.3.1. Kết Nối Bảo Mật (SSL/TLS)

Trong kết nối bảo mật, bạn sẽ thấy:

1. **Bắt Tay TLS:**
   - Client Hello: Client gửi phiên bản TLS, bộ mã hóa, ID phiên
   - Server Hello: Server chọn phiên bản TLS và bộ mã hóa
   - Certificate: Server gửi chứng chỉ của mình
   - Key Exchange: Trao đổi khóa để thiết lập kết nối mã hóa
   - Finished: Bắt tay hoàn tất, kết nối đã được mã hóa

2. **Dữ Liệu Mã Hóa:**
   - Sau khi bắt tay, tất cả dữ liệu giao tiếp sẽ hiển thị là "Application Data"
   - Dữ liệu này đã được mã hóa và không thể đọc được trực tiếp

#### 3.3.2. Kết Nối Không Bảo Mật

Trong kết nối không bảo mật, bạn sẽ thấy:

1. **Gói tin gRPC:**
   - Có thể nhìn thấy các phương thức gRPC được gọi
   - Headers của các gói tin không bị mã hóa

2. **Dữ Liệu:**
   - Dữ liệu được truyền đi có thể phân tích được (trừ khi framework tự mã hóa dữ liệu)

### 3.4. So Sánh và Phân Tích Sâu

1. **Xem Thông Tin Chi Tiết Của Gói Tin:**
   - Nhấp chuột phải vào một gói tin
   - Chọn "Follow" > "TCP Stream" hoặc "TLS Stream" (với kết nối TLS)

2. **Xem Thống Kê Hiệu Suất:**
   - Vào "Statistics" > "I/O Graph" để xem đồ thị lưu lượng
   - Vào "Statistics" > "TCP Stream Graphs" > "Round Trip Time" để xem độ trễ

3. **So Sánh:**
   - **Kích thước gói tin:** Kết nối TLS thường có gói tin lớn hơn do overhead của mã hóa
   - **Độ trễ:** Kết nối TLS thường có độ trễ cao hơn do xử lý mã hóa/giải mã
   - **Bảo mật:** Kết nối TLS bảo vệ dữ liệu khỏi bị đọc hoặc sửa đổi

## 4. Lưu Ý Quan Trọng

1. **Xác Nhận Huấn Luyện:**
   - Nếu client kết nối (READY) nhưng không thấy thông báo huấn luyện, có thể server chưa bắt đầu rounds huấn luyện
   - Thử thêm client khác hoặc khởi động lại server

2. **Phân Tích Wireshark Hiệu Quả:**
   - Chạy Wireshark trước khi bắt đầu server và client để bắt đầy đủ quá trình bắt tay TLS
   - Sử dụng bộ lọc để giảm số lượng gói tin hiển thị và dễ dàng phân tích

3. **Bắt Nhiều Kết Nối:**
   - Để so sánh, hãy chạy cả kết nối bảo mật và không bảo mật và bắt gói tin từ cả hai
   - Lọc theo các cổng khác nhau để phân biệt lưu lượng

## 5. Giải Quyết Sự Cố Thường Gặp

1. **Client kết nối nhưng không huấn luyện:**
   - Đảm bảo đã khởi động server trước client
   - Kiểm tra các thông báo lỗi trong console
   - Thử khởi động lại cả server và client

2. **Không bắt được gói tin trong Wireshark:**
   - Đảm bảo đã chọn đúng interface mạng
   - Kiểm tra bộ lọc đã sử dụng có chính xác không
   - Tạm thời loại bỏ bộ lọc để xem tất cả gói tin

3. **Lỗi TLS/SSL:**
   - Chạy tùy chọn 6 trong menu để sửa chứng chỉ
   - Đảm bảo chứng chỉ được tạo đúng cách
   - Kiểm tra ngày hết hạn của chứng chỉ
