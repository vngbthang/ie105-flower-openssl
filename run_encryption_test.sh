#!/bin/bash
# Script này chạy bài kiểm tra xác thực mã hóa cho Flower TLS/SSL
# Nó xác minh dữ liệu được truyền giữa client và server thực sự được mã hóa

# Màu sắc cho đầu ra
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # Không màu

BASE_DIR=$(dirname "$0")
cd "$BASE_DIR" || exit

echo -e "${BOLD}===== KIỂM TRA XÁC THỰC MÃ HÓA FLOWER TLS/SSL =====${NC}"
echo -e "${BLUE}Bài kiểm tra này xác minh rằng dữ liệu được truyền giữa client và server"
echo -e "trong hệ thống Học Liên Hợp thực sự được mã hóa.${NC}"
echo ""

# Kiểm tra quyền sudo (cần thiết cho tcpdump)
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}⚠️ Cảnh báo: Script này cần quyền sudo để bắt gói tin với tcpdump.${NC}"
    echo -e "Sẽ yêu cầu mật khẩu sudo..."
    SUDO="sudo"
else
    SUDO=""
fi
if [ "$EUID" -ne 0 ]; then
  echo -e "${YELLOW}Script này cần quyền sudo để thu thập gói tin mạng.${NC}"
  echo "Vui lòng nhập mật khẩu khi được nhắc."
  sudo -v
  if [ $? -ne 0 ]; then
    echo -e "${RED}Lỗi: Không thể lấy quyền sudo. Đang thoát.${NC}"
    exit 1
  fi
fi

# Kiểm tra các công cụ cần thiết
echo -e "${BOLD}Bước 1: Đang kiểm tra công cụ cần thiết...${NC}"
if ! command -v tcpdump &> /dev/null; then
  echo -e "${RED}Lỗi: tcpdump chưa được cài đặt.${NC}"
  echo "Công cụ này là cần thiết cho bài kiểm tra. Bạn có muốn cài đặt ngay bây giờ không? (y/n)"
  read -r INSTALL_TCPDUMP
  if [[ "$INSTALL_TCPDUMP" =~ ^[Yy]$ ]]; then
    echo "Đang cài đặt tcpdump..."
    sudo apt update && sudo apt install -y tcpdump
    if [ $? -ne 0 ]; then
      echo -e "${RED}Lỗi: Không thể cài đặt tcpdump. Đang thoát.${NC}"
      exit 1
    fi
    echo -e "${GREEN}tcpdump đã được cài đặt thành công.${NC}"
  else
    echo -e "${RED}tcpdump là cần thiết cho bài kiểm tra này. Đang thoát.${NC}"
    exit 1
  fi
else
  echo -e "${GREEN}✓ tcpdump đã được cài đặt.${NC}"
fi

# Cài đặt tshark nếu chưa có (tùy chọn)
if ! command -v tshark &> /dev/null; then
  echo -e "${YELLOW}Cảnh báo: tshark (Wireshark CLI) chưa được cài đặt.${NC}"
  echo "Mặc dù không bắt buộc, tshark cung cấp phân tích chi tiết hơn về lưu lượng mã hóa."
  echo "Bạn có muốn cài đặt tshark ngay bây giờ không? (y/n)"
  read -r INSTALL_TSHARK
  if [[ "$INSTALL_TSHARK" =~ ^[Yy]$ ]]; then
    echo "Đang cài đặt tshark..."
    sudo apt update && sudo apt install -y tshark
    if [ $? -ne 0 ]; then
      echo -e "${YELLOW}Cảnh báo: Không thể cài đặt tshark. Tiếp tục với khả năng phân tích hạn chế...${NC}"
    else
      echo -e "${GREEN}✓ tshark đã được cài đặt thành công.${NC}"
    fi
  else
    echo "Tiếp tục với khả năng phân tích hạn chế..."
  fi
else
  echo -e "${GREEN}✓ tshark (Wireshark CLI) đã được cài đặt.${NC}"
fi

# Kiểm tra các phụ thuộc Python
echo -e "\n${BOLD}Bước 2: Đang kiểm tra các phụ thuộc Python...${NC}"
python3 -c "import torch, flwr" &>/dev/null
if [ $? -ne 0 ]; then
  echo -e "${YELLOW}Một số phụ thuộc Python bị thiếu.${NC}"
  echo "Bài kiểm tra yêu cầu các gói torch, torchvision và flwr."
  echo "Bạn có muốn cài đặt những phụ thuộc này ngay bây giờ không? (y/n)"
  read -r INSTALL_DEPS
  if [[ "$INSTALL_DEPS" =~ ^[Yy]$ ]]; then
    echo "Đang cài đặt phụ thuộc Python..."
    pip install torch torchvision flwr
    if [ $? -ne 0 ]; then
      echo -e "${RED}Lỗi: Không thể cài đặt các gói Python cần thiết.${NC}"
      echo "Vui lòng cài đặt thủ công bằng:"
      echo "  pip install torch torchvision flwr"
      exit 1
    fi
    echo -e "${GREEN}✓ Các phụ thuộc Python đã được cài đặt thành công.${NC}"
  else
    echo -e "${RED}Các gói Python cần thiết chưa được cài đặt. Đang thoát.${NC}"
    exit 1
  fi
else
  echo -e "${GREEN}✓ Tất cả các phụ thuộc Python cần thiết đã được cài đặt.${NC}"
fi

# Kiểm tra các chứng chỉ có tồn tại không
echo -e "\n${BOLD}Bước 3: Đang kiểm tra các chứng chỉ TLS/SSL...${NC}"
if [ ! -f "$BASE_DIR/certs/ca/ca.pem" ] || [ ! -f "$BASE_DIR/certs/server/server.pem" ] || [ ! -f "$BASE_DIR/certs/client/client.pem" ]; then
  echo -e "${YELLOW}Một số chứng chỉ TLS/SSL bị thiếu.${NC}"
  echo "Bạn có muốn tạo chúng ngay bây giờ không? (y/n)"
  read -r GEN_CERTS
  if [[ "$GEN_CERTS" =~ ^[Yy]$ ]]; then
    echo "Đang tạo chứng chỉ..."
    chmod +x "$BASE_DIR/generate_certs.sh"
    bash "$BASE_DIR/generate_certs.sh"
    if [ $? -ne 0 ]; then
      echo -e "${RED}Lỗi: Không thể tạo chứng chỉ. Đang thoát.${NC}"
      exit 1
    fi
    echo -e "${GREEN}✓ Chứng chỉ đã được tạo thành công.${NC}"
  else
    echo -e "${RED}Chứng chỉ là cần thiết cho bài kiểm tra này. Đang thoát.${NC}"
    exit 1
  fi
else
  echo -e "${GREEN}✓ Các chứng chỉ TLS/SSL đã có sẵn.${NC}"
fi

# Chạy bài kiểm tra với sudo (cần thiết cho thu thập gói tin)
echo -e "\n${BOLD}Bước 4: Đang chạy kiểm tra xác thực mã hóa...${NC}"
echo "Việc này sẽ khởi động một server và client Flower với một mô hình ML đơn giản để thu thập và phân tích lưu lượng mạng."
echo -e "${YELLOW}Lưu ý: Bài kiểm tra này có thể mất vài phút để hoàn thành.${NC}"
echo ""

read -p "Nhấn Enter để bắt đầu kiểm tra..."

# Run with or without sudo depending on if we're already root
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}⚠️ Chạy script với quyền sudo để cho phép bắt gói tin...${NC}"
    # Make sure dependencies are available for the root user
    echo -e "${BLUE}Đang cài đặt các gói cần thiết cho người dùng root...${NC}"
    # Get the Python path
    PY_PATH=$(which python3)
    # Install dependencies for root if needed
    sudo "$PY_PATH" -m pip install numpy torch flwr torchvision --quiet
    # Run the script
    sudo "$PY_PATH" "$BASE_DIR/test_encryption.py"
else
    python3 "$BASE_DIR/test_encryption.py"
fi
TEST_RESULT=$?

# Kiểm tra tệp kết quả
if [ -f "$BASE_DIR/encryption_test_results.log" ]; then
  echo -e "\n${BOLD}Bước 5: Tóm tắt Kiểm tra${NC}"
  echo -e "${BLUE}Kết quả kiểm tra đã được lưu vào: ${NC}${BOLD}$BASE_DIR/encryption_test_results.log${NC}"
  
  # Hiển thị tóm tắt nhanh
  if grep -q "TEST PASSED" "$BASE_DIR/encryption_test_results.log"; then
    echo -e "${GREEN}✅ KIỂM TRA MÃ HÓA ĐÃ VƯỢT QUA: Dữ liệu được mã hóa đúng cách giữa client và server${NC}"
  elif grep -q "PARTIAL TEST PASSED" "$BASE_DIR/encryption_test_results.log"; then
    echo -e "${YELLOW}⚠️ KIỂM TRA MÃ HÓA VƯỢT QUA MỘT PHẦN: Bài kiểm tra cho thấy dữ liệu được mã hóa, nhưng với xác minh hạn chế${NC}"
  else
    echo -e "${RED}❌ KIỂM TRA MÃ HÓA THẤT BẠI: Bài kiểm tra cho thấy dữ liệu có thể KHÔNG được mã hóa đúng cách${NC}"
  fi
  
  echo -e "\nĐể xem đầy đủ kết quả kiểm tra, bạn có thể chạy:"
  echo -e "  cat \"$BASE_DIR/encryption_test_results.log\""
else
  echo -e "\n${YELLOW}Cảnh báo: Không thể tìm thấy tệp nhật ký kết quả kiểm tra.${NC}"
fi

echo -e "\n${BOLD}Để biết thêm thông tin về quy trình xác thực mã hóa TLS/SSL, xem:${NC}"
echo -e "${BLUE}$BASE_DIR/ENCRYPTION_VERIFICATION.md${NC}"
