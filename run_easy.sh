#!/bin/bash
# Script tổng hợp để chạy Flower Federated Learning dễ dàng
# Với các cập nhật mới để đảm bảo executor_fixed được sử dụng đúng cách và hỗ trợ bắt gói tin Wireshark
#
# Các tùy chọn hiện tại:
# 1: Chạy server bảo mật (SSL/TLS)
# 2: Chạy server không bảo mật
# 3: Chạy client bảo mật
# 4: Chạy client không bảo mật
# 5: Chạy chế độ mô phỏng
# 6: Sửa chữa chứng chỉ SSL/TLS
# 7: Chạy kiểm tra kết nối
# 8: Chạy bắt gói tin Wireshark
# 9: Thoát
#
# Cập nhật mới nhất: Hỗ trợ Wireshark và phát hiện client thực tốt hơn với executor_fixed.py

# Màu sắc
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Thư mục gốc
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo -e "${BLUE}5)${NC} Chạy chế độ mô phỏng (simulation)"
    echo -e "${BLUE}6)${NC} Sửa chữa chứng chỉ SSL/TLS"
    echo -e "${BLUE}7)${NC} Chạy kiểm tra kết nối"
    echo -e "${BLUE}8)${NC} Chạy bắt gói tin Wireshark"
    echo -e "${BLUE}9)${NC} Thoát"
    echo
    echo -n "Nhập lựa chọn của bạn (1-9): "='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Thư mục gốc
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Kiểm tra môi trường
echo -e "${BLUE}Kiểm tra môi trường...${NC}"
which python3 > /dev/null || { echo -e "${RED}Không tìm thấy Python3!${NC}"; exit 1; }
which flower-superlink > /dev/null || { echo -e "${YELLOW}Cảnh báo: flower-superlink không khả dụng. Sử dụng python API thay thế cho server.${NC}"; }
which flower-supernode > /dev/null || { echo -e "${YELLOW}Cảnh báo: flower-supernode không khả dụng. Sử dụng python API thay thế cho client.${NC}"; }

# Hiển thị thông báo về hướng dẫn phân tích Wireshark
if [ -f "$BASE_DIR/wireshark_analysis.md" ]; then
    echo -e "${GREEN}Lưu ý: Tài liệu hướng dẫn phân tích Wireshark có sẵn tại:${NC}"
    echo -e "${BLUE}$BASE_DIR/wireshark_analysis.md${NC}"
    echo
fi

# Sửa chữa các chứng chỉ SSL/TLS trước
fix_certificates() {
    echo -e "${BLUE}Đang kiểm tra và sửa chữa chứng chỉ SSL...${NC}"
    if [ -x "$BASE_DIR/regenerate_certificates.sh" ]; then
        "$BASE_DIR/regenerate_certificates.sh" > /dev/null
    fi
    
    if [ -x "$BASE_DIR/fix_chain_certificate.sh" ]; then
        "$BASE_DIR/fix_chain_certificate.sh" > /dev/null
    fi
    echo -e "${GREEN}Đã hoàn tất việc kiểm tra và sửa chữa chứng chỉ SSL/TLS!${NC}"
}

# Chạy server bảo mật
run_secure_server() {
    port=${1:-18443}
    echo -e "${BLUE}Khởi động server bảo mật trên cổng $port...${NC}"
    if [ -x "$BASE_DIR/start_server_superlink.sh" ]; then
        echo -e "${GREEN}Sử dụng SuperLink server với executor_fixed (khuyến nghị)...${NC}"
        "$BASE_DIR/start_server_superlink.sh" $port
    elif [ -x "$BASE_DIR/run_server_secure.sh" ]; then
        echo -e "${YELLOW}Sử dụng SuperLink server thông thường...${NC}"
        "$BASE_DIR/run_server_secure.sh" $port
    else
        echo -e "${YELLOW}SuperLink script không khả dụng, sử dụng Python API...${NC}"
        python3 "$BASE_DIR/run_mnist_flower_datasets.py" --mode server --secure --port $port --verbose
    fi
}

# Chạy server không bảo mật
run_insecure_server() {
    port=${1:-18080}
    echo -e "${BLUE}Khởi động server không bảo mật trên cổng $port...${NC}"
    if [ -x "$BASE_DIR/run_server_insecure.sh" ]; then
        echo -e "${GREEN}Sử dụng SuperLink server (khuyến nghị)...${NC}"
        "$BASE_DIR/run_server_insecure.sh" $port
    else
        echo -e "${YELLOW}SuperLink script không khả dụng, sử dụng Python API...${NC}"
        python3 "$BASE_DIR/run_mnist_flower_datasets.py" --mode server --insecure --port $port --verbose
    fi
}

# Chạy client bảo mật
run_secure_client() {
    client_id=${1:-0}
    port=${2:-18443}
    host=${3:-localhost}
    echo -e "${BLUE}Khởi động client bảo mật với ID $client_id kết nối tới $host:$port...${NC}"
    if [ -x "$BASE_DIR/start_client_supernode.sh" ]; then
        echo -e "${GREEN}Sử dụng SuperNode client (khuyến nghị cho kết nối thực sự)...${NC}"
        "$BASE_DIR/start_client_supernode.sh" $host $port $client_id
    elif [ -x "$BASE_DIR/run_client_supernode.sh" ]; then
        echo -e "${GREEN}Sử dụng SuperNode client thông thường...${NC}"
        "$BASE_DIR/run_client_supernode.sh" $host $port $client_id
    elif [ -x "$BASE_DIR/run_client_secure.sh" ]; then
        echo -e "${YELLOW}Sử dụng client API cũ (không khuyến khích, có thể gặp lỗi)...${NC}"
        "$BASE_DIR/run_client_secure.sh" $host $port $client_id
    else
        echo -e "${YELLOW}Sử dụng Python API cũ (không khuyến khích, có thể gặp lỗi)...${NC}"
        python3 "$BASE_DIR/run_mnist_flower_datasets.py" --mode client --secure --client-id $client_id --host $host --port $port --verbose
    fi
}

# Chạy client không bảo mật
run_insecure_client() {
    client_id=${1:-0}
    port=${2:-18080}
    host=${3:-localhost}
    echo -e "${BLUE}Khởi động client không bảo mật với ID $client_id kết nối tới $host:$port...${NC}"
    python3 "$BASE_DIR/run_mnist_flower_datasets.py" --mode client --insecure --client-id $client_id --host $host --port $port --verbose
}

# Hiển thị menu
show_menu() {
    echo -e "${GREEN}=== Flower Federated Learning với MNIST ===${NC}"
    echo -e "${YELLOW}Chọn một tùy chọn:${NC}"
    echo -e "${BLUE}1)${NC} Chạy server bảo mật (SSL/TLS) - cổng 18443"
    echo -e "${BLUE}2)${NC} Chạy server không bảo mật - cổng 18080"
    echo -e "${BLUE}3)${NC} Chạy client bảo mật"
    echo -e "${BLUE}4)${NC} Chạy client không bảo mật"
    echo -e "${BLUE}5)${NC} Chạy chế độ mô phỏng (simulation)"
    echo -e "${BLUE}6)${NC} Sửa chữa chứng chỉ SSL/TLS"
    echo -e "${BLUE}7)${NC} Chạy kiểm tra kết nối"
    echo -e "${BLUE}8)${NC} Chạy bắt gói tin Wireshark"
    echo -e "${BLUE}9)${NC} Thoát"
    echo
    echo -n "Nhập lựa chọn của bạn (1-9): "
}

# Xử lý menu
handle_menu() {
    local choice
    read choice
    case $choice in
        1) 
            fix_certificates
            run_secure_server
            ;;
        2) 
            run_insecure_server
            ;;
        3) 
            echo -n "Nhập client ID (mặc định: 0): "
            read client_id
            client_id=${client_id:-0}
            echo -n "Nhập cổng server (mặc định: 18443): "
            read port
            port=${port:-18443}
            echo -n "Nhập địa chỉ host (mặc định: localhost): "
            read host
            host=${host:-localhost}
            fix_certificates
            run_secure_client $client_id $port $host
            ;;
        4) 
            echo -n "Nhập client ID (mặc định: 0): "
            read client_id
            client_id=${client_id:-0}
            echo -n "Nhập cổng server (mặc định: 18080): "
            read port
            port=${port:-18080}
            echo -n "Nhập địa chỉ host (mặc định: localhost): "
            read host
            host=${host:-localhost}
            run_insecure_client $client_id $port $host
            ;;
        5) 
            echo -n "Nhập số lượng clients (mặc định: 3): "
            read num_clients
            num_clients=${num_clients:-3}
            echo -e "${BLUE}Chạy chế độ mô phỏng với $num_clients clients...${NC}"
            python3 "$BASE_DIR/run_mnist_flower_datasets.py" --mode simulation --num-clients $num_clients --verbose
            ;;
        6) 
            fix_certificates
            echo -e "${GREEN}Chứng chỉ SSL/TLS đã được sửa chữa thành công!${NC}"
            show_menu
            handle_menu
            ;;
        7) 
            echo -e "${BLUE}Đang chạy kiểm tra kết nối...${NC}"
            if [ -x "$BASE_DIR/test_connection_fixes.sh" ]; then
                "$BASE_DIR/test_connection_fixes.sh"
            else
                echo -e "${YELLOW}Script kiểm tra kết nối không tìm thấy. Chạy diagnose_ssl.py...${NC}"
                python3 "$BASE_DIR/diagnose_ssl.py"
            fi
            show_menu
            handle_menu
            ;;
        8)
            echo -e "${BLUE}Chạy bắt gói tin Wireshark để phân tích TLS...${NC}"
            port=${port:-18443}
            echo -e "${YELLOW}Chuẩn bị bắt gói tin trên cổng $port${NC}"
            echo -e "${YELLOW}Đảm bảo rằng bạn đã cài đặt wireshark và có quyền sudo${NC}"
            echo -e "${GREEN}Nhấn Enter để tiếp tục hoặc Ctrl+C để hủy...${NC}"
            read -r
            echo -e "${BLUE}Đang khởi động Wireshark để bắt gói tin...${NC}"
            if command -v wireshark > /dev/null 2>&1; then
                sudo wireshark -i lo -f "tcp port $port and tls" &
                echo -e "${GREEN}Wireshark đã được khởi động!${NC}"
                echo -e "${GREEN}Bây giờ bạn có thể chạy server và client trong các terminal khác.${NC}"
            else
                echo -e "${RED}Wireshark không được cài đặt. Vui lòng cài đặt Wireshark trước.${NC}"
                echo -e "${YELLOW}Bạn có thể cài đặt bằng lệnh: sudo apt install wireshark${NC}"
            fi
            echo
            echo -e "${GREEN}Nhấn Enter để quay lại menu chính...${NC}"
            read
            show_menu
            handle_menu
            ;;
        9) 
            echo -e "${GREEN}Cảm ơn bạn đã sử dụng!${NC}"
            exit 0
            ;;
        *) 
            echo -e "${RED}Lựa chọn không hợp lệ. Vui lòng chọn lại.${NC}"
            show_menu
            handle_menu
            ;;
    esac
}

# Chạy chương trình
fix_certificates
show_menu
handle_menu
