#!/bin/bash
# Script: run_mnist_flower_datasets.sh
# Mô tả: Chạy federated learning với Flower + MNIST, hỗ trợ các chế độ: simulation, direct, server, client
# Hỗ trợ TLS/SSL (secure) và không bảo mật (insecure)
#
# Sử dụng:
#   ./run_mnist_flower_datasets.sh simulation [options]       # Chạy mô phỏng (không dùng TLS)
#   ./run_mnist_flower_datasets.sh direct [options]           # Chạy server + clients (TLS)
#   ./run_mnist_flower_datasets.sh server [options]           # Chạy server riêng (TLS)
#   ./run_mnist_flower_datasets.sh client <id> [options]      # Chạy client riêng (TLS)
#   ./run_mnist_flower_datasets.sh direct-insecure [options]  # Chạy direct mode không bảo mật
#   ./run_mnist_flower_datasets.sh server-insecure [options]  # Chạy server không bảo mật
#   ./run_mnist_flower_datasets.sh client-insecure <id> [options] # Chạy client không bảo mật
#   ./run_mnist_flower_datasets.sh check-certs [host] [port]  # Kiểm tra chứng chỉ TLS/SSL
#   ./run_mnist_flower_datasets.sh help                       # Hiển thị trợ giúp
#
# Options:
#   --num-clients N     Số lượng clients (chỉ cho direct mode, mặc định: 3)
#   --rounds N          Số lượng vòng training (mặc định: 3)
#   --host HOST         Địa chỉ máy chủ (mặc định: localhost)
#   --port PORT         Cổng kết nối (mặc định: 18443)        # Đã đổi cổng mặc định sang cao hơn
#   --verbose           Hiển thị thông tin chi tiết
#
# Script sẽ tự động kiểm tra và tạo lại chứng chỉ SSL nếu cần thiết

# Màu sắc cho output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Chuyển đến thư mục chứa script
cd "$(dirname "$0")"

# Khởi tạo các biến mặc định
PYTHON=python3
SCRIPT=run_mnist_flower_datasets.py
CHECK_CERTS_SCRIPT=check_certs_debug.py
NUM_CLIENTS=3
ROUNDS=3
HOST="localhost"
PORT=18443  # Đã đổi sang cổng cao hơn để tránh cần quyền root
VERBOSE=""

# Hàm hiển thị trợ giúp
show_help() {
    echo -e "${BLUE}=== Federated Learning với Flower và MNIST ===${NC}"
    echo -e "Sử dụng:"
    echo -e "  ${GREEN}$0 simulation [options]${NC}      # Chạy mô phỏng (không dùng TLS)"
    echo -e "  ${GREEN}$0 direct [options]${NC}          # Chạy server + clients (TLS)"
    echo -e "  ${GREEN}$0 server [options]${NC}          # Chạy server riêng (TLS)"
    echo -e "  ${GREEN}$0 client <id> [options]${NC}     # Chạy client riêng (TLS)"
    echo -e "  ${GREEN}$0 direct-insecure [options]${NC} # Chạy direct mode không bảo mật"
    echo -e "  ${GREEN}$0 server-insecure [options]${NC} # Chạy server không bảo mật"
    echo -e "  ${GREEN}$0 client-insecure <id> [options]${NC} # Chạy client không bảo mật"
    echo -e "  ${GREEN}$0 check-certs [host] [port]${NC} # Kiểm tra chứng chỉ TLS/SSL"
    echo -e "  ${GREEN}$0 help${NC}                      # Hiển thị trợ giúp này"
    echo
    echo -e "${BLUE}Options:${NC}"
    echo -e "  ${YELLOW}--num-clients N${NC}     # Số lượng clients (chỉ cho direct mode, mặc định: 3)"
    echo -e "  ${YELLOW}--rounds N${NC}          # Số lượng vòng training (mặc định: 3)"
    echo -e "  ${YELLOW}--host HOST${NC}         # Địa chỉ máy chủ (mặc định: localhost)"
    echo -e "  ${YELLOW}--port PORT${NC}         # Cổng kết nối (mặc định: 8443)"
    echo -e "  ${YELLOW}--verbose${NC}           # Hiển thị thông tin chi tiết"
    echo
    echo -e "${BLUE}Chú ý:${NC} Đảm bảo đã tạo thư mục ${YELLOW}certs/${NC} với đúng cấu trúc trước khi chạy chế độ TLS/SSL!"
}

# Phân tích các tham số dòng lệnh
MODE="$1"
shift

# Nếu không có tham số hoặc yêu cầu trợ giúp
if [ -z "$MODE" ] || [ "$MODE" = "help" ]; then
    show_help
    exit 0
fi

# Xử lý các tham số tùy chọn
while [ $# -gt 0 ]; do
    case "$1" in
        --num-clients)
            NUM_CLIENTS="$2"
            shift 2
            ;;
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        *)
            # Nếu là client mode, tham số đầu tiên là client-id
            if [[ "$MODE" = "client" || "$MODE" = "client-insecure" ]] && [ -z "$CID" ]; then
                CID="$1"
                shift
            else
                echo -e "${RED}[-] Tham số không hợp lệ: $1${NC}"
                show_help
                exit 1
            fi
            ;;
    esac
done

# Chạy theo chế độ được chỉ định
case "$MODE" in
    simulation)
        echo -e "${GREEN}[+] Chạy mô phỏng federated learning (simulation mode, không TLS)${NC}"
        $PYTHON $SCRIPT --mode simulation --rounds $ROUNDS $VERBOSE
        ;;
    direct)
        echo -e "${GREEN}[+] Chạy federated learning direct mode (server + $NUM_CLIENTS client, có TLS)${NC}"
        echo -e "${BLUE}[*] Host: $HOST, Port: $PORT, Rounds: $ROUNDS${NC}"
        # Regenerate certificates
if [ -x "./regenerate_certificates.sh" ]; then
    echo -e "${YELLOW}[*] Checking and regenerating SSL certificates if needed...${NC}"
    ./regenerate_certificates.sh
fi

$PYTHON $SCRIPT --mode direct --secure --rounds $ROUNDS --port $PORT $VERBOSE
        ;;
    direct-insecure)
        echo -e "${GREEN}[+] Chạy federated learning direct mode (server + $NUM_CLIENTS client, không bảo mật)${NC}"
        echo -e "${BLUE}[*] Host: $HOST, Port: $PORT, Rounds: $ROUNDS${NC}"
        $PYTHON $SCRIPT --mode direct --insecure --rounds $ROUNDS --port $PORT $VERBOSE
        ;;
    server)
        echo -e "${GREEN}[+] Chạy server riêng biệt (có TLS)${NC}"
        echo -e "${BLUE}[*] Host: $HOST, Port: $PORT, Rounds: $ROUNDS${NC}"
        
        # Use the SuperLink server script (recommended in newer Flower versions)
        if [ -x "./run_server_secure.sh" ]; then
            echo -e "${YELLOW}[*] Starting server using SuperLink (recommended)...${NC}"
            ./run_server_secure.sh $PORT
        else
            # Fallback to the old method
            # Regenerate certificates
            if [ -x "./regenerate_certificates.sh" ]; then
                echo -e "${YELLOW}[*] Checking and regenerating SSL certificates if needed...${NC}"
                ./regenerate_certificates.sh
            fi

            $PYTHON $SCRIPT --mode server --secure --rounds $ROUNDS --port $PORT $VERBOSE
        fi
        ;;
    server-insecure)
        echo -e "${GREEN}[+] Chạy server riêng biệt (không bảo mật)${NC}"
        echo -e "${BLUE}[*] Host: $HOST, Port: $PORT, Rounds: $ROUNDS${NC}"
        
        # Use the SuperLink server script (recommended in newer Flower versions)
        if [ -x "./run_server_insecure.sh" ]; then
            echo -e "${YELLOW}[*] Starting insecure server using SuperLink (recommended)...${NC}"
            ./run_server_insecure.sh $PORT
        else
            # Fallback to the old method
            $PYTHON $SCRIPT --mode server --insecure --rounds $ROUNDS --port $PORT $VERBOSE
        fi
        ;;
    client)
        if [ -z "$CID" ]; then
            echo -e "${RED}[-] Thiếu client ID. Sử dụng: $0 client <id>${NC}"
            exit 1
        fi
        echo -e "${GREEN}[+] Chạy client riêng biệt (có TLS), ID=$CID${NC}"
        echo -e "${BLUE}[*] Kết nối đến: $HOST:$PORT${NC}"
        # Regenerate certificates
if [ -x "./regenerate_certificates.sh" ]; then
    echo -e "${YELLOW}[*] Checking and regenerating SSL certificates if needed...${NC}"
    ./regenerate_certificates.sh
fi

$PYTHON $SCRIPT --mode client --client-id $CID --secure --port $PORT $VERBOSE
        ;;
    client-insecure)
        if [ -z "$CID" ]; then
            echo -e "${RED}[-] Thiếu client ID. Sử dụng: $0 client-insecure <id>${NC}"
            exit 1
        fi
        echo -e "${GREEN}[+] Chạy client riêng biệt (không bảo mật), ID=$CID${NC}"
        echo -e "${BLUE}[*] Kết nối đến: $HOST:$PORT${NC}"
        $PYTHON $SCRIPT --mode client --client-id $CID --insecure --port $PORT $VERBOSE
        ;;
    check-certs)
        echo -e "${GREEN}[+] Kiểm tra chứng chỉ TLS/SSL${NC}"
        $PYTHON $CHECK_CERTS_SCRIPT --host $HOST --port $PORT
        ;;
    *)
        echo -e "${RED}[-] Chế độ không hợp lệ: $MODE${NC}"
        show_help
        exit 1
        ;;
esac
