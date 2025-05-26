#!/bin/bash
# Script để chạy client sử dụng flower-supernode để tương thích với flower-superlink server

# Lấy thư mục gốc
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CERT_DIR="$BASE_DIR/certs"
HOST=${1:-localhost}
PORT=${2:-18443}
CLIENT_ID=${3:-0} # This CLIENT_ID is now mainly for logging or if FLOWER_CLIENT_ID env var is used elsewhere
SECURE_MODE=${4:-true} # Default to secure mode

# Đặt PYTHONPATH để bao gồm thư mục hiện tại cho việc import module
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}" # Ensure project root is in PYTHONPATH for imports

ensure_certificates_exist() {
    if [ ! -f "$CERT_DIR/ca/ca.pem" ] || \
       (! "$SECURE_MODE" && ([ ! -f "$CERT_DIR/client/client.pem" ] || [ ! -f "$CERT_DIR/client/client.key" ])); then
        echo -e "${YELLOW}Một hoặc nhiều chứng chỉ SSL cần thiết bị thiếu.${NC}"
        echo -e "${BLUE}Đang chạy regenerate_certificates.sh để tạo lại...${NC}"
        if [ -x "$BASE_DIR/regenerate_certificates.sh" ]; then
            "$BASE_DIR/regenerate_certificates.sh"
        else
            echo -e "${RED}Không tìm thấy script regenerate_certificates.sh. Vui lòng tạo chứng chỉ thủ công.${NC}"
            exit 1
        fi
    fi

    # Fix chain certificate if needed for server, though client primarily needs CA
    if [ -x "$BASE_DIR/fix_chain_certificate.sh" ]; then
        echo "Đảm bảo chain certificate đúng định dạng..."
        "$BASE_DIR/fix_chain_certificate.sh"
    fi
}

echo "Khởi động Flower SuperNode client với ID $CLIENT_ID kết nối đến $HOST:$PORT..."
echo "Sử dụng flower-supernode CLI..."

# Set FLOWER_CLIENT_ID environment variable, MnistClient might use it for logging node_id
# However, data partitioning will now primarily use the 'cid' passed by Flower to client_fn_for_app
export FLOWER_CLIENT_ID="$CLIENT_ID"

CERT_ARGS=""
if [ "$SECURE_MODE" = true ]; then
    echo "Sử dụng kết nối bảo mật với TLS..."
    ensure_certificates_exist
    if [ ! -f "$CERT_DIR/ca/ca.pem" ]; then
        echo -e "${RED}Lỗi: CA certificate ($CERT_DIR/ca/ca.pem) không tìm thấy. Không thể chạy client bảo mật.${NC}"
        exit 1
    fi
    CERT_ARGS="--root-certificates \"$CERT_DIR/ca/ca.pem\""
else
    echo "Sử dụng kết nối không bảo mật..."
    CERT_ARGS="--insecure"
fi

# Đường dẫn đến tệp cấu hình nút
NODE_CONFIG_FILE="$BASE_DIR/client/node_config.json"

# Xây dựng và chạy lệnh flower-supernode
# Bây giờ sử dụng --node-config thay vì tham số vị trí cho tham chiếu ứng dụng client
CMD="flower-supernode \
    --superlink=\"$HOST:$PORT\" \
    $CERT_ARGS \
    --node-config=\"$NODE_CONFIG_FILE\""

echo "Đang thực thi lệnh: $CMD"
eval $CMD
