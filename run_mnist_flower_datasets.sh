#!/bin/bash

# Script để chạy federated learning với MNIST dataset từ Flower Datasets

# Lấy đường dẫn đến thư mục dự án
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "===== Học Liên Hợp (Federated Learning) với MNIST từ Flower Datasets ====="

# Kiểm tra và cài đặt flwr-datasets nếu cần
if ! pip list | grep -q flwr-datasets; then
    echo "Đang cài đặt flwr-datasets..."
    pip install -q "flwr-datasets[vision]"
fi

# Xử lý các tham số dòng lệnh
MODE="direct"
SECURE=true
ROUNDS=3
TIMEOUT=300

while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --insecure)
            SECURE=false
            shift
            ;;
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --help|-h)
            echo "Sử dụng: $0 [tùy chọn]"
            echo "Tùy chọn:"
            echo "  --mode {direct|server|client|simulation}  Chọn chế độ chạy (mặc định: direct)"
            echo "  --insecure                               Chạy mà không dùng TLS/SSL"
            echo "  --rounds NUMBER                          Số vòng huấn luyện (mặc định: 3)"
            echo "  --timeout SECONDS                        Thời gian chạy tối đa (mặc định: 300 giây)"
            echo "  --help, -h                               Hiển thị trợ giúp này"
            exit 0
            ;;
        *)
            echo "Tùy chọn không hợp lệ: $1"
            echo "Dùng --help để xem hướng dẫn"
            exit 1
            ;;
    esac
done

# Đặt biến môi trường để tăng khả năng debug
export PYTHONUNBUFFERED=1
export GRPC_VERBOSITY=debug
export GRPC_TRACE=tcp,http,secure_endpoint,transport_security

# Đặt timeout cho Python script
export FL_TIMEOUT=$TIMEOUT

# Hiển thị cấu hình
echo "Cấu hình chạy:"
echo "- Chế độ: $MODE"
echo "- Bảo mật TLS/SSL: $SECURE"
echo "- Số vòng huấn luyện: $ROUNDS"
echo "- Thời gian chạy tối đa: $TIMEOUT giây"
echo

# Chạy federated learning với các tùy chọn đã chọn
SECURE_OPT=""
if [ "$SECURE" = false ]; then
    SECURE_OPT="--insecure"
fi

echo "Khởi động federated learning MNIST với chế độ $MODE..."

# Thêm timeout command để tránh treo vô hạn
timeout $((TIMEOUT+30)) python3 "${BASE_DIR}/run_mnist_flower_datasets.py" --mode "$MODE" $SECURE_OPT --rounds "$ROUNDS"
EXIT_CODE=$?

# Kiểm tra mã thoát
if [ $EXIT_CODE -eq 124 ] || [ $EXIT_CODE -eq 143 ]; then
    echo "Chương trình đã bị dừng do quá thời gian chờ ($TIMEOUT giây)"
elif [ $EXIT_CODE -ne 0 ]; then
    echo "Chương trình kết thúc với mã lỗi: $EXIT_CODE"
else
    echo "Hoàn thành chạy federated learning MNIST từ Flower Datasets!"
fi

exit $EXIT_CODE
