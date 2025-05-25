#!/bin/bash
# Script: run_mnist_flower_datasets.sh
# Mô tả: Chạy federated learning với Flower + MNIST, hỗ trợ các chế độ: simulation, direct, server, client
# Hỗ trợ TLS/SSL (secure) và không bảo mật (insecure)
#
# Sử dụng:
#   ./run_mnist_flower_datasets.sh simulation         # Chạy mô phỏng (không dùng TLS, không có traffic thực)
#   ./run_mnist_flower_datasets.sh direct            # Chạy server + 3 client (TLS, thread nội bộ, traffic thực)
#   ./run_mnist_flower_datasets.sh server            # Chạy server riêng (TLS)
#   ./run_mnist_flower_datasets.sh client <id>       # Chạy client riêng (TLS), <id> = 0, 1, 2
#   ./run_mnist_flower_datasets.sh direct-insecure   # Chạy direct mode không bảo mật
#   ./run_mnist_flower_datasets.sh server-insecure   # Chạy server không bảo mật
#   ./run_mnist_flower_datasets.sh client-insecure <id> # Chạy client không bảo mật
#
# Đảm bảo đã tạo certs/ đúng chuẩn nếu dùng TLS/SSL!

set -e
cd "$(dirname "$0")"

PYTHON=python3
SCRIPT=run_mnist_flower_datasets.py

MODE="$1"
CID="$2"

case "$MODE" in
  simulation)
    echo "[+] Chạy mô phỏng federated learning (simulation mode, không TLS)"
    $PYTHON $SCRIPT --mode simulation
    ;;
  direct)
    echo "[+] Chạy federated learning direct mode (server + 3 client, có TLS)"
    $PYTHON $SCRIPT --mode direct --secure
    ;;
  direct-insecure)
    echo "[+] Chạy federated learning direct mode (server + 3 client, không bảo mật)"
    $PYTHON $SCRIPT --mode direct --insecure
    ;;
  server)
    echo "[+] Chạy server riêng biệt (có TLS)"
    $PYTHON $SCRIPT --mode server --secure
    ;;
  server-insecure)
    echo "[+] Chạy server riêng biệt (không bảo mật)"
    $PYTHON $SCRIPT --mode server --insecure
    ;;
  client)
    if [ -z "$CID" ]; then
      echo "[-] Thiếu client ID. Sử dụng: $0 client <id>"
      exit 1
    fi
    echo "[+] Chạy client riêng biệt (có TLS), ID=$CID"
    $PYTHON $SCRIPT --mode client --client-id $CID --secure
    ;;
  client-insecure)
    if [ -z "$CID" ]; then
      echo "[-] Thiếu client ID. Sử dụng: $0 client-insecure <id>"
      exit 1
    fi
    echo "[+] Chạy client riêng biệt (không bảo mật), ID=$CID"
    $PYTHON $SCRIPT --mode client --client-id $CID --insecure
    ;;
  *)
    echo "[-] Tham số không hợp lệ. Các chế độ hợp lệ: simulation, direct, direct-insecure, server, server-insecure, client, client-insecure"
    echo "Ví dụ:"
    echo "  $0 simulation"
    echo "  $0 direct"
    echo "  $0 server"
    echo "  $0 client 0"
    exit 1
    ;;
esac
