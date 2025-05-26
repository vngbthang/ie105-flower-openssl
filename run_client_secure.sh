#!/bin/bash
# Script to run a Flower client with secure connection

# Base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_DIR="$BASE_DIR/certs"

# Configuration
HOST=${1:-"localhost"} 
PORT=${2:-18443}
CLIENT_ID=${3:-0}

# Make sure server is running first
echo "Checking if server is running on $HOST:$PORT..."
timeout 2 bash -c "</dev/tcp/$HOST/$PORT" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "CẢNH BÁO: Không thấy server đang chạy ở $HOST:$PORT"
    echo "Sẽ thử các port khác (18443, 28443, 18080)..."
    
    # Try alternative ports
    ALTERNATIVE_PORTS=(18443 28443 18080 48443 58443)
    SERVER_FOUND=false
    
    for ALT_PORT in "${ALTERNATIVE_PORTS[@]}"; do
        if [ "$ALT_PORT" = "$PORT" ]; then
            continue  # Skip the original port that failed
        fi
        
        echo "Kiểm tra server ở $HOST:$ALT_PORT..."
        timeout 1 bash -c "</dev/tcp/$HOST/$ALT_PORT" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "Đã tìm thấy server đang chạy ở $HOST:$ALT_PORT"
            PORT=$ALT_PORT
            SERVER_FOUND=true
            break
        fi
    done
    
    if [ "$SERVER_FOUND" = false ]; then
        echo "Không thể tìm thấy server đang chạy trên bất kỳ cổng nào."
        echo "Hãy chạy server trước với lệnh: ./run_with_superlink.sh $PORT"
        echo "Hoặc: ./run_mnist_flower_datasets.sh server --port $PORT"
        echo -n "Bạn có muốn tiếp tục không? (y/n): "
        read answer
        if [ "$answer" != "y" ]; then
            exit 1
        fi
    fi
fi

# Make sure certificates exist and chain is properly formatted
if [ -x "$BASE_DIR/regenerate_certificates.sh" ]; then
    echo "Checking and regenerating certificates if needed..."
    "$BASE_DIR/regenerate_certificates.sh"
fi

# Fix the chain certificate to ensure it's properly formatted
if [ -x "$BASE_DIR/fix_chain_certificate.sh" ]; then
    echo "Ensuring the chain certificate is properly formatted..."
    "$BASE_DIR/fix_chain_certificate.sh"
fi

# Run the client with proper certificates
echo "Starting Flower client $CLIENT_ID connecting to $HOST:$PORT with SSL..."
python "$BASE_DIR/run_mnist_flower_datasets.py" \
    --mode client \
    --secure \
    --client-id $CLIENT_ID \
    --host $HOST \
    --port $PORT \
    --verbose

# Check if the client started successfully
if [ $? -ne 0 ]; then
    echo "SSL connection failed. Trying insecure mode on port 18080..."
    python "$BASE_DIR/run_mnist_flower_datasets.py" \
        --mode client \
        --insecure \
        --client-id $CLIENT_ID \
        --host $HOST \
        --port 18080 \
        --verbose
fi
