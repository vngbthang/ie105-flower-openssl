#!/bin/bash
# Script to run server with secure SSL/TLS

# Get the project base directory
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CERT_DIR="$BASE_DIR/certs"
PORT=${1:-18443}
SERVER_APP_PORT=9092  # Port for server app API - changed to 9092 to avoid conflict

# Check and regenerate certificates if needed
if [ -x "$BASE_DIR/regenerate_certificates.sh" ]; then
    echo "Checking and regenerating certificates if needed..."
    "$BASE_DIR/regenerate_certificates.sh"
fi

# Fix the chain certificate to ensure it's properly formatted
if [ -x "$BASE_DIR/fix_chain_certificate.sh" ]; then
    echo "Ensuring the chain certificate is properly formatted..."
    "$BASE_DIR/fix_chain_certificate.sh"
fi

# SuperLink is the recommended way to run Flower servers in newer versions
echo "Starting Flower SuperLink server on port $PORT with secure TLS..."

# Kiểm tra xem có cổng đã được sử dụng không
echo "Using fleet API port: $PORT"
echo "Using server app API port: $SERVER_APP_PORT"

# Kiểm tra port trước khi khởi động
nc -z localhost $PORT &> /dev/null
if [ $? -eq 0 ]; then
    echo "WARNING: Port $PORT is already in use. Trying alternative port..."
    PORT=$((PORT + 2000))
    echo "Using alternative port: Fleet API port: $PORT"
fi

nc -z localhost $SERVER_APP_PORT &> /dev/null
if [ $? -eq 0 ]; then
    echo "WARNING: Server app port $SERVER_APP_PORT is already in use. Trying alternative port..."
    SERVER_APP_PORT=$((SERVER_APP_PORT + 1000))
    echo "Using alternative server app port: $SERVER_APP_PORT"
fi

# Set Python path to include the project root
export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

# SuperLink in modern Flower already includes the server functionality
# Instead of starting a separate server app, we directly configure SuperLink with
# our app module path from server/server_app_new.py

# Set Python path to ensure imports work correctly
export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

# Start SuperLink directly, with our custom MNIST executor
echo "Starting Flower SuperLink server on port $PORT"
flower-superlink \
    --ssl-certfile="$CERT_DIR/server/server.pem" \
    --ssl-keyfile="$CERT_DIR/server/server.key" \
    --ssl-ca-certfile="$CERT_DIR/ca/ca.pem" \
    --fleet-api-address "[::]:$PORT" \
    --executor="server.executor:executor" \
    --executor-config="verbose=true min_available_clients=1 min_fit_clients=1 min_evaluate_clients=1 num_rounds=3"

echo "Flower SuperLink server has terminated."
