#!/bin/bash
# Script to run server in insecure mode

# Get the project base directory
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PORT=${1:-18080}

# SuperLink is the recommended way to run Flower servers in newer versions
echo "Starting Flower SuperLink server on port $PORT without security..."

# Kiểm tra xem có cổng đã được sử dụng không
echo "Using fleet API port: $PORT"

# Kiểm tra port trước khi khởi động
nc -z localhost $PORT &> /dev/null
if [ $? -eq 0 ]; then
    echo "WARNING: Port $PORT is already in use. Trying alternative port..."
    PORT=$((PORT + 2000))
    echo "Using alternative port: Fleet API port: $PORT"
fi

# Khởi động SuperLink không có SSL/TLS
flower-superlink \
    --insecure \
    --fleet-api-address=[::]:$PORT
