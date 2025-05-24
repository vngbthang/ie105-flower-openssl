#!/bin/bash

BASE_DIR=$(dirname "$0")
cd "$BASE_DIR" || exit 1

echo "===== PREPARING TO RUN FLOWER ENCRYPTION TEST ====="

# 1. Check if certificates exist
if [ ! -f "certs/ca/ca.pem" ] || [ ! -f "certs/server/server.key" ] || [ ! -f "certs/server/server.pem" ]; then
    echo "Certificates missing, regenerating..."
    ./generate_certs.sh
fi

# 2. Set up packet capture
echo "Setting up packet capture..."
CAPTURE_FILE="/tmp/flower_secure_traffic.pcap"
touch "$CAPTURE_FILE"
chmod 666 "$CAPTURE_FILE"

# Start tcpdump with sudo
echo "Starting packet capture on port 8443..."
sudo tcpdump -i lo -w "$CAPTURE_FILE" port 8443 &
TCPDUMP_PID=$!

# Give tcpdump time to start
sleep 2

echo "Now run the encryption test with:"
echo "python3 test_encryption.py"
echo ""
echo "When done, press Ctrl+C here to stop packet capture"

# Wait for Ctrl+C
trap 'echo "Stopping packet capture..."; sudo kill -INT $TCPDUMP_PID; exit 0' INT
wait $TCPDUMP_PID
