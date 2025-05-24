#!/bin/bash

echo "===== FLOWER TLS/SSL ENCRYPTION TEST ====="

WORKDIR=$(dirname "$0")
cd "$WORKDIR" || exit 1

# Fix permissions and regenerate certs if needed
echo "1. Checking and fixing SSL certificates..."
sudo bash fix_ssl_errors.sh

# Start the packet capture in a new terminal
echo ""
echo "2. Starting packet capture in a separate process..."
gnome-terminal -- bash -c "cd '$WORKDIR' && sudo bash start_packet_capture.sh; exec bash" 2>/dev/null || \
xterm -e "cd '$WORKDIR' && sudo bash start_packet_capture.sh" 2>/dev/null || \
konsole --workdir "$WORKDIR" -e "sudo bash start_packet_capture.sh" 2>/dev/null || \
sudo bash start_packet_capture.sh &

# Wait a moment for tcpdump to start
sleep 3

# Run the encryption test
echo ""
echo "3. Running encryption test..."
python3 test_encryption.py

echo ""
echo "Test completed! Check the encryption_test_results.log file for details."
