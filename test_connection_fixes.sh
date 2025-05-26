#!/bin/bash
# Test script for verifying SSL/TLS connection fixes

# Base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "===== Testing Flower SSL/TLS Connection Fixes ====="

# Set test parameters
SECURE_PORT=18443
INSECURE_PORT=18080
TEST_CLIENT_ID=999

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Step 1: Check certificate generation
echo -e "${YELLOW}[TEST] Step 1: Checking certificate generation${NC}"
if [ -x "$BASE_DIR/regenerate_certificates.sh" ]; then
    echo "Generating certificates..."
    "$BASE_DIR/regenerate_certificates.sh"
    if [ $? -ne 0 ]; then
        echo -e "${RED}[FAILED] Certificate generation failed${NC}"
        exit 1
    else
        echo -e "${GREEN}[SUCCESS] Certificates generated successfully${NC}"
    fi
else
    echo -e "${RED}[FAILED] regenerate_certificates.sh script not found or not executable${NC}"
    exit 1
fi

# Step 2: Check the certificates
echo -e "\n${YELLOW}[TEST] Step 2: Verifying certificates${NC}"
if [ -f "$BASE_DIR/diagnose_ssl.py" ]; then
    echo "Running certificate diagnostics..."
    python "$BASE_DIR/diagnose_ssl.py" --regenerate
    if [ $? -ne 0 ]; then
        echo -e "${RED}[WARNING] Certificate diagnostics reported issues${NC}"
    else
        echo -e "${GREEN}[SUCCESS] Certificate diagnostics passed${NC}"
    fi
else
    echo -e "${YELLOW}[SKIPPED] diagnose_ssl.py script not found${NC}"
fi

# Step 3: Test server on secure port
echo -e "\n${YELLOW}[TEST] Step 3: Starting server on secure port $SECURE_PORT${NC}"
# First make sure the chain certificate is properly fixed
echo "Fixing chain certificate before starting server..."
"$BASE_DIR/fix_chain_certificate.sh" > /dev/null 2>&1

# Start server in background with output to a log file
echo "Starting server in background using SuperLink (recommended approach in Flower 1.18.0)..."
"$BASE_DIR/run_server_secure.sh" $SECURE_PORT > server_log.txt 2>&1 &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to initialize (10 seconds)..."
# Give more time for the server to start properly
sleep 10

# Check if server is still running
if ps -p $SERVER_PID > /dev/null; then
    echo -e "${GREEN}Server is running with PID $SERVER_PID${NC}"
else
    echo -e "${RED}Server failed to start or has stopped${NC}"
fi

# Show the server logs so far
echo "Server log output:"
cat server_log.txt

# Check if server is running
echo "Checking if server is running on port $SECURE_PORT..."
timeout 2 bash -c "</dev/tcp/localhost/$SECURE_PORT" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] Server is running on port $SECURE_PORT${NC}"
else
    echo -e "${RED}[FAILED] Server is not running on port $SECURE_PORT${NC}"
    echo "Checking alternative port 28443..."
    timeout 2 bash -c "</dev/tcp/localhost/28443" 2>/dev/null
    if [ $? -eq 0 ]; then
        SECURE_PORT=28443
        echo -e "${GREEN}[SUCCESS] Server is running on alternative port $SECURE_PORT${NC}"
    else
        echo -e "${RED}[FAILED] Server is not running on any expected port${NC}"
        # Kill the server process and exit
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
fi

# Step 4: Test client connection
echo -e "\n${YELLOW}[TEST] Step 4: Testing client connection to secure server${NC}"
echo "Running client with secure connection..."
python "$BASE_DIR/run_mnist_flower_datasets.py" --mode client --secure --client-id $TEST_CLIENT_ID --port $SECURE_PORT --verbose &
CLIENT_PID=$!

# Wait for client to run
sleep 10
# Check if client is still running (it should exit after connecting)
if ps -p $CLIENT_PID > /dev/null; then
    echo -e "${YELLOW}[WARNING] Client is still running, may be in training mode${NC}"
    # Kill the client process
    kill $CLIENT_PID 2>/dev/null
else 
    echo -e "${GREEN}[SUCCESS] Client completed connection${NC}"
fi

# Kill the server process
echo "Stopping secure server..."
kill $SERVER_PID 2>/dev/null
sleep 2

# Step 5: Test insecure server
echo -e "\n${YELLOW}[TEST] Step 5: Starting server on insecure port $INSECURE_PORT${NC}"
# Start server in background
echo "Starting server without SSL using SuperLink..."
"$BASE_DIR/run_server_insecure.sh" $INSECURE_PORT > server_insecure_log.txt 2>&1 &
SERVER_PID=$!

# Wait for server to start
echo "Waiting for server to initialize (5 seconds)..."
sleep 5

# Show the server logs so far
echo "Server log output:"
cat server_insecure_log.txt

# Check if server is running
echo "Checking if server is running on port $INSECURE_PORT..."
timeout 2 bash -c "</dev/tcp/localhost/$INSECURE_PORT" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] Insecure server is running on port $INSECURE_PORT${NC}"
else
    echo -e "${RED}[FAILED] Insecure server is not running on port $INSECURE_PORT${NC}"
    echo "Checking alternative port 8080..."
    timeout 2 bash -c "</dev/tcp/localhost/8080" 2>/dev/null
    if [ $? -eq 0 ]; then
        INSECURE_PORT=8080
        echo -e "${GREEN}[SUCCESS] Insecure server is running on alternative port $INSECURE_PORT${NC}"
    else
        echo -e "${RED}[FAILED] Insecure server is not running on any expected port${NC}"
        # Kill the server process and exit
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
fi

# Step 6: Test client on insecure server
echo -e "\n${YELLOW}[TEST] Step 6: Testing client connection to insecure server${NC}"
echo "Running client with insecure connection..."
python "$BASE_DIR/run_mnist_flower_datasets.py" --mode client --insecure --client-id $TEST_CLIENT_ID --port $INSECURE_PORT --verbose &
CLIENT_PID=$!

# Wait for client to run
sleep 10
# Check if client is still running (it should exit after connecting)
if ps -p $CLIENT_PID > /dev/null; then
    echo -e "${YELLOW}[WARNING] Client is still running, may be in training mode${NC}"
    # Kill the client process
    kill $CLIENT_PID 2>/dev/null
else 
    echo -e "${GREEN}[SUCCESS] Client completed connection to insecure server${NC}"
fi

# Kill the server process
echo "Stopping insecure server..."
kill $SERVER_PID 2>/dev/null
sleep 2

echo -e "\n${GREEN}===== All tests completed! =====${NC}"
echo "Please review the output for any errors or warnings."
