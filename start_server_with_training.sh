#!/bin/bash
# Script to run server with secure SSL/TLS and auto-start training

# Get the project base directory
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CERT_DIR="$BASE_DIR/certs"
PORT=${1:-18443}

# Check and regenerate certificates if needed
if [ -x "$BASE_DIR/regenerate_certificates.sh" ]; then
    echo "Checking and regenerating certificates if needed..."
    "$BASE_DIR/regenerate_certificates.sh"
fi

# Fix chain certificate format
if [ -x "$BASE_DIR/fix_chain_certificate.sh" ]; then
    echo "Ensuring the chain certificate is properly formatted..."
    "$BASE_DIR/fix_chain_certificate.sh"
fi

# Set Python path to include the project root
export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

echo "Starting Flower SuperLink server on port $PORT with secure TLS..."
echo "Using custom executor from server.executor:executor"

# Start server in background
flower-superlink \
    --ssl-certfile="$CERT_DIR/server/server.pem" \
    --ssl-keyfile="$CERT_DIR/server/server.key" \
    --ssl-ca-certfile="$CERT_DIR/ca/ca.pem" \
    --fleet-api-address "[::]:$PORT" \
    --executor="server.executor:executor" \
    --executor-config="verbose=true min_available_clients=1 min_fit_clients=1 min_evaluate_clients=1 num_rounds=3" &

# Save the server process ID
SERVER_PID=$!

echo "Server process started with PID $SERVER_PID"

# Wait for the server to start up
sleep 10
echo "Starting the training run..."

# Manually call start_run through the Python API
python - << EOF
import sys
import logging
import importlib.util
import time
import threading

# Import the executor module
sys.path.append("$BASE_DIR")
from server.executor import executor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("flower-mnist-trigger")

# Give the server time to initialize
time.sleep(2)

logger.info("Triggering start_run via Python API")
try:
    # Set up a separate thread to call start_run
    def trigger_run():
        try:
            logger.info("Calling start_run method")
            executor.start_run()
            logger.info("start_run method called successfully")
        except Exception as e:
            logger.error(f"Error triggering start_run: {e}")

    # Start the thread
    thread = threading.Thread(target=trigger_run)
    thread.daemon = True
    thread.start()
    
    # Wait for the thread to complete
    thread.join(timeout=5)
    logger.info("Training should now be starting")
except Exception as e:
    logger.error(f"Error: {e}")
EOF

# Wait for the server process
wait $SERVER_PID
echo "Flower SuperLink server has terminated."
