#!/bin/bash
# Script to run server with secure SSL/TLS using Flower SuperLink

# Get the project base directory
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CERT_DIR="$BASE_DIR/certs"
PORT=${1:-18443}

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

# Set Python path to include the project root
export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

# Check if port is already in use
nc -z localhost $PORT &> /dev/null
if [ $? -eq 0 ]; then
    echo "WARNING: Port $PORT is already in use. Trying alternative port..."
    PORT=$((PORT + 2000))
    echo "Using alternative port: Fleet API port: $PORT"
fi

echo "Starting Flower SuperLink server on port $PORT with secure TLS..."
echo "Using fixed executor from server.executor_fixed:executor"

# Start SuperLink with the correct parameters for 1.18.0, with auto registration
flower-superlink \
    --ssl-certfile="$CERT_DIR/server/server.pem" \
    --ssl-keyfile="$CERT_DIR/server/server.key" \
    --ssl-ca-certfile="$CERT_DIR/ca/ca.pem" \
    --fleet-api-address "[::]:$PORT" \
    --executor="server.executor_fixed:executor" \
    --executor-config="verbose=true min_available_clients=1 min_fit_clients=1 min_evaluate_clients=1 num_rounds=3 auto_register_clients=true"

echo "Flower SuperLink server has terminated."