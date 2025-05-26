#!/bin/bash
# Script to run Flower server using the newer flower-superlink command
# This avoids the deprecated flwr.server.start_server() function

# Base directory
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_DIR="$BASE_DIR/certs"

# Configuration
PORT=${1:-18443}
ROUNDS=${2:-3}
MIN_CLIENTS=${3:-1}

# Make sure certificates exist
if [ -x "$BASE_DIR/regenerate_certificates.sh" ]; then
    echo "Checking and regenerating certificates if needed..."
    "$BASE_DIR/regenerate_certificates.sh"
fi

# Ensure chain file is properly formatted
if [ -x "$BASE_DIR/fix_chain_file.py" ]; then
    echo "Fixing chain file format..."
    python "$BASE_DIR/fix_chain_file.py"
fi

# Check if flower-superlink command exists
if ! command -v flower-superlink &> /dev/null; then
    echo "flower-superlink command not found. Installing flwr package..."
    pip install flwr --upgrade
fi

# Start the server using flower-superlink
echo "Starting Flower server with SuperLink on port $PORT..."
flower-superlink \
    --ssl-certfile="$CERT_DIR/server/chain.pem" \
    --ssl-keyfile="$CERT_DIR/server/server.key" \
    --ssl-ca-certfile="$CERT_DIR/ca/ca.pem" \
    --fleet-api-address="0.0.0.0:$PORT" \
    --executor-config="verbose=true min_available_clients=$MIN_CLIENTS min_fit_clients=$MIN_CLIENTS min_evaluate_clients=$MIN_CLIENTS num_rounds=$ROUNDS"
