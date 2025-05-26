#!/bin/bash
# Script to run client with secure SSL/TLS using Flower SuperNode

# Get the project base directory
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CERT_DIR="$BASE_DIR/certs"
HOST=${1:-localhost}
PORT=${2:-18443}
CLIENT_ID=${3:-0}

# Check and regenerate certificates if needed
if [ -x "$BASE_DIR/regenerate_certificates.sh" ]; then
    echo "Checking and regenerating certificates if needed..."
    "$BASE_DIR/regenerate_certificates.sh"
fi

# Set Python path to include the project root
export PYTHONPATH="$BASE_DIR:$PYTHONPATH"

# Set FLOWER_CLIENT_ID environment variable
export FLOWER_CLIENT_ID="$CLIENT_ID"

# Config file path
NODE_CONFIG_FILE="$BASE_DIR/client/node_config.json"

echo "Starting Flower SuperNode client with ID $CLIENT_ID connecting to $HOST:$PORT..."
echo "Using node config from $NODE_CONFIG_FILE"

# Start the client with secure connection
flower-supernode \
    --superlink="$HOST:$PORT" \
    --root-certificates="$CERT_DIR/ca/ca.pem" \
    --node-config="$NODE_CONFIG_FILE"

echo "Flower SuperNode client has terminated."