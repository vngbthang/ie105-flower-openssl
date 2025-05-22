#!/bin/bash

# Get the base directory
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Start the Flower client using the flower-supernode CLI tool with secure mTLS
echo "Starting Flower supernode client with secure mTLS connection..."
cd "${BASE_DIR}"

# Set PYTHONPATH to include the current directory for module imports
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}"

# Try to use the flower-supernode CLI with the appropriate parameters
if command -v flower-supernode &> /dev/null; then
    echo "Using flower-supernode CLI..."
    flower-supernode \
        --superlink='localhost:8443' \
        --root-certificates="${BASE_DIR}/certs/ca/ca.pem"
else
    echo "flower-supernode command not found. Falling back to direct Python execution..."
    python "${BASE_DIR}/client/client_supernode.py"
fi
