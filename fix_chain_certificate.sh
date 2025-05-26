#!/bin/bash
# Script to fix SSL chain certificate for Flower

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_DIR="$BASE_DIR/certs"
SERVER_CERT="$CERT_DIR/server/server.pem"
CA_CERT="$CERT_DIR/ca/ca.pem"
CHAIN_CERT="$CERT_DIR/server/chain.pem"

echo "Creating proper SSL chain certificate for Flower..."

# Check if the certificate files exist
if [ ! -f "$SERVER_CERT" ]; then
    echo "ERROR: Server certificate not found: $SERVER_CERT"
    exit 1
fi

if [ ! -f "$CA_CERT" ]; then
    echo "ERROR: CA certificate not found: $CA_CERT"
    exit 1
fi

# Backup the old chain certificate if it exists
if [ -f "$CHAIN_CERT" ]; then
    cp "$CHAIN_CERT" "${CHAIN_CERT}.bak"
    echo "Backed up existing chain certificate to ${CHAIN_CERT}.bak"
fi

# Create the chain certificate by concatenating server and CA certificates
cat "$SERVER_CERT" "$CA_CERT" > "$CHAIN_CERT"

# Set proper permissions
chmod 644 "$CHAIN_CERT"

echo "Chain certificate created successfully at $CHAIN_CERT"

# Verify the chain certificate
echo "Verifying the chain certificate..."
openssl verify -CAfile "$CA_CERT" "$SERVER_CERT" && echo "Server certificate verified successfully with CA!"

# Print certificate info
echo "Chain certificate information:"
openssl x509 -in "$CHAIN_CERT" -text -noout | grep "Subject\|Issuer\|Validity" || echo "Could not read chain certificate"

echo "Done."
