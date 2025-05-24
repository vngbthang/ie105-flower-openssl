#!/bin/bash
set -e

echo "===== Tạo chứng chỉ SSL/TLS cho Flower ====="

# Tạo CA
echo "1. Tạo Certificate Authority (CA)..."
mkdir -p certs/ca certs/server certs/client
openssl genrsa -out certs/ca/ca.key 4096
openssl req -x509 -new -nodes -key certs/ca/ca.key -sha256 -days 3650 -out certs/ca/ca.pem -subj "/CN=Flower CA"

# Server - with proper Subject Alternative Names (SAN) for localhost
echo "2. Tạo chứng chỉ server với SAN cho localhost..."
openssl genrsa -out certs/server/server.key 4096

# Create a config file for SAN
cat > certs/server/server.cnf << EOF
[req]
distinguished_name = req_distinguished_name
req_extensions = v3_req

[req_distinguished_name]

[v3_req]
subjectAltName = @alt_names

[alt_names]
DNS.1 = localhost
IP.1 = 127.0.0.1
IP.2 = ::1
EOF

# Create CSR with SAN
openssl req -new -key certs/server/server.key -out certs/server/server.csr \
  -subj "/CN=localhost" -config certs/server/server.cnf -extensions v3_req

# Sign the server certificate with our CA
openssl x509 -req -in certs/server/server.csr \
  -CA certs/ca/ca.pem -CAkey certs/ca/ca.key -CAcreateserial \
  -out certs/server/server.pem -days 3650 -sha256 \
  -extensions v3_req -extfile certs/server/server.cnf

# Client
openssl genrsa -out certs/client/client.key 4096
openssl req -new -key certs/client/client.key -out certs/client/client.csr -subj "/CN=client"
openssl x509 -req -in certs/client/client.csr -CA certs/ca/ca.pem -CAkey certs/ca/ca.key -CAcreateserial -out certs/client/client.pem -days 3650 -sha256

echo "Chứng chỉ đã tạo xong ở thư mục certs/"
