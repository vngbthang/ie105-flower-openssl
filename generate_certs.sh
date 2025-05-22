#!/bin/bash
set -e

# Tạo CA
mkdir -p certs/ca certs/server certs/client
openssl genrsa -out certs/ca/ca.key 4096
openssl req -x509 -new -nodes -key certs/ca/ca.key -sha256 -days 3650 -out certs/ca/ca.pem -subj "/CN=Flower CA"

# Server
openssl genrsa -out certs/server/server.key 4096
openssl req -new -key certs/server/server.key -out certs/server/server.csr -subj "/CN=localhost"
openssl x509 -req -in certs/server/server.csr -CA certs/ca/ca.pem -CAkey certs/ca/ca.key -CAcreateserial -out certs/server/server.pem -days 3650 -sha256

# Client
openssl genrsa -out certs/client/client.key 4096
openssl req -new -key certs/client/client.key -out certs/client/client.csr -subj "/CN=client"
openssl x509 -req -in certs/client/client.csr -CA certs/ca/ca.pem -CAkey certs/ca/ca.key -CAcreateserial -out certs/client/client.pem -days 3650 -sha256

echo "Chứng chỉ đã tạo xong ở thư mục certs/"
