#!/bin/bash
# Script để kiểm tra và tạo lại các chứng chỉ SSL nếu cần thiết

set -e  # Dừng nếu có bất kỳ lệnh nào bị lỗi

CERT_DIR="./certs"
SERVER_CERT="$CERT_DIR/server/server.pem"
SERVER_KEY="$CERT_DIR/server/server.key" 
CA_CERT="$CERT_DIR/ca/ca.pem"
CA_KEY="$CERT_DIR/ca/ca.key"
CHAIN_CERT="$CERT_DIR/server/chain.pem"
PKCS12_CERT="$CERT_DIR/combined/server.pfx"

echo "============================================"
echo "Kiểm tra và tạo lại chứng chỉ SSL/TLS"
echo "============================================"

# Kiểm tra và tạo thư mục nếu không tồn tại
mkdir -p "$CERT_DIR/server" "$CERT_DIR/ca" "$CERT_DIR/combined"

# Kiểm tra xem chứng chỉ đã tồn tại chưa
check_certs() {
    local missing=0
    
    if [ ! -f "$SERVER_CERT" ]; then
        echo "[THIẾU] Server certificate: $SERVER_CERT"
        missing=1
    else
        echo "[OK] Server certificate tồn tại"
    fi
    
    if [ ! -f "$SERVER_KEY" ]; then
        echo "[THIẾU] Server key: $SERVER_KEY"
        missing=1
    else
        echo "[OK] Server key tồn tại"
    fi
    
    if [ ! -f "$CA_CERT" ]; then
        echo "[THIẾU] CA certificate: $CA_CERT"
        missing=1
    else
        echo "[OK] CA certificate tồn tại"
    fi
    
    if [ ! -f "$CA_KEY" ]; then
        echo "[THIẾU] CA key: $CA_KEY"
        missing=1
    else
        echo "[OK] CA key tồn tại"
    fi
    
    return $missing
}

# Tạo certificate chain
create_chain_cert() {
    echo "Tạo chain certificate từ server và CA certificates..."
    cat "$SERVER_CERT" "$CA_CERT" > "$CHAIN_CERT"
    chmod 644 "$CHAIN_CERT"
    echo "Đã tạo chain certificate: $CHAIN_CERT"
    
    # Kiểm tra chain cert
    openssl verify -CAfile "$CA_CERT" "$SERVER_CERT"
}

# Tạo PKCS#12 certificate (có thể thích hợp hơn cho một số thực thi)
create_pkcs12_cert() {
    echo "Tạo PKCS#12 certificate..."
    openssl pkcs12 -export -out "$PKCS12_CERT" \
        -inkey "$SERVER_KEY" -in "$SERVER_CERT" \
        -certfile "$CA_CERT" -passout pass:flower
    chmod 600 "$PKCS12_CERT"
    echo "Đã tạo PKCS#12 certificate: $PKCS12_CERT"
}

# Tạo chứng chỉ CA (Certificate Authority)
create_ca_cert() {
    echo "Tạo CA certificate và key..."
    openssl req -x509 -newkey rsa:4096 \
        -keyout "$CA_KEY" -out "$CA_CERT" \
        -days 3650 -nodes -subj "/CN=Flower CA"
    chmod 600 "$CA_KEY"
    chmod 644 "$CA_CERT"
    echo "Đã tạo CA certificate và key"
}

# Tạo chứng chỉ server được ký bởi CA
create_server_cert() {
    echo "Tạo server certificate và key..."
    
    # Tạo server.cnf nếu không tồn tại
    if [ ! -f "$CERT_DIR/server/server.cnf" ]; then
        echo "Tạo file cấu hình server.cnf..."
        cat > "$CERT_DIR/server/server.cnf" << EOF
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
    fi
    
    # Tạo server key và CSR (Certificate Signing Request)
    openssl req -newkey rsa:4096 -nodes \
        -keyout "$SERVER_KEY" -out "$CERT_DIR/server/server.csr" \
        -subj "/CN=localhost" -config "$CERT_DIR/server/server.cnf"
    
    # Ký CSR với CA certificate để tạo server certificate
    openssl x509 -req -in "$CERT_DIR/server/server.csr" \
        -out "$SERVER_CERT" -CA "$CA_CERT" -CAkey "$CA_KEY" \
        -CAcreateserial -days 3650 \
        -extensions v3_req -extfile "$CERT_DIR/server/server.cnf"
    
    # Thiết lập quyền phù hợp
    chmod 600 "$SERVER_KEY"
    chmod 644 "$SERVER_CERT"
    
    echo "Đã tạo server certificate và key"
}

# Kiểm tra và khởi tạo chứng chỉ nếu cần
check_certs
if [ $? -ne 0 ]; then
    echo "Một hoặc nhiều chứng chỉ thiếu, đang tạo mới..."
    create_ca_cert
    create_server_cert
fi

# Luôn tạo lại chain cert và PKCS12 để đảm bảo chúng là mới nhất
create_chain_cert
create_pkcs12_cert

# Kiểm tra kết quả
echo "============================================"
echo "Kiểm tra kết quả:"
echo "============================================"
echo "1. Server certificate information:"
openssl x509 -in "$SERVER_CERT" -text -noout | grep "Subject\\|Issuer\\|Validity"

echo "2. CA certificate information:"
openssl x509 -in "$CA_CERT" -text -noout | grep "Subject\\|Issuer\\|Validity"

echo "3. Xác thực server certificate với CA:"
openssl verify -CAfile "$CA_CERT" "$SERVER_CERT" && echo "Xác thực thành công!"

echo "============================================"
echo "Chứng chỉ SSL/TLS đã sẵn sàng được sử dụng."
echo "============================================"

# Thông báo cấu hình khởi động
echo "Gợi ý khởi động server:"
echo "python run_mnist_flower_datasets.py --mode server --secure --port 18443"
echo 
echo "Gợi ý khởi động client (sau khi server đã chạy):"
echo "python run_mnist_flower_datasets.py --mode client --secure --host localhost --port 18443"
