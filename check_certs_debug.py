#!/usr/bin/env python3

"""
Script to verify TLS/SSL certificates for Flower federated learning
"""

import os
import sys
import ssl
import socket
import argparse
from pathlib import Path
import OpenSSL.crypto

# Base directory for certificates
BASE_DIR = Path(__file__).parent.absolute()
CERT_DIR = BASE_DIR / "certs"

def print_cert_info(cert_path):
    """Print detailed information about a certificate"""
    print(f"\n=== Thông tin chứng chỉ: {cert_path} ===")
    try:
        with open(cert_path, "rb") as f:
            cert_data = f.read()
            
        # Parse certificate using OpenSSL
        cert = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, cert_data)
        
        # Extract basic info
        subject = cert.get_subject()
        issuer = cert.get_issuer()
        not_before = cert.get_notBefore().decode('ascii')
        not_after = cert.get_notAfter().decode('ascii')
        serial = cert.get_serial_number()
        
        print(f"Subject: {subject.CN}")
        print(f"Issuer: {issuer.CN}")
        print(f"Serial Number: {serial}")
        print(f"Not Valid Before: {not_before}")
        print(f"Not Valid After: {not_after}")
        print(f"Certificate Size: {len(cert_data)} bytes")
        
        # Extract extensions
        ext_count = cert.get_extension_count()
        print(f"\nExtensions ({ext_count}):")
        for i in range(ext_count):
            ext = cert.get_extension(i)
            print(f"  {ext.get_short_name().decode('utf-8')}: {str(ext)}")
            
        return True
    except Exception as e:
        print(f"Lỗi khi đọc chứng chỉ: {str(e)}")
        return False
        
def verify_cert_chain(ca_path, cert_path):
    """Verify a certificate against a CA certificate"""
    print(f"\n=== Xác minh chuỗi chứng chỉ: {cert_path} với {ca_path} ===")
    try:
        # Create a certificate store and add CA cert
        store = OpenSSL.crypto.X509Store()
        
        # Load the CA cert
        with open(ca_path, "rb") as f:
            ca_cert_data = f.read()
        ca_cert = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, ca_cert_data)
        store.add_cert(ca_cert)
        
        # Load the certificate to verify
        with open(cert_path, "rb") as f:
            cert_data = f.read()
        cert = OpenSSL.crypto.load_certificate(OpenSSL.crypto.FILETYPE_PEM, cert_data)
        
        # Create a certificate context
        store_ctx = OpenSSL.crypto.X509StoreContext(store, cert)
        
        # Verify the certificate
        result = store_ctx.verify_certificate()
        if result is None:
            print("✅ Xác minh thành công!")
            return True
    except OpenSSL.crypto.X509StoreContextError as e:
        print(f"❌ Xác minh thất bại: {e}")
    except Exception as e:
        print(f"Lỗi khi xác minh: {str(e)}")
    
    return False

def test_ssl_connection(server_host="localhost", server_port=8443, ca_path=None):
    """Test SSL connection to a server"""
    print(f"\n=== Kiểm tra kết nối SSL đến {server_host}:{server_port} ===")
    try:
        # Create SSL context
        context = ssl.create_default_context()
        if ca_path:
            context.load_verify_locations(ca_path)
            print(f"Đã tải chứng chỉ CA từ {ca_path}")
            
        # Create a socket
        with socket.create_connection((server_host, server_port)) as sock:
            with context.wrap_socket(sock, server_hostname=server_host) as ssock:
                print(f"✅ Kết nối SSL thành công!")
                print(f"Phiên bản SSL: {ssock.version()}")
                print(f"Cipher: {ssock.cipher()}")
                cert = ssock.getpeercert()
                print(f"Thông tin chứng chỉ từ server: {cert}")
                return True
    except ConnectionRefusedError:
        print(f"❌ Kết nối bị từ chối. Đảm bảo server đang chạy ở cổng {server_port}")
    except ssl.SSLCertVerificationError as e:
        print(f"❌ Lỗi xác minh chứng chỉ: {e}")
    except ssl.SSLError as e:
        print(f"❌ Lỗi SSL: {e}")
    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
    
    return False

def check_all_certs():
    """Check all certificates in the certs directory"""
    print("\n=== Kiểm tra tất cả chứng chỉ ===")
    
    ca_path = CERT_DIR / "ca/ca.pem"
    server_cert_path = CERT_DIR / "server/server.pem"
    server_key_path = CERT_DIR / "server/server.key"
    client_cert_path = CERT_DIR / "client/client.pem"
    client_key_path = CERT_DIR / "client/client.key"
    
    # Verify CA certificate exists and is valid
    if not ca_path.exists():
        print(f"❌ Không tìm thấy chứng chỉ CA: {ca_path}")
        return False
        
    # Check server certificate
    if not server_cert_path.exists():
        print(f"❌ Không tìm thấy chứng chỉ server: {server_cert_path}")
        return False
    
    # Check server key
    if not server_key_path.exists():
        print(f"❌ Không tìm thấy khóa server: {server_key_path}")
        return False
        
    # Print CA certificate info
    print_cert_info(ca_path)
    
    # Print server certificate info
    print_cert_info(server_cert_path)
    
    # Verify server certificate against CA
    verify_cert_chain(ca_path, server_cert_path)
    
    # If client certificate exists, also verify it
    if client_cert_path.exists():
        print_cert_info(client_cert_path)
        verify_cert_chain(ca_path, client_cert_path)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Kiểm tra và xác minh chứng chỉ TLS/SSL")
    parser.add_argument("--check-all", action="store_true", help="Kiểm tra tất cả chứng chỉ")
    parser.add_argument("--cert-path", type=str, help="Đường dẫn đến chứng chỉ cần kiểm tra")
    parser.add_argument("--test-connection", action="store_true", help="Kiểm tra kết nối đến server")
    parser.add_argument("--port", type=int, default=8443, help="Cổng SSL của server")
    parser.add_argument("--host", type=str, default="localhost", help="Host của server")
    
    args = parser.parse_args()
    
    if args.check_all:
        check_all_certs()
    elif args.cert_path:
        print_cert_info(args.cert_path)
    elif args.test_connection:
        ca_path = CERT_DIR / "ca/ca.pem"
        if ca_path.exists():
            test_ssl_connection(args.host, args.port, ca_path)
        else:
            print(f"❌ Không tìm thấy chứng chỉ CA: {ca_path}")
            test_ssl_connection(args.host, args.port)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
