#!/usr/bin/env python3

"""
Quick script to check SSL certificates and show detailed errors
"""

import os
import sys
from pathlib import Path
import ssl
import subprocess

# Base directory for certificates
BASE_DIR = Path(__file__).parent.absolute()
CERT_DIR = BASE_DIR / "certs"

def check_certificate_files():
    """Check if certificate files exist and print their info."""
    print("===== Certificate File Check =====")
    
    # Check directories
    for dir_name in ["ca", "server", "client"]:
        dir_path = CERT_DIR / dir_name
        if not dir_path.exists():
            print(f"❌ Directory missing: {dir_path}")
            return False
        print(f"✓ Directory exists: {dir_path}")
    
    # Check CA files
    ca_cert = CERT_DIR / "ca/ca.pem"
    ca_key = CERT_DIR / "ca/ca.key"
    
    if not ca_cert.exists():
        print(f"❌ CA certificate missing: {ca_cert}")
        return False
    
    if not ca_key.exists():
        print(f"❌ CA key missing: {ca_key}")
        return False
    
    # Check server files
    server_cert = CERT_DIR / "server/server.pem"
    server_key = CERT_DIR / "server/server.key"
    
    if not server_cert.exists():
        print(f"❌ Server certificate missing: {server_cert}")
        return False
    
    if not server_key.exists():
        print(f"❌ Server key missing: {server_key}")
        return False
    
    # Check client files
    client_cert = CERT_DIR / "client/client.pem"
    client_key = CERT_DIR / "client/client.key"
    
    if not client_cert.exists():
        print(f"❌ Client certificate missing: {client_cert}")
        return False
    
    if not client_key.exists():
        print(f"❌ Client key missing: {client_key}")
        return False
    
    print("✅ All certificate files exist")
    return True

def show_certificate_info():
    """Display information about the certificates using OpenSSL."""
    print("\n===== Certificate Information =====")
    
    # Check CA certificate
    print("\nCA Certificate Info:")
    subprocess.run(["openssl", "x509", "-in", str(CERT_DIR / "ca/ca.pem"), 
                   "-text", "-noout"], check=False)
    
    # Check server certificate
    print("\nServer Certificate Info:")
    subprocess.run(["openssl", "x509", "-in", str(CERT_DIR / "server/server.pem"), 
                   "-text", "-noout"], check=False)
    
    # Verify server certificate against CA
    print("\nVerifying server certificate against CA:")
    subprocess.run(["openssl", "verify", "-CAfile", str(CERT_DIR / "ca/ca.pem"),
                   str(CERT_DIR / "server/server.pem")], check=False)
    
    # Check client certificate
    print("\nClient Certificate Info:")
    subprocess.run(["openssl", "x509", "-in", str(CERT_DIR / "client/client.pem"), 
                   "-text", "-noout"], check=False)
    
    # Verify client certificate against CA
    print("\nVerifying client certificate against CA:")
    subprocess.run(["openssl", "verify", "-CAfile", str(CERT_DIR / "ca/ca.pem"),
                   str(CERT_DIR / "client/client.pem")], check=False)

def check_subject_alt_names():
    """Check if server certificate has appropriate Subject Alternative Names."""
    print("\n===== Checking Subject Alternative Names =====")
    
    # Extract SAN from server certificate
    try:
        result = subprocess.run(
            ["openssl", "x509", "-in", str(CERT_DIR / "server/server.pem"), 
             "-text", "-noout"], 
            check=True, capture_output=True, text=True
        )
        
        cert_text = result.stdout
        
        if "DNS:localhost" in cert_text:
            print("✅ Server certificate includes DNS:localhost in Subject Alternative Name")
        else:
            print("❌ Server certificate missing DNS:localhost in Subject Alternative Name")
            print("This is required for gRPC with TLS to work correctly!")
        
        if "IP Address:127.0.0.1" in cert_text:
            print("✅ Server certificate includes IP:127.0.0.1 in Subject Alternative Name")
        else:
            print("❌ Server certificate missing IP:127.0.0.1 in Subject Alternative Name")
        
    except subprocess.CalledProcessError as e:
        print(f"Error checking certificate: {e}")
        return False

if __name__ == "__main__":
    if check_certificate_files():
        show_certificate_info()
        check_subject_alt_names()
    else:
        print("\n❌ Certificate files missing. Please run ./generate_certs.sh first.")
        sys.exit(1)
