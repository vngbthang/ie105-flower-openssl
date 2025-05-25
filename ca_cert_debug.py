#!/usr/bin/env python3
import os
import ssl
import socket

def verify_cert(ca_cert_path):
    print(f"Verifying CA certificate: {ca_cert_path}")
    
    if not os.path.exists(ca_cert_path):
        print(f"CA certificate file does not exist: {ca_cert_path}")
        return False
    
    try:
        # Create an SSL context with the CA certificate
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.load_verify_locations(ca_cert_path)
        print("Successfully loaded CA certificate")
        return True
    except Exception as e:
        print(f"Error loading CA certificate: {e}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        verify_cert(sys.argv[1])
    else:
        print("Usage: python3 ca_cert_debug.py <ca_cert_path>")
