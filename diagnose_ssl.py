#!/usr/bin/env python3
"""
Diagnostic tool for SSL/TLS setup in Flower Federated Learning.
This script checks certificates and attempts to establish a secure connection.
"""

import os
import sys
import socket
import ssl
from pathlib import Path
import grpc
import argparse

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def check_certificate_files():
    """Check if all required certificate files exist and are readable."""
    print_header("Checking Certificate Files")
    
    cert_dir = Path("./certs")
    required_files = [
        cert_dir / "ca/ca.pem",
        cert_dir / "ca/ca.key",
        cert_dir / "server/server.pem",
        cert_dir / "server/server.key",
        cert_dir / "server/chain.pem"
    ]
    
    all_exist = True
    for file_path in required_files:
        if file_path.exists():
            size = file_path.stat().st_size
            permissions = oct(file_path.stat().st_mode)[-3:]
            print(f"✓ {file_path} exists (Size: {size} bytes, Permissions: {permissions})")
        else:
            print(f"✗ {file_path} does not exist!")
            all_exist = False
    
    return all_exist

def read_certificate(cert_path):
    """Read and return certificate content."""
    try:
        with open(cert_path, 'rb') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {cert_path}: {e}")
        return None

def test_ssl_connection(host, port):
    """Test SSL connection to the specified host:port."""
    print_header(f"Testing SSL Connection to {host}:{port}")
    
    # Check if anything is listening on the port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    try:
        print(f"Attempting to connect to {host}:{port}...")
        result = sock.connect_ex((host, port))
        if result == 0:
            print(f"✓ Port {port} is open on {host}")
        else:
            print(f"✗ Cannot connect to {host}:{port} - No service is listening")
            return False
    except Exception as e:
        print(f"✗ Error checking port: {e}")
        return False
    finally:
        sock.close()
    
    # Try SSL connection
    try:
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        context.load_verify_locations(cafile="./certs/ca/ca.pem")
        
        print(f"Establishing SSL connection to {host}:{port}...")
        with socket.create_connection((host, port), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=host) as ssock:
                print(f"✓ SSL connection established!")
                print(f"  - SSL Version: {ssock.version()}")
                print(f"  - Cipher: {ssock.cipher()[0]}")
                cert = ssock.getpeercert()
                print(f"  - Server certificate subject: {cert.get('subject')}")
                return True
    except Exception as e:
        print(f"✗ SSL connection failed: {e}")
        return False

def test_grpc_ssl_connection(host, port):
    """Test gRPC with SSL connection."""
    print_header(f"Testing gRPC SSL Connection to {host}:{port}")
    
    try:
        # Read the CA certificate
        with open("./certs/ca/ca.pem", 'rb') as f:
            ca_cert = f.read()
        
        # Create SSL credentials
        creds = grpc.ssl_channel_credentials(ca_cert)
        
        print(f"Creating gRPC channel to {host}:{port} with SSL...")
        channel = grpc.secure_channel(f"{host}:{port}", creds)
        
        # Try to connect with a short timeout
        try:
            grpc.channel_ready_future(channel).result(timeout=5)
            print(f"✓ gRPC channel connected successfully!")
            return True
        except grpc.FutureTimeoutError:
            print(f"✗ gRPC channel connection timed out")
            return False
        finally:
            channel.close()
    except Exception as e:
        print(f"✗ gRPC SSL connection failed: {e}")
        return False

def diagnose_system():
    """Run diagnostics on the SSL/TLS setup."""
    parser = argparse.ArgumentParser(description="Diagnose SSL/TLS setup for Flower")
    parser.add_argument("--host", default="localhost", help="Host to connect to")
    parser.add_argument("--port", type=int, default=18443, help="Port to connect to")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate certificates before testing")
    args = parser.parse_args()
    
    # Print system info
    print_header("System Information")
    print(f"Python version: {sys.version}")
    print(f"grpc version: {grpc.__version__}")
    print(f"Testing connection to: {args.host}:{args.port}")
    
    # Regenerate certificates if requested
    if args.regenerate and os.path.exists("./regenerate_certificates.sh"):
        print_header("Regenerating Certificates")
        os.system("chmod +x ./regenerate_certificates.sh")
        os.system("./regenerate_certificates.sh")
    
    # Check certificate files
    files_ok = check_certificate_files()
    if not files_ok:
        print("\n⚠️  Some certificate files are missing. Please regenerate them.")
        return
    
    # Test SSL connection
    ssl_ok = test_ssl_connection(args.host, args.port)
    
    # Test gRPC SSL connection if basic SSL connection worked
    if ssl_ok:
        grpc_ok = test_grpc_ssl_connection(args.host, args.port)
    else:
        grpc_ok = False
    
    # Print summary
    print_header("Diagnosis Summary")
    print(f"Certificate files check: {'✓ PASS' if files_ok else '✗ FAIL'}")
    print(f"SSL connection check: {'✓ PASS' if ssl_ok else '✗ FAIL'}")
    print(f"gRPC SSL connection check: {'✓ PASS' if grpc_ok else '✗ FAIL'}")
    
    if not (files_ok and ssl_ok and grpc_ok):
        print("\nDiagnosis completed with errors. Please check the issues above.")
    else:
        print("\nAll diagnoses passed! The SSL/TLS setup looks good.")

if __name__ == "__main__":
    diagnose_system()
