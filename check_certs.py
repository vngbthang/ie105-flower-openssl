#!/usr/bin/env python3

import os
import sys
from pathlib import Path

print("Test script starting...")

# Print versions
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Base directory for certificates
BASE_DIR = Path(os.getcwd())
CERT_DIR = BASE_DIR / "certs"

print(f"Base directory: {BASE_DIR}")
print(f"Certificate directory: {CERT_DIR}")

# Check certificate files
server_cert_path = CERT_DIR / "server/server.pem"
server_key_path = CERT_DIR / "server/server.key"
ca_cert_path = CERT_DIR / "ca/ca.pem"

print(f"Server cert exists: {server_cert_path.exists()}")
print(f"Server key exists: {server_key_path.exists()}")
print(f"CA cert exists: {ca_cert_path.exists()}")

# Try to load certificates
try:
    with open(server_cert_path, "rb") as f:
        server_cert = f.read()
        print(f"Server cert size: {len(server_cert)} bytes")
    
    with open(server_key_path, "rb") as f:
        server_key = f.read()
        print(f"Server key size: {len(server_key)} bytes")
    
    with open(ca_cert_path, "rb") as f:
        ca_cert = f.read()
        print(f"CA cert size: {len(ca_cert)} bytes")
    
    print("All certificates loaded successfully!")
except Exception as e:
    print(f"Error loading certificates: {e}")

# Import flower
try:
    import flwr
    print(f"Flower version: {flwr.__version__}")
except ImportError:
    print("Flower not installed!")

print("Test script completed.")
