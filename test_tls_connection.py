#!/usr/bin/env python3

"""
Simple test script to verify TLS connectivity in Flower
"""

import os
import sys
from pathlib import Path
import grpc
import flwr as fl
import threading
import time
import subprocess

# Base directory for certificates
BASE_DIR = Path(__file__).parent.absolute()
CERT_DIR = BASE_DIR / "certs"

def load_certificates():
    """Load SSL certificates."""
    try:
        # Verify that all required certificate files exist
        required_files = [
            CERT_DIR / "ca/ca.pem",
            CERT_DIR / "server/server.key",
            CERT_DIR / "server/server.pem",
            CERT_DIR / "client/client.pem",
            CERT_DIR / "client/client.key"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                print(f"❌ Missing certificate file: {file_path}")
                return None
        
        # Load certificates
        with open(CERT_DIR / "server/server.pem", 'rb') as f:
            server_cert = f.read()
        with open(CERT_DIR / "server/server.key", 'rb') as f:
            server_key = f.read()
        with open(CERT_DIR / "ca/ca.pem", 'rb') as f:
            ca_cert = f.read()
            
        print("✓ All certificates loaded successfully")
        return (server_cert, server_key, ca_cert)
    except Exception as e:
        print(f"❌ Error loading certificates: {e}")
        return None

def start_server(certificates, port=8443):
    """Start a simple Flower server with SSL."""
    print(f"Starting secure Flower server on port {port}...")
    try:
        # Configure and start server
        server = fl.server.start_server(
            server_address=f"[::]:{port}",
            config=fl.server.ServerConfig(num_rounds=1),
            certificates=certificates,
            strategy=fl.server.strategy.FedAvg(
                min_available_clients=1,
                min_fit_clients=1,
                min_evaluate_clients=1,
            ),
        )
        return server
    except Exception as e:
        print(f"❌ Server error: {e}")
        return None

def test_client_connection(ca_cert, port=8443):
    """Test if a client can connect securely."""
    print("Testing client connection to secure server...")
    try:
        # Create secure channel
        channel = grpc.secure_channel(
            f"localhost:{port}",
            grpc.ssl_channel_credentials(root_certificates=ca_cert)
        )
        
        # Try to connect with a short timeout
        try:
            grpc.channel_ready_future(channel).result(timeout=5)
            print("✅ Client successfully connected to secure server!")
            return True
        except grpc.FutureTimeoutError:
            print("❌ Client connection timed out")
            return False
    except Exception as e:
        print(f"❌ Client connection error: {e}")
        return False

if __name__ == "__main__":
    print("===== Testing TLS Connectivity =====")
    
    # Load certificates
    certificates = load_certificates()
    if certificates is None:
        print("❌ Certificate loading failed - exiting")
        sys.exit(1)
    
    # Extract CA certificate for client
    ca_cert = certificates[2]
    
    # Start server in a separate thread
    server_thread = threading.Thread(
        target=start_server,
        args=(certificates, 8443),
        daemon=True
    )
    
    try:
        server_thread.start()
        print("Server thread started - waiting for initialization")
        time.sleep(3)  # Give server time to start
        
        # Try client connection
        success = test_client_connection(ca_cert, 8443)
        
        if success:
            print("\n✅ TLS CONNECTION TEST SUCCESSFUL!")
            print("Your certificates and TLS configuration are working correctly.")
        else:
            print("\n❌ TLS CONNECTION TEST FAILED!")
            print("There appears to be an issue with your TLS configuration.")
            
        # Always exit after test
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        sys.exit(1)
