#!/usr/bin/env python3

"""
Simple Flower server with TLS
"""

import sys
from pathlib import Path
import flwr as fl

# Base directory for certificates
BASE_DIR = Path(__file__).parent.absolute()
CERT_DIR = BASE_DIR / "certs"

def main():
    # Load certificates
    print("Loading certificates...")
    try:
        with open(CERT_DIR / "server/server.pem", 'rb') as f:
            server_cert = f.read()
        with open(CERT_DIR / "server/server.key", 'rb') as f:
            server_key = f.read()
        with open(CERT_DIR / "ca/ca.pem", 'rb') as f:
            ca_cert = f.read()
        
        print("Certificates loaded successfully")
    except Exception as e:
        print(f"Error loading certificates: {e}")
        return 1
    
    # Create certificate tuple
    certificates = (server_cert, server_key, ca_cert)
    
    # Start server
    print("Starting secure Flower server...")
    try:
        fl.server.start_server(
            server_address="[::]:8443",
            config=fl.server.ServerConfig(num_rounds=1),
            certificates=certificates,
            strategy=fl.server.strategy.FedAvg(
                min_available_clients=1,
                min_fit_clients=1,
                min_evaluate_clients=1,
            ),
        )
    except Exception as e:
        print(f"Server error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
