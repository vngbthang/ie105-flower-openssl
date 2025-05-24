#!/usr/bin/env python3

"""
Simple Flower client with TLS
"""

import sys
from pathlib import Path
import flwr as fl
import numpy as np

# Base directory for certificates
BASE_DIR = Path(__file__).parent.absolute()
CERT_DIR = BASE_DIR / "certs"

class SimpleClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        """Return random parameters to simulate a model."""
        return [np.random.rand(10, 10).astype(np.float32) for _ in range(2)]
    
    def fit(self, parameters, config):
        """Simulate model training."""
        print("Client: Performing fit()")
        return self.get_parameters(config), 10, {}
    
    def evaluate(self, parameters, config):
        """Simulate model evaluation."""
        print("Client: Performing evaluate()")
        return 0.5, 10, {"accuracy": 0.95}

def main():
    # Load CA certificate
    print("Loading CA certificate...")
    try:
        with open(CERT_DIR / "ca/ca.pem", 'rb') as f:
            ca_cert = f.read()
        
        print("CA certificate loaded successfully")
    except Exception as e:
        print(f"Error loading CA certificate: {e}")
        return 1
    
    # Create client and connect to server
    print("Starting secure Flower client...")
    try:
        fl.client.start_client(
            server_address="localhost:8443",
            client=SimpleClient(),
            root_certificates=ca_cert
        )
    except Exception as e:
        print(f"Client error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
