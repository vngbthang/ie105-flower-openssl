#!/usr/bin/env python3

import flwr as fl
from flwr.client import NumPyClient
import ssl
import os

class DummyClient(NumPyClient):
    def get_parameters(self, config): return []
    def fit(self, parameters, config): return [], 0, {}
    def evaluate(self, parameters, config): return 0.0, 0, {}

def main():
    # Use absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create SSL context
    context = ssl.create_default_context(
        ssl.Purpose.SERVER_AUTH,
        cafile=os.path.join(base_dir, "certs/ca/ca.pem")
    )
    context.load_cert_chain(
        certfile=os.path.join(base_dir, "certs/client/client.pem"),
        keyfile=os.path.join(base_dir, "certs/client/client.key")
    )
    
    # Read the certificate files as bytes for the newer interface
    with open(os.path.join(base_dir, "certs/ca/ca.pem"), 'rb') as f:
        ca_cert = f.read()
    with open(os.path.join(base_dir, "certs/client/client.pem"), 'rb') as f:
        client_cert = f.read()
    with open(os.path.join(base_dir, "certs/client/client.key"), 'rb') as f:
        client_key = f.read()
    
    # For mTLS: certificates = (client_cert, client_key, ca_cert)
    # But for client, we only need to provide ca_cert as root_certificates
    fl.client.start_numpy_client(
        server_address="localhost:8443",
        client=DummyClient(),
        root_certificates=ca_cert
    )

if __name__ == "__main__":
    main()
