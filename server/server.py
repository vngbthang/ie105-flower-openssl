import flwr as fl
import ssl
import os

def main():
    # Tạo SSLContext cho server, bật mTLS
    # Method 1: Use correct format for certificates with start_server()
    # Use absolute paths to ensure the certificate files are found correctly
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server_cert_file = os.path.join(base_dir, "certs/server/server.pem")
    server_key_file = os.path.join(base_dir, "certs/server/server.key")
    ca_cert_file = os.path.join(base_dir, "certs/ca/ca.pem")
    
    # Read the certificate files as bytes
    with open(server_cert_file, 'rb') as f:
        server_cert = f.read()
    with open(server_key_file, 'rb') as f:
        server_key = f.read()
    with open(ca_cert_file, 'rb') as f:
        ca_cert = f.read()
    
    # Pass the certificates as a tuple of bytes
    certificates = (server_cert, server_key, ca_cert)
    
    fl.server.start_server(
        server_address="[::]:8443",
        config=fl.server.ServerConfig(num_rounds=1),
        certificates=certificates,
    )

if __name__ == "__main__":
    main()
