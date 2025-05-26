#!/usr/bin/env python3
"""
Test script to check server binding capabilities with various configurations.
"""
import socket
import ssl
import argparse
from pathlib import Path

def test_socket_bind(host, port):
    """Test if we can bind to the specified host:port"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            s.listen(1)
            print(f"✓ Successfully bound to {host}:{port}")
            return True
    except Exception as e:
        print(f"✗ Failed to bind to {host}:{port}: {e}")
        return False

def test_ssl_configuration():
    """Test SSL configuration by creating an SSL context and loading certificates"""
    cert_dir = Path("./certs")
    server_cert = cert_dir / "server/server.pem"
    server_key = cert_dir / "server/server.key"
    chain_cert = cert_dir / "server/chain.pem"
    ca_cert = cert_dir / "ca/ca.pem"
    
    try:
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(certfile=str(server_cert), keyfile=str(server_key))
        print(f"✓ Successfully loaded server certificate and key")
    except Exception as e:
        print(f"✗ Failed to load server certificate and key: {e}")
        return False
        
    try:
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_cert_chain(certfile=str(chain_cert), keyfile=str(server_key))
        print(f"✓ Successfully loaded certificate chain and key")
    except Exception as e:
        print(f"✗ Failed to load certificate chain and key: {e}")
        
    try:
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_verify_locations(cafile=str(ca_cert))
        print(f"✓ Successfully loaded CA certificate")
        return True
    except Exception as e:
        print(f"✗ Failed to load CA certificate: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test server binding capabilities")
    parser.add_argument("--start-port", type=int, default=8443, help="Starting port number to test")
    parser.add_argument("--ports", type=int, default=5, help="Number of ports to test")
    args = parser.parse_args()
    
    print("=== Testing SSL Configuration ===")
    ssl_ok = test_ssl_configuration()
    
    print("\n=== Testing Socket Binding ===")
    hosts = ["localhost", "127.0.0.1", "0.0.0.0"] 
    
    for host in hosts:
        print(f"\nTesting binding to {host}:")
        for port in range(args.start_port, args.start_port + args.ports):
            test_socket_bind(host, port)
            
    print("\n=== Testing SSL Socket ===")
    if ssl_ok:
        host = "localhost"
        port = args.start_port
        try:
            # Create a socket and bind
            context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            context.load_cert_chain(
                certfile="./certs/server/chain.pem",
                keyfile="./certs/server/server.key"
            )
            
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
                s.listen(1)
                print(f"✓ Successfully bound SSL socket to {host}:{port}")
                
                # Wrap with SSL
                with context.wrap_socket(s, server_side=True) as ss:
                    print(f"✓ Successfully created SSL socket on {host}:{port}")
                    
                    # Don't actually accept connections, just test the setup
                    print("Press Ctrl+C to exit...")
                    try:
                        while True:
                            pass
                    except KeyboardInterrupt:
                        print("\nExiting...")
                        
        except Exception as e:
            print(f"✗ Failed to create SSL socket: {e}")
    else:
        print("Skipping SSL socket test due to SSL configuration issues.")

if __name__ == "__main__":
    main()
