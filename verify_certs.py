#!/usr/bin/env python3

"""
Script to verify that the SSL/TLS certificates are properly configured
"""

import os
import sys
from pathlib import Path
import ssl

# Base directory for certificates
BASE_DIR = Path(__file__).parent.absolute()
CERT_DIR = BASE_DIR / "certs"

def verify_cert_files():
    """Verify all certificate files exist and have proper permissions."""
    print("Checking certificate files...")
    
    # Define required files
    required_files = [
        CERT_DIR / "ca/ca.pem",
        CERT_DIR / "ca/ca.key",
        CERT_DIR / "server/server.pem",
        CERT_DIR / "server/server.key",
        CERT_DIR / "client/client.pem",
        CERT_DIR / "client/client.key"
    ]
    
    # Check existence and permissions
    all_good = True
    for file_path in required_files:
        if not file_path.exists():
            print(f"❌ Missing: {file_path}")
            all_good = False
        else:
            print(f"✓ Found: {file_path}")
            # Check permissions
            if not os.access(file_path, os.R_OK):
                print(f"  ⚠️ Warning: {file_path} is not readable")
                all_good = False
    
    return all_good

def verify_cert_content():
    """Verify certificate content and validity."""
    print("\nVerifying certificate content...")
    
    try:
        # Load CA certificate
        ca_cert_path = CERT_DIR / "ca/ca.pem"
        context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        context.load_verify_locations(cafile=str(ca_cert_path))
        print(f"✓ Successfully loaded CA certificate")
        
        # Verify server certificate against CA
        server_cert_path = CERT_DIR / "server/server.pem"
        try:
            with open(server_cert_path, 'rb') as f:
                server_cert_data = f.read()
            server_cert = ssl.PEM_cert_to_DER_cert(server_cert_data.decode())
            context.verify_certificate(server_cert)
            print(f"✓ Server certificate is valid against CA")
        except Exception as e:
            print(f"❌ Server certificate validation failed: {e}")
            return False
            
        # Verify client certificate against CA
        client_cert_path = CERT_DIR / "client/client.pem"
        try:
            with open(client_cert_path, 'rb') as f:
                client_cert_data = f.read()
            client_cert = ssl.PEM_cert_to_DER_cert(client_cert_data.decode())
            context.verify_certificate(client_cert)
            print(f"✓ Client certificate is valid against CA")
        except Exception as e:
            print(f"❌ Client certificate validation failed: {e}")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Certificate verification error: {e}")
        return False

def main():
    """Main verification function."""
    print("===== SSL/TLS Certificate Verification =====")
    
    # Verify files
    files_ok = verify_cert_files()
    if not files_ok:
        print("\n❌ Certificate files check failed")
        print("Running ./generate_certs.sh to fix...")
        os.system("bash ./generate_certs.sh")
        
        # Verify again
        files_ok = verify_cert_files()
        if not files_ok:
            print("\n❌ Certificate files still missing after regeneration")
            return False
    
    # Verify content
    content_ok = verify_cert_content()
    
    if files_ok and content_ok:
        print("\n✅ All certificates are valid and properly configured")
        return True
    else:
        print("\n❌ Certificate verification failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
