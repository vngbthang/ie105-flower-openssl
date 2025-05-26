#!/usr/bin/env python3
"""
Fix the chain file format for gRPC to work with SSL/TLS.
The gRPC expects a specific format for the chain file.
"""

import sys
from pathlib import Path
from cryptography import x509
from cryptography.hazmat.backends import default_backend

def extract_pem_certs(cert_file):
    """Extract individual PEM certificates from a file that may contain multiple certificates."""
    with open(cert_file, 'rb') as f:
        data = f.read()
    
    # Split by the PEM certificate markers
    pem_certs = []
    start_marker = b'-----BEGIN CERTIFICATE-----'
    end_marker = b'-----END CERTIFICATE-----'
    
    start_pos = 0
    while True:
        start_pos = data.find(start_marker, start_pos)
        if start_pos == -1:
            break
        
        end_pos = data.find(end_marker, start_pos) + len(end_marker)
        if end_pos == -1:
            break
        
        cert_data = data[start_pos:end_pos]
        pem_certs.append(cert_data)
        start_pos = end_pos
    
    return pem_certs

def fix_chain_file(chain_file, output_file=None):
    """Fix the chain file format to be compatible with gRPC."""
    if output_file is None:
        output_file = chain_file
    
    # Extract the certificates
    pem_certs = extract_pem_certs(chain_file)
    
    if len(pem_certs) < 2:
        print(f"Error: Less than 2 certificates found in {chain_file}")
        return False
    
    # Load certificates to verify and order them
    certificates = []
    for cert_data in pem_certs:
        try:
            cert = x509.load_pem_x509_certificate(cert_data, default_backend())
            certificates.append((cert, cert_data))
        except Exception as e:
            print(f"Error loading certificate: {e}")
            continue
    
    # Ensure proper order: server certificate first, then any intermediate certs, then root CA
    # For simplicity we'll just ensure the server cert is first (assuming it's not self-signed)
    server_cert = None
    ca_certs = []
    
    for cert, cert_data in certificates:
        # Check if it's a self-signed certificate (CA)
        if cert.issuer == cert.subject:
            ca_certs.append(cert_data)
        else:
            # Assume first non-self-signed is the server cert
            if server_cert is None:
                server_cert = cert_data
            else:
                ca_certs.append(cert_data)
    
    # If we couldn't identify a server cert, use the first one
    if server_cert is None and certificates:
        server_cert = certificates[0][1]
        ca_certs = [cert_data for _, cert_data in certificates[1:]]
    
    # Write the certificates in the correct order
    with open(output_file, 'wb') as f:
        if server_cert:
            f.write(server_cert)
            f.write(b'\n')
        
        # Add all CA certificates
        for cert_data in ca_certs:
            f.write(cert_data)
            f.write(b'\n')
    
    print(f"Chain file fixed and saved to {output_file}")
    return True

if __name__ == "__main__":
    cert_dir = Path(__file__).parent / "certs"
    chain_file = cert_dir / "server/chain.pem"
    fixed_chain_file = cert_dir / "server/fixed-chain.pem"
    
    print(f"Fixing chain file: {chain_file}")
    if fix_chain_file(chain_file, fixed_chain_file):
        # Backup original and replace
        backup_file = cert_dir / "server/chain.pem.bak"
        chain_file.rename(backup_file)
        fixed_chain_file.rename(chain_file)
        print(f"Original chain file backed up to {backup_file}")
        print(f"Fixed chain file saved as {chain_file}")
