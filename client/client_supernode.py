#!/usr/bin/env python3
"""
Flower client strategy implementation.
This file contains the NumPyClient implementation to be used with flower-supernode CLI.

To use this with flower-supernode:
    flower-supernode --insecure --superlink='localhost:8443'

Or with secure connection:
    flower-supernode \
        --superlink='localhost:8443' \
        --root-certificates='certs/ca/ca.pem'
"""

import flwr as fl
from flwr.client import NumPyClient
import numpy as np

class DummyClient(NumPyClient):
    """Simple NumPyClient implementation for demonstration purposes."""
    
    def get_parameters(self, config):
        """Return empty parameters."""
        print("Dummy client: Returning parameters")
        return []
    
    def fit(self, parameters, config):
        """Return empty parameters, 0 samples, and empty metrics."""
        print("Dummy client: Performing fit")
        return [], 0, {}
    
    def evaluate(self, parameters, config):
        """Return 0.0 loss, 0 samples, and empty metrics."""
        print("Dummy client: Performing evaluation")
        return 0.0, 0, {}

# Create a client instance
client = DummyClient()

# Định nghĩa hàm client_fn để trả về một client mới cho mỗi request
def get_client_fn():
    """Tạo một client mới cho mỗi request.
    
    Hàm này được sử dụng bởi flower-supernode CLI để tạo client instance.
    """
    return client

# Sẵn sàng cho flower-supernode CLI sử dụng
# Lưu ý: Không cần đoạn mã khởi động client ở đây nữa, CLI sẽ làm điều đó

# Chỉ chạy khi script được chạy trực tiếp (cho việc testing)
if __name__ == "__main__":
    import os
    
    # Lấy đường dẫn tuyệt đối đến chứng chỉ CA
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ca_cert_path = os.path.join(base_dir, "certs/ca/ca.pem")
    
    # Đọc chứng chỉ CA
    with open(ca_cert_path, "rb") as f:
        ca_cert = f.read()
    
    print("Lưu ý: Khi sử dụng flower-supernode CLI, đoạn code này không được chạy.")
    print("Chạy script này trực tiếp chỉ dành cho việc testing.")
    
    # Khởi động client với chứng chỉ CA (chỉ cho testing)
    fl.client.start_client(
        server_address="localhost:8443",
        client=client,
        root_certificates=ca_cert
    )
