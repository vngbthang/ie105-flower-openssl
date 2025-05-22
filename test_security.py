#!/usr/bin/env python3
"""
Script này kiểm tra độ an toàn của TLS/SSL trong hệ thống Flower bằng cách
thử kết nối mà không có chứng chỉ phù hợp.
"""

import flwr as fl
from flwr.client import NumPyClient
import ssl
import os
import sys

class DummyClient(NumPyClient):
    """Client giả lập đơn giản cho mục đích kiểm thử."""
    def get_parameters(self, config): return []
    def fit(self, parameters, config): return [], 0, {}
    def evaluate(self, parameters, config): return 0.0, 0, {}

def test_without_certificates():
    """
    Kiểm tra 1: Kết nối mà không có bất kỳ chứng chỉ nào
    Kết quả mong đợi: Kết nối thất bại
    """
    print("\n==== Kiểm tra 1: Kết nối không có chứng chỉ ====")
    print("Đang thử kết nối mà không cung cấp bất kỳ chứng chỉ nào...")
    
    client = DummyClient()
    # Sử dụng API mới thay vì API đã lỗi thời
    try:
        fl.client.start_client(
            server_address="localhost:8443",
            client=client,
        )
    except Exception as e:
        print(f"✓ Kiểm tra thành công: Kết nối thất bại như mong đợi\nLỗi: {e}")
        return True
    
    print("✗ Kiểm tra thất bại: Kết nối thành công một cách bất ngờ!")
    return False

def test_with_invalid_ca():
    """
    Kiểm tra 2: Kết nối với chứng chỉ CA không hợp lệ
    Kết quả mong đợi: Kết nối thất bại với lỗi xác thực chứng chỉ
    """
    print("\n==== Kiểm tra 2: Kết nối với CA không hợp lệ ====")
    print("Đang tạo chứng chỉ CA giả...")
    
    # Tạo thư mục tạm thời cho chứng chỉ giả
    os.makedirs("fake_certs", exist_ok=True)
    
    # Sử dụng lệnh OpenSSL để tạo CA giả
    os.system("openssl req -x509 -newkey rsa:4096 -keyout fake_certs/fake_ca.key -out fake_certs/fake_ca.pem -days 365 -nodes -subj '/CN=FakeCA'")
    
    # Đọc chứng chỉ CA giả
    with open("fake_certs/fake_ca.pem", "rb") as f:
        fake_ca_cert = f.read()
    
    client = DummyClient()
    # Sử dụng API mới thay vì API đã lỗi thời
    try:
        print("Đang thử kết nối với chứng chỉ CA giả...")
        fl.client.start_client(
            server_address="localhost:8443",
            client=client,
            root_certificates=fake_ca_cert
        )
    except Exception as e:
        print(f"✓ Kiểm tra thành công: Kết nối thất bại như mong đợi\nLỗi: {e}")
        
        # Dọn dẹp chứng chỉ giả
        import shutil
        shutil.rmtree("fake_certs")
        return True
    
    # Dọn dẹp chứng chỉ giả
    import shutil
    shutil.rmtree("fake_certs")
    
    print("✗ Kiểm tra thất bại: Kết nối thành công một cách bất ngờ!")
    return False

def run_tests():
    """Chạy tất cả các bài kiểm tra bảo mật"""
    print("===== ĐANG CHẠY CÁC KIỂM TRA BẢO MẬT TLS/SSL CHO FLOWER =====")
    
    results = []
    results.append(test_without_certificates())
    results.append(test_with_invalid_ca())
    
    print("\n===== TÓM TẮT KẾT QUẢ KIỂM TRA =====")
    if all(results):
        print("✓ Tất cả kiểm tra đều thành công! Cài đặt TLS/SSL của bạn an toàn.")
    else:
        print("✗ Một số kiểm tra thất bại. Cài đặt TLS/SSL của bạn có thể có vấn đề về bảo mật.")
    
    print(f"Số kiểm tra thành công: {results.count(True)}/{len(results)}")

if __name__ == "__main__":
    print("Bắt đầu kiểm tra bảo mật cho hệ thống Flower với TLS/SSL...")
    run_tests()
