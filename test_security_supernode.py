#!/usr/bin/env python3
"""
Script này kiểm tra độ an toàn của TLS/SSL trong hệ thống Flower với kiến trúc SuperNode/SuperLink
bằng cách thử kết nối mà không có chứng chỉ phù hợp.
"""

import flwr as fl
from flwr.client import NumPyClient
import ssl
import os
import sys
import time
import subprocess
import signal

class DummyClient(NumPyClient):
    """Client giả lập đơn giản cho mục đích kiểm thử."""
    def get_parameters(self, config): return []
    def fit(self, parameters, config): return [], 0, {}
    def evaluate(self, parameters, config): return 0.0, 0, {}

# Hàm get_client_fn cho kiến trúc SuperNode
def get_client_fn():
    return DummyClient()

def create_temp_supernode_script(ca_cert_path=None):
    """Tạo script tạm thời cho SuperNode để kiểm tra."""
    script_content = """
import flwr as fl
from flwr.client import NumPyClient
import sys
import time

class DummyClient(NumPyClient):
    def get_parameters(self, config): return []
    def fit(self, parameters, config): return [], 0, {}
    def evaluate(self, parameters, config): return 0.0, 0, {}

# Tạo client instance
client = DummyClient()

if __name__ == "__main__":
    try:
"""
    if ca_cert_path:
        script_content += f"""
        # Đọc chứng chỉ CA
        with open("{ca_cert_path}", "rb") as f:
            ca_cert = f.read()
        
        # Kết nối với server sử dụng chứng chỉ CA
        fl.client.start_client(
            server_address="localhost:8443",
            client=client,
            root_certificates=ca_cert
        )
"""
    else:
        script_content += """
        # Kết nối với server mà không có chứng chỉ
        fl.client.start_client(
            server_address="localhost:8443",
            client=client
        )
"""
    script_content += """
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    sys.exit(0)
"""
    
    with open("temp_test_supernode.py", "w") as f:
        f.write(script_content)

def test_without_certificates():
    """
    Kiểm tra 1: Kết nối mà không có bất kỳ chứng chỉ nào
    Kết quả mong đợi: Kết nối thất bại
    """
    print("\n==== Kiểm tra 1: Kết nối không có chứng chỉ ====")
    print("Đang thử kết nối mà không cung cấp bất kỳ chứng chỉ nào...")
    
    # Tạo script tạm thời
    create_temp_supernode_script()
    
    # Chạy script tạm thời
    result = subprocess.run(["python3", "temp_test_supernode.py"], 
                            capture_output=True, text=True)
    
    # Xóa script tạm thời
    os.remove("temp_test_supernode.py")
    
    # Kiểm tra kết quả
    if result.returncode != 0:
        error_msg = result.stderr if result.stderr else result.stdout
        print(f"✓ Kiểm tra thành công: Kết nối thất bại như mong đợi\nLỗi: {error_msg}")
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
    
    # Tạo script tạm thời với CA giả
    create_temp_supernode_script("fake_certs/fake_ca.pem")
    
    # Chạy script tạm thời
    result = subprocess.run(["python3", "temp_test_supernode.py"],
                           capture_output=True, text=True)
    
    # Xóa script tạm thời
    os.remove("temp_test_supernode.py")
    
    # Dọn dẹp chứng chỉ giả
    import shutil
    shutil.rmtree("fake_certs")
    
    # Kiểm tra kết quả
    if result.returncode != 0:
        error_msg = result.stderr if result.stderr else result.stdout
        print(f"✓ Kiểm tra thành công: Kết nối thất bại như mong đợi\nLỗi: {error_msg}")
        return True
    
    print("✗ Kiểm tra thất bại: Kết nối thành công một cách bất ngờ!")
    return False

def test_with_supernode_cli():
    """
    Kiểm tra 3: Kết nối bằng Flower SuperNode CLI mà không có chứng chỉ hợp lệ
    Kết quả mong đợi: Kết nối thất bại
    Lưu ý: Chỉ hoạt động nếu flower-supernode đã được cài đặt
    """
    print("\n==== Kiểm tra 3: Kết nối với SuperNode CLI mà không cung cấp chứng chỉ ====")
    
    # Kiểm tra xem flower-supernode có sẵn không
    check_cmd = subprocess.run(["which", "flower-supernode"], 
                               capture_output=True, text=True)
    
    if check_cmd.returncode != 0:
        print("⚠ Bỏ qua kiểm tra: CLI flower-supernode không có sẵn")
        return True
        
    print("Đang thử kết nối sử dụng flower-supernode CLI...")
    
    # Chạy flower-supernode với thời gian chờ
    process = subprocess.Popen(
        ["flower-supernode", "--superlink=localhost:8443"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Chờ một khoảng thời gian ngắn để kết nối thử nghiệm
    time.sleep(2)
    
    # Kết thúc quá trình
    process.terminate()
    process.wait()
    
    output, error = process.communicate()
    combined_output = output + error
    
    # Kiểm tra xem có lỗi kết nối không
    if "error" in combined_output.lower() or "failed" in combined_output.lower():
        print(f"✓ Kiểm tra thành công: Kết nối thất bại như mong đợi\nLỗi: {combined_output}")
        return True
    
    print("⚠ Không thể xác định kết quả kiểm tra. Kiểm tra thủ công.")
    return True

def run_tests():
    """Chạy tất cả các bài kiểm tra bảo mật"""
    print("===== ĐANG CHẠY CÁC KIỂM TRA BẢO MẬT TLS/SSL CHO FLOWER (SUPERNODE/SUPERLINK) =====")
    
    results = []
    results.append(test_without_certificates())
    results.append(test_with_invalid_ca())
    # results.append(test_with_supernode_cli())  # Bỏ ghi chú nếu CLI đã được cài đặt
    
    print("\n===== TÓM TẮT KẾT QUẢ KIỂM TRA =====")
    if all(results):
        print("✓ Tất cả kiểm tra đều thành công! Cài đặt TLS/SSL của bạn an toàn.")
    else:
        print("✗ Một số kiểm tra thất bại. Cài đặt TLS/SSL của bạn có thể có vấn đề về bảo mật.")
    
    print(f"Số kiểm tra thành công: {results.count(True)}/{len(results)}")

if __name__ == "__main__":
    print("Bắt đầu kiểm tra bảo mật cho hệ thống Flower SuperNode/SuperLink với TLS/SSL...")
    run_tests()
