#!/usr/bin/env python3
"""
Script này xác minh rằng dữ liệu được truyền giữa client và server trong hệ thống Học Liên Hợp
sử dụng Flower với TLS/SSL thực sự được mã hóa.

Bài kiểm tra:
1. Ghi lại lưu lượng mạng trong phiên huấn luyện FL bảo mật
2. Huấn luyện một mô hình với tập dữ liệu đơn giản
3. Phân tích các gói tin đã ghi để xác minh mã hóa
4. So sánh với kết nối không bảo mật (tùy chọn)
"""

import os
import sys
import time
import subprocess
import tempfile
import threading
import signal
import numpy as np
from pathlib import Path

# Flower imports
import flwr as fl
from flwr.client import NumPyClient, start_client
from flwr.common import NDArrays, Scalar, FitRes
import ssl

# ML libraries for our test dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# For packet capture
import shlex
import socket

# Base directory for certificates
BASE_DIR = Path(__file__).parent.absolute()
CERT_DIR = BASE_DIR / "certs"
SERVER_PORT = 8443

def regenerate_certificates():
    """Tạo lại các chứng chỉ SSL/TLS."""
    print("Đang tạo lại các chứng chỉ SSL/TLS...")
    cert_script = BASE_DIR / "generate_certs.sh"
    
    if not cert_script.exists():
        print(f"Lỗi: Không tìm thấy script tạo chứng chỉ: {cert_script}")
        return False
    
    try:
        # Make sure script is executable
        os.chmod(cert_script, os.stat(cert_script).st_mode | 0o111)
        
        # Run the certificate generation script
        result = subprocess.run(
            ["bash", str(cert_script)], 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        print("Đã tạo thành công các chứng chỉ mới:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi tạo lại chứng chỉ: {e}")
        if e.stdout:
            print(f"Stdout: {e.stdout}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False


# Global flag to control the server thread
server_running = False

class MnistNet(nn.Module):
    """Mô hình đơn giản cho phân loại MNIST"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class MnistClient(NumPyClient):
    """Client cho học liên hợp MNIST"""
    def __init__(self, model, trainloader, testloader):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def get_parameters(self, config):
        """Lấy tham số mô hình dưới dạng danh sách các mảng NumPy."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
        """Huấn luyện mô hình trên tập dữ liệu cục bộ."""
        set_parameters(self.model, parameters)
        train(self.model, self.trainloader, epochs=1, device=self.device)
        
        # Thêm một số thông báo để theo dõi trong quá trình ghi lưu lượng
        print("CLIENT: Đang gửi tham số mô hình đến server...")
        
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        """Đánh giá mô hình trên tập dữ liệu cục bộ."""
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.testloader, device=self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


def set_parameters(model, parameters):
    """Đặt tham số mô hình từ danh sách các mảng NumPy."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.Tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def train(model, trainloader, epochs, device):
    """Huấn luyện mô hình trong một số epoch."""
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def test(model, testloader, device):
    """Đánh giá mô hình trên tập dữ liệu kiểm tra."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return loss / len(testloader), accuracy


def load_data():
    """Tải tập dữ liệu MNIST."""
    transform = ToTensor()
    trainset = MNIST(
        root="./data", 
        train=True, 
        download=True, 
        transform=transform
    )
    testset = MNIST(
        root="./data", 
        train=False, 
        download=True, 
        transform=transform
    )
    
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64)
    
    return trainloader, testloader


def run_server(secure=True):
    """Chạy Flower server."""
    global server_running
    server_running = True
    
    if secure:
        print("Đang khởi động Flower server bảo mật với TLS/SSL...")
        try:
            # Kiểm tra tồn tại chứng chỉ
            if not (CERT_DIR / "server/server.pem").exists():
                raise FileNotFoundError(f"Không tìm thấy chứng chỉ server: {CERT_DIR / 'server/server.pem'}")
            if not (CERT_DIR / "server/server.key").exists():
                raise FileNotFoundError(f"Không tìm thấy khóa server: {CERT_DIR / 'server/server.key'}")
            if not (CERT_DIR / "ca/ca.pem").exists():
                raise FileNotFoundError(f"Không tìm thấy chứng chỉ CA: {CERT_DIR / 'ca/ca.pem'}")
                
            # Tải các chứng chỉ
            with open(CERT_DIR / "server/server.pem", 'rb') as f:
                server_cert = f.read()
            with open(CERT_DIR / "server/server.key", 'rb') as f:
                server_key = f.read()
            with open(CERT_DIR / "ca/ca.pem", 'rb') as f:
                ca_cert = f.read()
            
            # Truyền chứng chỉ đến Flower server
            certificates = (server_cert, server_key, ca_cert)
            
            print(f"Đã tải thành công các chứng chỉ từ {CERT_DIR}")
        except Exception as e:
            print(f"Lỗi khi tải chứng chỉ: {e}")
            return
        
        fl.server.start_server(
            server_address=f"[::]:8443",
            config=fl.server.ServerConfig(num_rounds=1),
            certificates=certificates,
            strategy=fl.server.strategy.FedAvg(
                min_available_clients=1,
                min_fit_clients=1,
                min_evaluate_clients=1,
            ),
        )
    else:
        print("Đang khởi động Flower server không bảo mật (không có TLS/SSL)...")
        fl.server.start_server(
            server_address=f"[::]:8080",
            config=fl.server.ServerConfig(num_rounds=1),
            strategy=fl.server.strategy.FedAvg(
                min_available_clients=1,
                min_fit_clients=1,
                min_evaluate_clients=1,
            ),
        )
    
    server_running = False


def run_client(secure=True):
    """Chạy Flower client với một mô hình MNIST đơn giản."""
    # Tải dữ liệu
    trainloader, testloader = load_data()
    
    # Khởi tạo mô hình
    model = MnistNet()
    
    # Tạo client
    client = MnistClient(model, trainloader, testloader)
    
    if secure:
        print("Đang khởi động Flower client bảo mật với TLS/SSL...")
        try:
            # Kiểm tra tồn tại chứng chỉ CA
            if not (CERT_DIR / "ca/ca.pem").exists():
                raise FileNotFoundError(f"Không tìm thấy chứng chỉ CA: {CERT_DIR / 'ca/ca.pem'}")
            
            # Tải chứng chỉ CA
            with open(CERT_DIR / "ca/ca.pem", 'rb') as f:
                ca_cert = f.read()
            
            # Khởi động client với TLS/SSL
            start_client(
                server_address="localhost:8443",
                client=client,
                root_certificates=ca_cert
            )
        except Exception as e:
            print(f"Lỗi khi tải chứng chỉ client: {e}")
    else:
        print("Đang khởi động Flower client không bảo mật (không có TLS/SSL)...")
        start_client(
            server_address="localhost:8080",
            client=client
        )


def capture_packets(output_file, interface="lo", port=SERVER_PORT, duration=30):
    """Bắt gói tin mạng sử dụng tcpdump."""
    print(f"Đang bắt đầu bắt gói tin trên giao diện {interface}, cổng {port}...")
    
    # Get the current user ID
    current_uid = os.geteuid()
    
    # Create an empty capture file first with proper permissions
    with open(output_file, 'wb') as f:
        pass
    os.chmod(output_file, 0o666)  # Make it readable by all
    
    # Xây dựng lệnh tcpdump với sudo nếu cần thiết
    if current_uid != 0:  # Not running as root
        if os.system("which sudo >/dev/null 2>&1") == 0:
            cmd = f"sudo tcpdump -i {interface} -w {output_file} port {port} -v"
        else:
            print("⚠️ Cảnh báo: Không tìm thấy sudo và bạn không phải là root. Bắt gói tin có thể thất bại.")
            cmd = f"tcpdump -i {interface} -w {output_file} port {port} -v"
    else:
        cmd = f"tcpdump -i {interface} -w {output_file} port {port} -v"
    
    try:
        # Start tcpdump
        process = subprocess.Popen(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Chờ khoảng thời gian đã chỉ định
        time.sleep(duration)
        
        # Stop tcpdump
        process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate()
        print(f"Bắt gói tin hoàn tất. Đầu ra được lưu vào {output_file}")
        if stderr:
            print(f"Đầu ra tcpdump: {stderr.decode()}")
            
        return True
    except Exception as e:
        print(f"Lỗi trong quá trình bắt gói tin: {e}")
        return False


def analyze_capture(capture_file):
    """Phân tích các gói tin được bắt để kiểm tra mã hóa."""
    print(f"Đang phân tích các gói tin được bắt từ {capture_file}...")
    
    encryption_evidence = {
        'ssl_traffic': False,
        'tls_versions': set(),
        'handshake_count': 0,
        'app_data_count': 0,
        'cipher_suites': set(),
        'total_packets': 0
    }
    
    # Đếm tổng số gói tin
    try:
        cmd = f"tshark -r {capture_file} | wc -l"
        result = subprocess.run(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        encryption_evidence['total_packets'] = int(result.stdout.strip())
        print(f"Tổng số gói tin được bắt: {encryption_evidence['total_packets']}")
    except Exception:
        print("Không thể đếm tổng số gói tin")
    
    # Kiểm tra lưu lượng SSL/TLS
    try:
        cmd = f"tshark -r {capture_file} -Y 'ssl' -T fields -e ssl.record.content_type"
        result = subprocess.run(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.stdout.strip():
            encryption_evidence['ssl_traffic'] = True
            content_types = result.stdout.strip().split('\n')
            encryption_evidence['handshake_count'] = content_types.count('22')  # Handshake
            encryption_evidence['app_data_count'] = content_types.count('23')   # Application Data
            
            print(f"Thông điệp bắt tay SSL/TLS: {encryption_evidence['handshake_count']}")
            print(f"Thông điệp dữ liệu ứng dụng SSL/TLS: {encryption_evidence['app_data_count']}")
    except Exception as e:
        print(f"Lỗi khi phân tích lưu lượng SSL/TLS: {e}")
    
    # Kiểm tra phiên bản TLS được sử dụng
    try:
        cmd = f"tshark -r {capture_file} -Y 'ssl.handshake.type == 1' -T fields -e ssl.handshake.version"
        result = subprocess.run(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.stdout.strip():
            versions = result.stdout.strip().split('\n')
            version_map = {
                '0x0301': 'TLS 1.0',
                '0x0302': 'TLS 1.1',
                '0x0303': 'TLS 1.2',
                '0x0304': 'TLS 1.3'
            }
            
            for version in versions:
                if version in version_map:
                    encryption_evidence['tls_versions'].add(version_map[version])
                else:
                    encryption_evidence['tls_versions'].add(f"Không xác định ({version})")
            
            print(f"Phiên bản TLS được phát hiện: {', '.join(encryption_evidence['tls_versions'])}")
    except Exception as e:
        print(f"Lỗi khi phân tích phiên bản TLS: {e}")
    
    # Kiểm tra các bộ mã hóa
    try:
        cmd = f"tshark -r {capture_file} -Y 'ssl.handshake.type == 1' -T fields -e ssl.handshake.ciphersuite"
        result = subprocess.run(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.stdout.strip():
            ciphers = set()
            for line in result.stdout.strip().split('\n'):
                if line and ',' in line:  # Nhiều bộ mã hóa trong client hello
                    for cipher in line.split(','):
                        ciphers.add(cipher.strip())
                elif line:
                    ciphers.add(line.strip())
            
            if len(ciphers) > 5:  # Nếu quá nhiều để hiển thị
                print(f"Bộ mã hóa được cung cấp: {len(ciphers)} bộ mã khác nhau")
            else:
                print(f"Bộ mã hóa được cung cấp: {', '.join(ciphers)}")
            
            encryption_evidence['cipher_suites'] = ciphers
    except Exception as e:
        print(f"Lỗi khi phân tích bộ mã hóa: {e}")
    
    # Đánh giá cuối cùng
    if encryption_evidence['ssl_traffic'] and encryption_evidence['app_data_count'] > 0:
        print("\nĐÁNH GIÁ MÃ HÓA: Lưu lượng cho thấy bằng chứng mạnh mẽ về mã hóa TLS/SSL")
        return True
    elif encryption_evidence['ssl_traffic']:
        print("\nĐÁNH GIÁ MÃ HÓA: Lưu lượng cho thấy một số bằng chứng về mã hóa TLS/SSL")
        return True
    else:
        print("\nĐÁNH GIÁ MÃ HÓA: Không tìm thấy bằng chứng về mã hóa TLS/SSL")
        return False


def extract_plaintext(capture_file):
    """Cố gắng trích xuất văn bản thuần túy từ các gói tin được bắt."""
    print(f"Đang cố gắng trích xuất văn bản thuần túy từ {capture_file}...")
    
    # Sử dụng lệnh strings để trích xuất chuỗi có thể đọc được từ dữ liệu nhị phân
    # Tìm kiếm các thuật ngữ phổ biến có thể xuất hiện trong dữ liệu hoặc tham số mô hình
    keywords = ["model", "tensor", "param", "weight", "gradient", "numpy", "array", "fit", "train"]
    found_plaintext = []
    
    for keyword in keywords:
        cmd = f"strings {capture_file} | grep -i -A 3 -B 3 '{keyword}'"
        try:
            result = subprocess.run(
                shlex.split(cmd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.stdout.strip():
                found_plaintext.append(f"Đã tìm thấy tham chiếu '{keyword}':")
                found_plaintext.append(result.stdout.strip())
        except Exception as e:
            print(f"Lỗi khi tìm kiếm '{keyword}': {e}")
    
    # Tìm kiếm cụ thể các thông điệp giao thức Flower
    cmd = f"strings {capture_file} | grep -i -A 3 -B 3 'flwr'"
    try:
        result = subprocess.run(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.stdout.strip():
            found_plaintext.append("Đã tìm thấy tham chiếu đến framework Flower:")
            found_plaintext.append(result.stdout.strip())
    except Exception:
        pass
    
    if found_plaintext:
        print("Đã tìm thấy văn bản thuần túy tiềm năng trong lưu lượng:")
        for line in found_plaintext:
            print(line)
        return "\n".join(found_plaintext)
    else:
        print("Không tìm thấy văn bản thuần túy liên quan đến mô hình trong lưu lượng.")
        return ""


def cleanup(*args):
    """Hàm dọn dẹp cho trình xử lý tín hiệu."""
    print("Đang dọn dẹp...")
    # Kết thúc tất cả các tiến trình Python
    os.system("pkill -f 'test_encryption.py'")
    os.system("pkill -f 'tcpdump'")
    sys.exit(0)


def test_secure_communication():
    """Kiểm tra kết nối bảo mật giữa server và client với việc bắt gói tin."""
    print("\n===== ĐANG KIỂM TRA KẾT NỐI BẢO MẬT (VỚI TLS/SSL) =====")
    
    # Thiết lập các bộ xử lý tín hiệu để dọn dẹp
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    # Sử dụng tệp bắt gói tin từ tcpdump chạy riêng biệt
    secure_capture_file = os.path.join(tempfile.gettempdir(), "flower_secure_traffic.pcap")
    
    # Check if packet capture file exists and is being written to
    if not os.path.exists(secure_capture_file):
        print(f"⚠️ Cảnh báo: Tệp bắt gói tin {secure_capture_file} không tồn tại.")
        print("Hãy chắc chắn bạn đã chạy script start_packet_capture.sh trước.")
    else:
        print(f"✓ Sử dụng tệp bắt gói tin: {secure_capture_file}")
    
    # Khởi động server trong một luồng riêng
    server_thread = threading.Thread(target=run_server, args=(True,))
    server_thread.daemon = True
    server_thread.start()
    time.sleep(2)  # Cho server thời gian để khởi động
    
    # Chạy client
    try:
        run_client(secure=True)
    except Exception as e:
        print(f"Lỗi client: {e}")
    
    # Đợi một chút để đảm bảo gói tin được bắt
    print("Đợi 3 giây cho gói tin được bắt...")
    time.sleep(3)
    
    # Phân tích bản ghi
    print("\n===== PHÂN TÍCH KẾT NỐI BẢO MẬT =====")
    is_encrypted = False
    if os.path.exists(secure_capture_file) and os.path.getsize(secure_capture_file) > 0:
        is_encrypted = analyze_capture(secure_capture_file)
    else:
        print(f"❌ Không thể phân tích tệp bắt gói tin: {secure_capture_file} không tồn tại hoặc rỗng")
    
    # Cố gắng trích xuất văn bản thuần túy
    plaintext = extract_plaintext(secure_capture_file)
    
    # Chờ server hoàn thành
    while server_running:
        time.sleep(1)
    
    return is_encrypted, secure_capture_file, plaintext


def test_insecure_communication():
    """Kiểm tra kết nối không bảo mật giữa server và client với việc bắt gói tin."""
    print("\n===== ĐANG KIỂM TRA KẾT NỐI KHÔNG BẢO MẬT (KHÔNG CÓ TLS/SSL) =====")
    
    # Tạo tệp tạm thời để bắt gói tin
    insecure_capture_file = os.path.join(tempfile.gettempdir(), "flower_insecure_traffic.pcap")
    
    # Bắt đầu bắt gói tin trong một luồng riêng
    capture_thread = threading.Thread(
        target=capture_packets, 
        args=(insecure_capture_file, "lo", 8080, 30)
    )
    capture_thread.daemon = True
    capture_thread.start()
    time.sleep(1)  # Cho tcpdump thời gian để khởi động
    
    # Khởi động server trong một luồng riêng
    server_thread = threading.Thread(target=run_server, args=(False,))
    server_thread.daemon = True
    server_thread.start()
    time.sleep(2)  # Cho server thời gian để khởi động
    
    # Chạy client
    try:
        run_client(secure=False)
    except Exception as e:
        print(f"Lỗi client: {e}")
    
    # Đảm bảo quá trình bắt gói tin hoàn thành
    capture_thread.join()
    
    # Phân tích bản ghi để kiểm tra văn bản thuần túy
    print("\n===== PHÂN TÍCH KẾT NỐI KHÔNG BẢO MẬT =====")
    is_encrypted = analyze_capture(insecure_capture_file)
    
    # Cố gắng trích xuất văn bản thuần túy
    plaintext = extract_plaintext(insecure_capture_file)
    
    # Chờ server hoàn thành
    while server_running:
        time.sleep(1)
    
    return is_encrypted, insecure_capture_file, plaintext


def compare_captures(secure_file, insecure_file):
    """So sánh các bắt gói tin bảo mật và không bảo mật."""
    print("\n===== SO SÁNH KẾT NỐI BẢO MẬT VÀ KHÔNG BẢO MẬT =====")
    
    # Compare file sizes first (encrypted files tend to be larger due to overhead)
    secure_size = os.path.getsize(secure_file)
    insecure_size = os.path.getsize(insecure_file)
    
    print(f"Kích thước tệp bắt bảo mật: {secure_size / 1024:.2f} KB")
    print(f"Kích thước tệp bắt không bảo mật: {insecure_size / 1024:.2f} KB")
    print(f"Chênh lệch kích thước: {abs(secure_size - insecure_size) / 1024:.2f} KB")
    
    # Kiểm tra tiêu đề gRPC trong cả hai bản ghi
    secure_grpc = check_for_grpc(secure_file)
    insecure_grpc = check_for_grpc(insecure_file)
    
    print(f"Tiêu đề gRPC hiển thị trong bắt bảo mật: {'Không' if secure_grpc else 'Có'}")
    print(f"Tiêu đề gRPC hiển thị trong bắt không bảo mật: {'Không' if not insecure_grpc else 'Có'}")
    
    return {
        'secure_size': secure_size,
        'insecure_size': insecure_size,
        'secure_grpc_hidden': not secure_grpc,
        'insecure_grpc_visible': insecure_grpc
    }


def check_for_grpc(capture_file):
    """Kiểm tra tiêu đề gRPC có thể nhìn thấy trong bản ghi gói tin."""
    cmd = f"tshark -r {capture_file} -Y 'http2' -T fields -e http2.header.value"
    
    try:
        result = subprocess.run(
            shlex.split(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def main():
    """Hàm chính để kiểm tra mã hóa trong kết nối Flower."""
    print("===== KIỂM TRA MÃ HÓA TRONG KẾT NỐI FLOWER =====")
    
    # Kiểm tra xem các chứng chỉ đã tồn tại chưa
    cert_files = [
        CERT_DIR / "server/server.pem",
        CERT_DIR / "server/server.key",
        CERT_DIR / "ca/ca.pem",
        CERT_DIR / "client/client.pem",
        CERT_DIR / "client/client.key"
    ]
    
    missing_certs = [str(cert) for cert in cert_files if not cert.exists()]
    
    if missing_certs:
        print(f"Lỗi: Không tìm thấy các chứng chỉ TLS/SSL cần thiết: {', '.join(missing_certs)}")
        print("Đang tạo lại các chứng chỉ...")
        
        if not regenerate_certificates():
            print("❌ Không thể tạo lại chứng chỉ. Kiểm tra lỗi ở trên.")
            return
        
        # Verify certificates were created
        still_missing = [str(cert) for cert in cert_files if not cert.exists()]
        if still_missing:
            print(f"❌ Vẫn còn thiếu các chứng chỉ sau khi tạo lại: {', '.join(still_missing)}")
            return
        
        print("✅ Đã tạo lại thành công tất cả các chứng chỉ cần thiết.")
    
    # Tạo tệp ghi kết quả
    log_file = os.path.join(BASE_DIR, "encryption_test_results.log")
    with open(log_file, "w") as f:
        f.write("KẾT QUẢ KIỂM TRA MÃ HÓA TLS/SSL FLOWER\n")
        f.write(f"Ngày: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
    
    # Check for tcpdump
    try:
        subprocess.run(["which", "tcpdump"], check=True, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("Lỗi: tcpdump chưa được cài đặt. Vui lòng cài đặt bằng lệnh 'sudo apt install tcpdump'.")
        return
    
    # Check for tshark
    has_tshark = True
    try:
        subprocess.run(["which", "tshark"], check=True, stdout=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("Cảnh báo: tshark (Wireshark CLI) chưa được cài đặt. Một số phân tích sẽ bị giới hạn.")
        # Try to install tshark if we have sudo access
        if os.geteuid() == 0 or os.system("sudo -n true 2>/dev/null") == 0:
            print("Đang cố gắng cài đặt tshark tự động...")
            
            # Check which package manager is available
            if os.system("which dnf >/dev/null 2>&1") == 0:
                os.system("sudo dnf install -y wireshark")
            elif os.system("which apt-get >/dev/null 2>&1") == 0:
                os.system("sudo apt-get update && sudo apt-get install -y tshark")
            elif os.system("which yum >/dev/null 2>&1") == 0:
                os.system("sudo yum install -y wireshark")
                
            # Check if installation was successful
            if os.system("which tshark >/dev/null 2>&1") == 0:
                print("✅ tshark đã được cài đặt thành công!")
                has_tshark = True
            else:
                print("❌ Không thể cài đặt tshark tự động.")
                print("Hãy xem xét cài đặt bằng lệnh 'sudo apt install tshark' hoặc 'sudo dnf install wireshark'.")
                has_tshark = False
        else:
            print("Hãy xem xét cài đặt bằng lệnh 'sudo apt install tshark' hoặc 'sudo dnf install wireshark'.")
            has_tshark = False
    
    # Test secure communication
    print("\n" + "=" * 70)
    print("GIAI ĐOẠN 1: KIỂM TRA KẾT NỐI BẢO MẬT VỚI TLS/SSL")
    print("=" * 70)
    secure_is_encrypted, secure_file, secure_plaintext = test_secure_communication()
    
    # Test insecure communication (optional)
    test_insecure = False
    if has_tshark:  # Only run comparative test if tshark is available
        response = input("\nBạn có muốn chạy kiểm tra kết nối không bảo mật để so sánh không? (y/n): ")
        test_insecure = response.lower() == 'y'
    
    if test_insecure:
        print("\n" + "=" * 70)
        print("GIAI ĐOẠN 2: KIỂM TRA KẾT NỐI KHÔNG BẢO MẬT ĐỂ SO SÁNH")
        print("=" * 70)
        try:
            insecure_is_encrypted, insecure_file, insecure_plaintext = test_insecure_communication()
            
            # Compare the two captures
            print("\n" + "=" * 70)
            print("GIAI ĐOẠN 3: SO SÁNH KẾT NỐI BẢO MẬT VÀ KHÔNG BẢO MẬT")
            print("=" * 70)
            comparison = compare_captures(secure_file, insecure_file)
            
            print("\n" + "=" * 70)
            print("KẾT QUẢ KIỂM TRA MÃ HÓA")
            print("=" * 70)
            print(f"Kết nối bảo mật sử dụng TLS/SSL có bằng chứng mã hóa: {'✓ Có' if secure_is_encrypted else '✗ Không'}")
            print(f"Kết nối không bảo mật có bằng chứng mã hóa: {'✓ Có' if insecure_is_encrypted else '✗ Không'}")
            
            if not secure_plaintext and insecure_plaintext:
                result = "✅ KIỂM TRA THÀNH CÔNG: Kết nối bảo mật đã mã hóa dữ liệu thành công!"
                print(f"\n{result}")
                print("  - Không tìm thấy dữ liệu văn bản thuần túy trong bản ghi kết nối bảo mật")
                print("  - Tìm thấy dữ liệu văn bản thuần túy trong bản ghi kết nối không bảo mật")
            elif not secure_plaintext:
                result = "✅ KIỂM TRA THÀNH CÔNG MỘT PHẦN: Không tìm thấy văn bản thuần túy trong kết nối bảo mật"
                print(f"\n{result}")
                print("  - Tuy nhiên, cũng không tìm thấy văn bản thuần túy trong bản ghi không bảo mật")
                print("  - Bài kiểm tra có thể cần được cải tiến để phát hiện dữ liệu mô hình một cách đáng tin cậy hơn")
            else:
                result = "❌ KIỂM TRA THẤT BẠI: Dữ liệu văn bản thuần túy được tìm thấy trong bản ghi bảo mật!"
                print(f"\n{result}")
                print("  - Điều này cho thấy mã hóa TLS/SSL có thể không hoạt động như mong đợi")
            
            # Save results to log file
            with open(log_file, "a") as f:
                f.write("KẾT QUẢ KIỂM TRA SO SÁNH:\n")
                f.write(f"- Kết nối bảo mật được mã hóa: {'Có' if secure_is_encrypted else 'Không'}\n")
                f.write(f"- Kết nối không bảo mật được mã hóa: {'Có' if insecure_is_encrypted else 'Không'}\n")
                f.write(f"- Đánh giá cuối cùng: {result}\n\n")
                f.write("CHỈ SỐ CHI TIẾT:\n")
                f.write(f"- Kích thước bản ghi bảo mật: {comparison['secure_size'] / 1024:.2f} KB\n")
                f.write(f"- Kích thước bản ghi không bảo mật: {comparison['insecure_size'] / 1024:.2f} KB\n")
                f.write(f"- Tiêu đề gRPC ẩn trong kết nối bảo mật: {'Có' if comparison['secure_grpc_hidden'] else 'Không'}\n")
                f.write(f"- Tiêu đề gRPC hiển thị trong kết nối không bảo mật: {'Có' if comparison['insecure_grpc_visible'] else 'Không'}\n")
        
        except Exception as e:
            print(f"\nLỗi trong quá trình kiểm tra kết nối không bảo mật: {e}")
            print("\n===== KẾT QUẢ KIỂM TRA MÃ HÓA (CHỈ KẾT NỐI BẢO MẬT) =====")
            print(f"Kết nối bảo mật sử dụng TLS/SSL có bằng chứng mã hóa: {'✓ Có' if secure_is_encrypted else '✗ Không'}")
            
            if not secure_plaintext:
                result = "✅ KIỂM TRA THÀNH CÔNG MỘT PHẦN: Không tìm thấy văn bản thuần túy trong kết nối bảo mật"
                print(f"\n{result}")
                print("  - Kiểm tra so sánh không được thực hiện thành công")
            else:
                result = "❌ KIỂM TRA THẤT BẠI: Dữ liệu văn bản thuần túy được tìm thấy trong bản ghi bảo mật!"
                print(f"\n{result}")
            
            # Save results to log file
            with open(log_file, "a") as f:
                f.write("KẾT QUẢ KIỂM TRA CHỈ BẢO MẬT:\n")
                f.write(f"- Kết nối bảo mật được mã hóa: {'Có' if secure_is_encrypted else 'Không'}\n")
                f.write(f"- Đánh giá cuối cùng: {result}\n\n")
    else:
        # Chỉ chạy kiểm tra bảo mật
        print("\n" + "=" * 70)
        print("KẾT QUẢ KIỂM TRA MÃ HÓA (CHỈ KẾT NỐI BẢO MẬT)")
        print("=" * 70)
        print(f"Kết nối bảo mật sử dụng TLS/SSL có bằng chứng mã hóa: {'✓ Có' if secure_is_encrypted else '✗ Không'}")
        
        if not secure_plaintext:
            result = "✅ KIỂM TRA THÀNH CÔNG: Không tìm thấy văn bản thuần túy trong kết nối bảo mật"
            print(f"\n{result}")
            print("  - Để có bài kiểm tra đầy đủ hơn, hãy cài đặt tshark và chạy kiểm tra so sánh")
        else:
            result = "❌ KIỂM TRA THẤT BẠI: Dữ liệu văn bản thuần túy được tìm thấy trong bản ghi bảo mật!"
            print(f"\n{result}")
        
        # Save results to log file
        with open(log_file, "a") as f:
            f.write("KẾT QUẢ KIỂM TRA CHỈ BẢO MẬT:\n")
            f.write(f"- Kết nối bảo mật được mã hóa: {'Có' if secure_is_encrypted else 'Không'}\n")
            f.write(f"- Đánh giá cuối cùng: {result}\n\n")
    
    print("\nDọn dẹp: Đang xóa các tệp bắt gói tin tạm thời...")
    try:
        os.remove(secure_file)
        if test_insecure and 'insecure_file' in locals():
            os.remove(insecure_file)
        print("Đã hoàn tất dọn dẹp.")
    except Exception as e:
        print(f"Lỗi trong quá trình dọn dẹp: {e}")
    
    print(f"\nKết quả kiểm tra đã được lưu vào {log_file}")
    print("\nĐã hoàn tất kiểm tra xác minh mã hóa!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        cleanup()
