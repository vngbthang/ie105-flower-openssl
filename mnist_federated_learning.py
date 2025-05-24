#!/usr/bin/env python3
"""
Script này thực hiện huấn luyện mô hình MNIST đơn giản sử dụng học liên hợp (Federated Learning)
với framework Flower.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np

# Flower imports
import flwr as fl
from flwr.client import NumPyClient, start_client
from flwr.common import NDArrays, Scalar, FitRes

# ML libraries for our test dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

# Base directory for certificates
BASE_DIR = Path(__file__).parent.absolute()
CERT_DIR = BASE_DIR / "certs"
SERVER_PORT = 8443

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
        
        # Thêm một số thông báo để theo dõi trong quá trình train
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

def run_server(secure=True):
    """Chạy Flower server."""
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

def main():
    """Hàm chính để chạy quá trình học liên hợp."""
    print("===== HỌC LIÊN HỢP (FEDERATED LEARNING) VỚI BỘ DỮ LIỆU MNIST =====")
    
    # Hỏi người dùng muốn chạy client hay server
    mode = input("Bạn muốn chạy phần nào? (1: Server, 2: Client): ")
    
    # Hỏi người dùng có muốn sử dụng TLS/SSL không
    use_ssl = input("Bạn có muốn sử dụng kết nối bảo mật TLS/SSL không? (y/n): ").lower() == 'y'
    
    if mode == "1":
        run_server(secure=use_ssl)
    elif mode == "2":
        run_client(secure=use_ssl)
    else:
        print("Lựa chọn không hợp lệ. Vui lòng chọn 1 (Server) hoặc 2 (Client).")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nChương trình đã bị dừng bởi người dùng.")
        sys.exit(0)
