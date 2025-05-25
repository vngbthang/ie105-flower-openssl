#!/usr/bin/env python3
"""
Script để chạy federated learning với MNIST dataset
sử dụng flwr-datasets và Flower framework với TLS/SSL
"""

import os
import sys
from pathlib import Path
import argparse
import threading
import time

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Flower imports
import flwr as fl
from flwr.common import NDArrays, Scalar, FitRes

# Thư viện local
from mnist_flower_datasets import load_mnist_partitions

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

def train(model, dataloader, epochs, device):
    """Huấn luyện mô hình trong một số epoch."""
    print("Bắt đầu huấn luyện mô hình...", flush=True)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}", flush=True)
        batch_count = 0
        for batch in dataloader:
            # Xác định đúng tên trường cho image và label
            if "image" in batch:
                images, labels = batch["image"], batch["label"]
            elif "img" in batch:
                images, labels = batch["img"], batch["label"]
            else:
                # Lấy 2 trường đầu tiên trong batch, giả định là image và label
                keys = list(batch.keys())
                images, labels = batch[keys[0]], batch[keys[1]]
            
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_count += 1
            if batch_count % 10 == 0:
                print(f"  Đã xử lý {batch_count} batches", flush=True)
    print("Hoàn thành huấn luyện mô hình", flush=True)

def test(model, dataloader, device):
    """Đánh giá mô hình trên tập dữ liệu kiểm tra."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            # Xác định đúng tên trường cho image và label
            if "image" in batch:
                images, labels = batch["image"], batch["label"]
            elif "img" in batch:
                images, labels = batch["img"], batch["label"]
            else:
                # Lấy 2 trường đầu tiên trong batch, giả định là image và label
                keys = list(batch.keys())
                images, labels = batch[keys[0]], batch[keys[1]]
                
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return loss / len(dataloader), accuracy

class MnistClient(fl.client.NumPyClient):
    """Client cho học liên hợp MNIST"""
    def __init__(self, model, trainloader, testloader, client_id):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.client_id = client_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def get_parameters(self, config):
        """Lấy tham số mô hình dưới dạng danh sách các mảng NumPy."""
        print(f"Client {self.client_id}: Đang lấy tham số mô hình...", flush=True)
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
        """Huấn luyện mô hình trên tập dữ liệu cục bộ."""
        print(f"Client {self.client_id}: Đang nhận tham số từ server...", flush=True)
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        epochs = config.get("epochs", 1) if config else 1
        print(f"Client {self.client_id}: Training for {epochs} epochs", flush=True)
        
        train(self.model, self.trainloader, epochs=epochs, device=self.device)
        
        print(f"Client {self.client_id}: Đang gửi tham số mô hình đến server...", flush=True)
        
        return self.get_parameters(config), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        """Đánh giá mô hình trên tập dữ liệu cục bộ."""
        print(f"Client {self.client_id}: Đang đánh giá mô hình...", flush=True)
        
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        loss, accuracy = test(self.model, self.testloader, device=self.device)
        print(f"Client {self.client_id}: Độ chính xác: {accuracy:.4f}, Loss: {loss:.4f}", flush=True)
        
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

def run_client(client_id, secure=True):
    """Chạy Flower client với một mô hình MNIST."""
    # Tải dữ liệu
    print(f"Client {client_id}: Đang tải dữ liệu...", flush=True)
    train_loaders, test_loader = load_mnist_partitions(num_partitions=3)
    
    # Chọn partition phù hợp với client_id
    trainloader = train_loaders[client_id % len(train_loaders)]
    
    # Khởi tạo mô hình
    print(f"Client {client_id}: Đang khởi tạo mô hình...", flush=True)
    model = MnistNet()
    
    # Tạo client
    client = MnistClient(model, trainloader, test_loader, client_id)
    
    if secure:
        print(f"Client {client_id}: Đang khởi động với TLS/SSL...", flush=True)
        try:
            # Kiểm tra tồn tại chứng chỉ CA
            ca_path = CERT_DIR / "ca/ca.pem"
            if not ca_path.exists():
                raise FileNotFoundError(f"Không tìm thấy chứng chỉ CA: {ca_path}")
            
            print(f"Client {client_id}: Đã tìm thấy chứng chỉ CA tại: {ca_path}", flush=True)
            with open(ca_path, "rb") as f:
                ca_cert = f.read()
            
            # Kiểm tra xem chứng chỉ CA có hợp lệ không
            if not ca_cert or len(ca_cert) < 100:  # Kiểm tra đơn giản
                raise ValueError(f"Client {client_id}: Chứng chỉ CA có vẻ không hợp lệ")
                
            print(f"Client {client_id}: Kết nối đến server tại localhost:{SERVER_PORT} với TLS/SSL", flush=True)
            
            # Khởi động client với TLS/SSL
            fl.client.start_client(
                server_address=f"localhost:{SERVER_PORT}",
                client=client,
                root_certificates=ca_cert
            )
            print(f"Client {client_id}: Đã hoàn thành quá trình federated learning", flush=True)
        except FileNotFoundError as e:
            print(f"Client {client_id}: Lỗi khi tìm tệp chứng chỉ: {e}", flush=True)
            print(f"Client {client_id}: Thử lại không bảo mật...", flush=True)
            fl.client.start_client(
                server_address=f"localhost:8080",
                client=client
            )
        except ValueError as e:
            print(f"Client {client_id}: Lỗi với chứng chỉ: {e}", flush=True)
            print(f"Client {client_id}: Thử lại không bảo mật...", flush=True)
            fl.client.start_client(
                server_address=f"localhost:8080",
                client=client
            )
        except Exception as e:
            print(f"Client {client_id}: Lỗi khi khởi động với TLS/SSL: {str(e)}", flush=True)
            print(f"Client {client_id}: Thử lại không bảo mật...", flush=True)
            fl.client.start_client(
                server_address=f"localhost:8080",
                client=client
            )
    else:
        print(f"Client {client_id}: Đang khởi động không bảo mật...", flush=True)
        # Khởi động client không bảo mật
        fl.client.start_client(
            server_address="localhost:8080",
            client=client
        )

def run_server(secure=True, num_rounds=3):
    """Chạy Flower server."""
    print(f"Server: Khởi động với {num_rounds} rounds...", flush=True)
    
    # Tạo chiến lược
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=1,
        min_fit_clients=1,
        min_evaluate_clients=1,
        on_fit_config_fn=lambda server_round: {
            "server_round": server_round,
            "epochs": 1,
            "batch_size": 64
        },
        on_evaluate_config_fn=lambda server_round: {
            "server_round": server_round
        }
    )
    
    if secure:
        print("Server: Đang khởi động với TLS/SSL...", flush=True)
        try:
            # Kiểm tra và tải các chứng chỉ
            server_cert_path = CERT_DIR / "server/server.pem"
            server_key_path = CERT_DIR / "server/server.key"
            ca_path = CERT_DIR / "ca/ca.pem"
            
            # Kiểm tra tất cả các tệp chứng chỉ
            for path in [server_cert_path, server_key_path, ca_path]:
                if not path.exists():
                    raise FileNotFoundError(f"Không tìm thấy file: {path}")
                print(f"Server: Đã tìm thấy file chứng chỉ: {path}", flush=True)
            
            # Đọc nội dung các file chứng chỉ
            with open(server_cert_path, "rb") as f:
                server_cert = f.read()
                if not server_cert or len(server_cert) < 100:  # Kiểm tra đơn giản
                    raise ValueError("Server cert có vẻ không hợp lệ")
                    
            with open(server_key_path, "rb") as f:
                server_key = f.read()
                if not server_key or len(server_key) < 100:  # Kiểm tra đơn giản
                    raise ValueError("Server key có vẻ không hợp lệ")
                    
            with open(ca_path, "rb") as f:
                ca_cert = f.read()
                if not ca_cert or len(ca_cert) < 100:  # Kiểm tra đơn giản
                    raise ValueError("CA cert có vẻ không hợp lệ")
            
            print(f"Server: Đang lắng nghe tại cổng {SERVER_PORT} với TLS/SSL", flush=True)
            
            # Khởi động server với TLS/SSL
            fl.server.start_server(
                server_address=f"[::]:{SERVER_PORT}",
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=strategy,
                certificates=(server_cert, server_key, ca_cert)
            )
        except FileNotFoundError as e:
            print(f"Server: Lỗi khi tìm tệp chứng chỉ: {e}", flush=True)
            print("Server: Thử lại không bảo mật...", flush=True)
            fl.server.start_server(
                server_address="[::]:8080",
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=strategy
            )
        except ValueError as e:
            print(f"Server: Lỗi với chứng chỉ: {e}", flush=True)
            print("Server: Thử lại không bảo mật...", flush=True)
            fl.server.start_server(
                server_address="[::]:8080",
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=strategy
            )
        except Exception as e:
            print(f"Server: Lỗi khi khởi động với TLS/SSL: {str(e)}", flush=True)
            print("Server: Thử lại không bảo mật...", flush=True)
            fl.server.start_server(
                server_address="[::]:8080",
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=strategy
            )
    else:
        print("Server: Đang khởi động không bảo mật...", flush=True)
        # Khởi động server không bảo mật
        fl.server.start_server(
            server_address="[::]:8080",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy
        )

def run_simulation():
    """Chạy mô phỏng federated learning với 3 client."""
    print("Bắt đầu mô phỏng federated learning...", flush=True)
    
    # Tải dữ liệu
    train_loaders, test_loader = load_mnist_partitions(num_partitions=3)
    
    # Định nghĩa hàm client_fn
    def client_fn(cid):
        # Khởi tạo mô hình cho client
        model = MnistNet()
        # Chọn trainloader dựa trên client id
        trainloader = train_loaders[int(cid)]
        # Trả về MnistClient
        return MnistClient(model, trainloader, test_loader, int(cid))
    
    # Tạo chiến lược
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=3,
        min_fit_clients=3,
        min_evaluate_clients=3,
        on_fit_config_fn=lambda server_round: {
            "server_round": server_round,
            "epochs": 1,
            "batch_size": 64
        },
        on_evaluate_config_fn=lambda server_round: {
            "server_round": server_round
        }
    )
    
    # Chạy mô phỏng
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

def run_direct():
    """Chạy trực tiếp server và client trong các thread riêng biệt."""
    print("Bắt đầu chạy trực tiếp federated learning...", flush=True)
    
    # Khởi động server trong một thread riêng
    server_thread = threading.Thread(target=run_server, args=(True, 3))
    server_thread.daemon = True
    server_thread.start()
    
    # Đợi server khởi động
    print("Đợi 5 giây để server khởi động...", flush=True)
    time.sleep(5)
    
    # Khởi động các client trong các thread riêng
    client_threads = []
    for i in range(3):
        print(f"Khởi động client {i}...", flush=True)
        client_thread = threading.Thread(target=run_client, args=(i, True))
        client_thread.daemon = True
        client_thread.start()
        client_threads.append(client_thread)
        # Đợi một chút giữa việc khởi động các client
        time.sleep(2)
    
    # Thêm một thời gian chờ tối đa để không bị treo vô hạn
    max_runtime = 300  # Giới hạn thời gian chạy là 5 phút
    start_time = time.time()
    
    try:
        while time.time() - start_time < max_runtime:
            # Kiểm tra xem tất cả các client có hoạt động không
            time.sleep(5)
            print(f"Federated learning đang chạy... Đã chạy {int(time.time() - start_time)} giây", flush=True)
        
        print(f"\nĐã hết thời gian chờ tối đa ({max_runtime} giây). Kết thúc chương trình.", flush=True)
    except KeyboardInterrupt:
        print("\nNhận tín hiệu ngắt, đang kết thúc...", flush=True)

def main():
    """Hàm chính để chạy federated learning."""
    # Đảm bảo khai báo global biến SERVER_PORT trước khi sử dụng
    global SERVER_PORT
    
    parser = argparse.ArgumentParser(description='Federated Learning với MNIST')
    parser.add_argument('--mode', type=str, default='direct', 
                        choices=['server', 'client', 'simulation', 'direct'],
                        help='Chế độ chạy (server/client/simulation/direct)')
    parser.add_argument('--secure', action='store_true', 
                        help='Sử dụng TLS/SSL')
    parser.add_argument('--insecure', action='store_true', 
                        help='Không sử dụng TLS/SSL (ưu tiên hơn --secure)')
    parser.add_argument('--client-id', type=int, default=0,
                        help='ID của client (chỉ dùng cho mode client)')
    parser.add_argument('--rounds', type=int, default=3,
                        help='Số rounds cho federated learning')
    parser.add_argument('--port', type=int, default=SERVER_PORT,
                        help='Cổng kết nối cho server')
    args = parser.parse_args()
    
    # Cập nhật cổng kết nối nếu được chỉ định
    SERVER_PORT = args.port
    
    # Xác định sử dụng TLS/SSL hay không (ưu tiên --insecure)
    secure = args.secure and not args.insecure
    
    # Hiển thị thông tin cấu hình
    print(f"Cấu hình chạy:")
    print(f"- Mode: {args.mode}")
    print(f"- Secure: {secure}")
    print(f"- Rounds: {args.rounds}")
    print(f"- Server Port: {SERVER_PORT}")
    if args.mode == 'client':
        print(f"- Client ID: {args.client_id}")
    
    # Chạy theo mode được chỉ định
    try:
        if args.mode == 'server':
            run_server(secure=secure, num_rounds=args.rounds)
        elif args.mode == 'client':
            run_client(client_id=args.client_id, secure=secure)
        elif args.mode == 'simulation':
            run_simulation()
        elif args.mode == 'direct':
            run_direct()
        else:
            print(f"Mode không hợp lệ: {args.mode}")
            parser.print_help()
    except Exception as e:
        print(f"Lỗi không mong đợi: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nChương trình bị dừng bởi người dùng.")
        sys.exit(0)
