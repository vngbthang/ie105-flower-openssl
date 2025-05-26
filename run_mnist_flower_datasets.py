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
import socket
import logging

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Logger cho script này
logger = logging.getLogger('flower-mnist')

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
SERVER_PORT = 18443  # Cổng mặc định đã thay đổi sang cao hơn
FALLBACK_PORT = 28443  # Cổng dự phòng thứ hai nếu không thể bind
INSECURE_PORT = 18080  # Cổng mặc định cho kết nối không bảo mật
SERVER_HOST = "localhost"

# Kiểm tra môi trường
logger.info(f"Python version: {sys.version}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"Flower version: {fl.__version__}")
logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Certificate directory: {CERT_DIR}")

# Kiểm tra nếu chạy với quyền root (không khuyến nghị)
if os.geteuid() == 0:
    logger.warning("Đang chạy với quyền root. Điều này không được khuyến nghị cho bảo mật.")

# Kiểm tra xem các thư mục chứng chỉ có tồn tại không
if not CERT_DIR.exists():
    logger.error(f"Thư mục chứng chỉ {CERT_DIR} không tồn tại!")
    logger.info("Hãy chạy regenerate_certificates.sh để tạo các chứng chỉ.")
else:
    logger.info("Thư mục chứng chỉ đã tồn tại.")
    
# Kiểm tra kết nối mạng cơ bản
def check_network_connectivity():
    try:
        # Kiểm tra xem localhost có thể bind không
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))  # Bind vào một port ngẫu nhiên
            random_port = s.getsockname()[1]
            logger.info(f"Kiểm tra mạng: Đã bind thành công vào 127.0.0.1:{random_port}")
        return True
    except Exception as e:
        logger.error(f"Kiểm tra mạng thất bại: {e}")
        return False

# Chạy kiểm tra mạng
check_network_connectivity()

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
    logger.info("Bắt đầu huấn luyện mô hình...")
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
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
                logger.info(f"  Đã xử lý {batch_count} batches")
    logger.info("Hoàn thành huấn luyện mô hình")

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
        logger.info(f"Client {self.client_id}: Đang lấy tham số mô hình...")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def fit(self, parameters, config):
        """Huấn luyện mô hình trên tập dữ liệu cục bộ."""
        logger.info(f"Client {self.client_id}: Đang nhận tham số từ server...")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        epochs = config.get("epochs", 1) if config else 1
        logger.info(f"Client {self.client_id}: Training for {epochs} epochs")
        
        train(self.model, self.trainloader, epochs=epochs, device=self.device)
        
        logger.info(f"Client {self.client_id}: Đang gửi tham số mô hình đến server...")
        
        return self.get_parameters(config), len(self.trainloader.dataset), {}
    
    def evaluate(self, parameters, config):
        """Đánh giá mô hình trên tập dữ liệu cục bộ."""
        logger.info(f"Client {self.client_id}: Đang đánh giá mô hình...")
        
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        
        loss, accuracy = test(self.model, self.testloader, device=self.device)
        logger.info(f"Client {self.client_id}: Độ chính xác: {accuracy:.4f}, Loss: {loss:.4f}")
        
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

def run_client(client_id, secure=True):
    """Chạy Flower client với một mô hình MNIST."""
    # Tải dữ liệu
    logger.info(f"Client {client_id}: Đang tải dữ liệu...")
    train_loaders, test_loader = load_mnist_partitions(num_partitions=3)
    
    # Chọn partition phù hợp với client_id
    trainloader = train_loaders[client_id % len(train_loaders)]
    
    # Khởi tạo mô hình
    logger.info(f"Client {client_id}: Đang khởi tạo mô hình...")
    model = MnistNet()
    
    # Tạo client
    client = MnistClient(model, trainloader, test_loader, client_id)
    
    if secure:
        logger.info(f"Client {client_id}: Đang khởi động với TLS/SSL...")
        try:
            # Kiểm tra tồn tại chứng chỉ CA
            ca_path = CERT_DIR / "ca/ca.pem"
            if not ca_path.exists():
                raise FileNotFoundError(f"Không tìm thấy chứng chỉ CA: {ca_path}")
            
            logger.info(f"Client {client_id}: Đã tìm thấy chứng chỉ CA tại: {ca_path}")
            with open(ca_path, "rb") as f:
                ca_cert = f.read()
            
            # Kiểm tra xem chứng chỉ CA có hợp lệ không
            if not ca_cert or len(ca_cert) < 100:  # Kiểm tra đơn giản
                raise ValueError(f"Client {client_id}: Chứng chỉ CA có vẻ không hợp lệ")
                
            logger.info(f"Client {client_id}: Kết nối đến server tại {SERVER_HOST}:{SERVER_PORT} với TLS/SSL")
            
            # Danh sách các cổng để thử
            ports_to_try = [SERVER_PORT, FALLBACK_PORT, 18080, 48443, 58443]
            connected = False
            last_error = None
            
            for port in ports_to_try:
                try:
                    logger.info(f"Client {client_id}: Đang thử kết nối đến {SERVER_HOST}:{port} với TLS/SSL...")
                    
                    # Check if server is running with a socket connect test
                    try:
                        import socket
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                            sock.settimeout(3)
                            result = sock.connect_ex((SERVER_HOST, port))
                            if result != 0:
                                logger.warning(f"Client {client_id}: Không thấy server đang chạy ở {SERVER_HOST}:{port}")
                                continue
                    except Exception as sock_err:
                        logger.warning(f"Client {client_id}: Lỗi khi kiểm tra kết nối: {sock_err}")
                    
                    # Khởi động client với TLS/SSL - API chuẩn mới cho Flower 1.18.0
                    logger.info(f"Client {client_id}: Kết nối đến {SERVER_HOST}:{port} sử dụng API chuẩn với TLS/SSL...")
                    fl.client.start_client(
                        server_address=f"{SERVER_HOST}:{port}",
                        client=client,
                        root_certificates=ca_cert
                    )
                    
                    # Nếu đến đây, client đã kết nối thành công
                    logger.info(f"Client {client_id}: Đã kết nối thành công đến {SERVER_HOST}:{port}")
                    logger.info(f"Client {client_id}: Đã hoàn thành quá trình federated learning")
                    connected = True
                    break
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"Client {client_id}: Không thể kết nối đến {SERVER_HOST}:{port}: {str(e)}")
                    # Thử port tiếp theo
                    continue
            
            if not connected:
                logger.error(f"Client {client_id}: Không thể kết nối đến server trên bất kỳ cổng nào. Lỗi cuối cùng: {str(last_error)}")
                raise last_error
        except FileNotFoundError as e:
            logger.error(f"Client {client_id}: Lỗi khi tìm tệp chứng chỉ: {e}")
            
            # Thử nhiều cổng không bảo mật
            insecure_ports = [18080, 8080]
            connected = False
            
            for insecure_port in insecure_ports:
                try:
                    logger.info(f"Client {client_id}: Thử kết nối không bảo mật với cổng {insecure_port}...")
                    
                    # Check if server is running with a socket connect test
                    try:
                        import socket
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                            sock.settimeout(3)
                            result = sock.connect_ex((SERVER_HOST, insecure_port))
                            if result != 0:
                                logger.warning(f"Client {client_id}: Không thấy server đang chạy ở {SERVER_HOST}:{insecure_port}")
                                continue
                    except Exception as sock_err:
                        logger.warning(f"Client {client_id}: Lỗi khi kiểm tra kết nối: {sock_err}")
                        continue
                    
                    fl.client.start_client(
                        server_address=f"{SERVER_HOST}:{insecure_port}",
                        client=client
                    )
                    logger.info(f"Client {client_id}: Đã kết nối thành công đến {SERVER_HOST}:{insecure_port} (không bảo mật)")
                    connected = True
                    break
                except Exception as port_error:
                    logger.warning(f"Client {client_id}: Không thể kết nối đến {SERVER_HOST}:{insecure_port}: {port_error}")
            
            if not connected:
                logger.error(f"Client {client_id}: Không thể kết nối không bảo mật đến bất kỳ cổng nào")
                raise
        except ValueError as e:
            logger.error(f"Client {client_id}: Lỗi với chứng chỉ: {e}")
            
            # Thử nhiều cổng không bảo mật
            insecure_ports = [18080, 8080]
            connected = False
            
            for insecure_port in insecure_ports:
                logger.info(f"Client {client_id}: Thử kết nối không bảo mật với cổng {insecure_port}...")
                
                try:
                    # Kiểm tra xem server có đang chạy không
                    import socket
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        sock.settimeout(3)
                        result = sock.connect_ex((SERVER_HOST, insecure_port))
                        if result != 0:
                            logger.warning(f"Client {client_id}: Không thấy server đang chạy ở {SERVER_HOST}:{insecure_port}")
                            continue
                            
                    # Thử kết nối không bảo mật
                    fl.client.start_client(
                        server_address=f"{SERVER_HOST}:{insecure_port}",
                        client=client
                    )
                    logger.info(f"Client {client_id}: Đã kết nối thành công đến {SERVER_HOST}:{insecure_port} (không bảo mật)")
                    connected = True
                    break
                except Exception as port_error:
                    logger.warning(f"Client {client_id}: Không thể kết nối đến {SERVER_HOST}:{insecure_port}: {port_error}")
            
            if not connected:
                logger.error(f"Client {client_id}: Không thể kết nối không bảo mật đến bất kỳ cổng nào")
                raise Exception("Không thể kết nối đến server")
                
        except Exception as e:
            logger.error(f"Client {client_id}: Lỗi khi khởi động với TLS/SSL: {str(e)}")
            
            # Thử nhiều cổng không bảo mật
            insecure_ports = [18080, 8080]
            connected = False
            
            for insecure_port in insecure_ports:
                logger.info(f"Client {client_id}: Thử kết nối không bảo mật với cổng {insecure_port}...")
                
                try:
                    # Kiểm tra xem server có đang chạy không
                    import socket
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        sock.settimeout(3)
                        result = sock.connect_ex((SERVER_HOST, insecure_port))
                        if result != 0:
                            logger.warning(f"Client {client_id}: Không thấy server đang chạy ở {SERVER_HOST}:{insecure_port}")
                            continue
                            
                    # Thử kết nối không bảo mật
                    fl.client.start_client(
                        server_address=f"{SERVER_HOST}:{insecure_port}",
                        client=client
                    )
                    logger.info(f"Client {client_id}: Đã kết nối thành công đến {SERVER_HOST}:{insecure_port} (không bảo mật)")
                    connected = True
                    break
                except Exception as port_error:
                    logger.warning(f"Client {client_id}: Không thể kết nối đến {SERVER_HOST}:{insecure_port}: {port_error}")
            
            if not connected:
                logger.error(f"Client {client_id}: Không thể kết nối không bảo mật đến bất kỳ cổng nào")
                raise Exception("Không thể kết nối đến server")
    else:
        logger.info(f"Client {client_id}: Đang khởi động không bảo mật...")
        # Khởi động client không bảo mật
        try:
            logger.info(f"Client {client_id}: Thử kết nối không bảo mật đến {SERVER_HOST}:18080...")
            fl.client.start_client(
                server_address=f"{SERVER_HOST}:18080",
                client=client
            )
            logger.info(f"Client {client_id}: Đã kết nối thành công đến {SERVER_HOST}:18080 (không bảo mật)")
        except Exception as e:
            # Thử port 8080 làm dự phòng
            logger.warning(f"Client {client_id}: Không thể kết nối đến {SERVER_HOST}:18080: {str(e)}")
            logger.info(f"Client {client_id}: Đang thử port 8080 thay thế...")
            fl.client.start_client(
                server_address=f"{SERVER_HOST}:8080",
                client=client
            )
            logger.info(f"Client {client_id}: Đã kết nối thành công đến {SERVER_HOST}:8080 (không bảo mật)")

def run_server(secure=True, num_rounds=3):
    """Chạy Flower server."""
    global SERVER_PORT
    logger.info(f"Server: Khởi động với {num_rounds} rounds...")
    
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
        logger.info("Server: Đang khởi động với TLS/SSL...")
        try:
            # Kiểm tra và tải các chứng chỉ
            server_cert_path = CERT_DIR / "server/server.pem"
            server_key_path = CERT_DIR / "server/server.key"
            ca_path = CERT_DIR / "ca/ca.pem"
            
            # Kiểm tra tất cả các tệp chứng chỉ
            for path in [server_cert_path, server_key_path, ca_path]:
                if not path.exists():
                    raise FileNotFoundError(f"Không tìm thấy file: {path}")
                logger.info(f"Server: Đã tìm thấy file chứng chỉ: {path}")
            
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
            
            logger.info(f"Server: Đang lắng nghe tại {SERVER_HOST}:{SERVER_PORT} với TLS/SSL")
            
            # Khởi động server với TLS/SSL
            # Luôn tạo lại chain certificate (server + CA) để đảm bảo nó hợp lệ
            chain_path = CERT_DIR / "server/chain.pem"
            logger.info(f"Server: Tạo lại chain certificate từ server cert và CA cert...")
            
            # Tạo chain certificate bằng cách kết hợp server cert và CA cert
            with open(server_cert_path, "rb") as f_server:
                server_cert_data = f_server.read()
            with open(ca_path, "rb") as f_ca:
                ca_cert_data = f_ca.read()
            with open(chain_path, "wb") as f_chain:
                f_chain.write(server_cert_data)
                f_chain.write(ca_cert_data)
            
            # Set permissions
            os.chmod(chain_path, 0o644)  # Read for all, write for owner
            logger.info(f"Server: Đã tạo chain certificate tại {chain_path}")
            
            # Đọc chain certificate
            with open(chain_path, "rb") as f:
                chain_cert = f.read()
                if not chain_cert or len(chain_cert) < 100:
                    raise ValueError("Chain cert có vẻ không hợp lệ")
            
            # Chuẩn bị các port để thử
            ports_to_try = [SERVER_PORT, FALLBACK_PORT, 18080, 48443, 58443]
            bind_addresses = ["0.0.0.0", "localhost"]
            
            server_started = False
            last_error = None
            
            # Thử từng cổng và địa chỉ kết hợp cho đến khi thành công
            for bind_address in bind_addresses:
                if server_started:
                    break
                    
                for port in ports_to_try:
                    try:
                        bind_endpoint = f"{bind_address}:{port}"
                        logger.info(f"Server: Đang thử lắng nghe tại {bind_endpoint} với TLS/SSL...")
                        
                        # Test socket binding trước
                        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                            try:
                                s.bind((bind_address if bind_address != "localhost" else "127.0.0.1", port))
                                logger.info(f"Binding test success on {bind_endpoint}")
                            except Exception as socket_err:
                                logger.warning(f"Không thể bind socket tới {bind_endpoint}: {socket_err}")
                                continue
                        
                        # Khởi động server với TLS/SSL
                        # Sử dụng API mới của Flower 1.18.0
                        try:
                            logger.info(f"Server: Thử sử dụng API mới start_server_secure...")
                            # API mới trong phiên bản Flower mới
                            import flwr as fl
                            fl.server.start_server_secure(
                                server_address=bind_endpoint,
                                config=fl.server.ServerConfig(num_rounds=num_rounds),
                                strategy=strategy,
                                certificates=(
                                    server_cert,  # server_cert
                                    server_key,   # server_key
                                    ca_cert       # root_certificates
                                )
                            )
                        except AttributeError:
                            # Fallback nếu không có API mới
                            logger.info(f"Server: API mới không khả dụng, sử dụng API cũ...")
                            fl.server.start_server(
                                server_address=bind_endpoint,
                                config=fl.server.ServerConfig(num_rounds=num_rounds),
                                strategy=strategy,
                                certificates=(chain_cert, server_key, ca_cert)
                            )
                        
                        # Nếu đến đây, server đã khởi động thành công
                        SERVER_PORT = port  # Cập nhật port toàn cục
                        logger.info(f"Server: Khởi động thành công tại {bind_endpoint}")
                        server_started = True
                        break
                        
                    except Exception as e:
                        last_error = e
                        logger.warning(f"Server: Không thể khởi động tại {bind_endpoint}: {str(e)}")
                        continue
            
            if not server_started:
                logger.error(f"Server: Không thể khởi động trên bất kỳ địa chỉ và cổng nào. Lỗi cuối cùng: {str(last_error)}")
                raise last_error
            
        except FileNotFoundError as e:
            logger.error(f"Server: Lỗi khi tìm tệp chứng chỉ: {e}")
            logger.info("Server: Thử lại không bảo mật với cổng 18080...")
            fl.server.start_server(
                server_address="[::]:18080",
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=strategy
            )
        except ValueError as e:
            logger.error(f"Server: Lỗi với chứng chỉ: {e}")
            logger.info("Server: Thử lại không bảo mật với cổng 18080...")
            fl.server.start_server(
                server_address="[::]:18080",
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=strategy
            )
        except Exception as e:
            logger.error(f"Server: Lỗi khi khởi động với TLS/SSL: {str(e)}")
            logger.info("Server: Thử lại không bảo mật với cổng 18080...")
            try:
                fl.server.start_server(
                    server_address="[::]:18080",
                    config=fl.server.ServerConfig(num_rounds=num_rounds),
                    strategy=strategy
                )
            except Exception as insecure_error:
                logger.error(f"Server: Không thể khởi động không bảo mật tại cổng 18080: {str(insecure_error)}")
                logger.info("Server: Thử với cổng 8080...")
                fl.server.start_server(
                    server_address="[::]:8080",
                    config=fl.server.ServerConfig(num_rounds=num_rounds),
                    strategy=strategy
                )
    else:
        logger.info("Server: Đang khởi động không bảo mật...")
        # Khởi động server không bảo mật
        try:
            logger.info("Server: Thử khởi động không bảo mật trên cổng 18080...")
            fl.server.start_server(
                server_address="[::]:18080",
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=strategy
            )
        except Exception as e:
            logger.error(f"Server: Không thể khởi động không bảo mật tại cổng 18080: {str(e)}")
            logger.info("Server: Thử với cổng 8080...")
            fl.server.start_server(
                server_address="[::]:8080",
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=strategy
            )

def run_simulation():
    """Chạy mô phỏng federated learning với nhiều client."""
    num_clients = args.num_clients
    logger.info(f"Bắt đầu mô phỏng federated learning với {num_clients} client và {args.rounds} vòng...")
    
    # Tải dữ liệu
    train_loaders, test_loader = load_mnist_partitions(num_partitions=num_clients)
    
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
    
    # Chạy mô phỏng với số vòng được chỉ định
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

def run_direct():
    """Chạy trực tiếp server và client trong các thread riêng biệt."""
    global NUM_CLIENTS, args
    
    # Sử dụng tham số num_clients nếu được cung cấp
    num_clients = args.num_clients
    
    logger.info(f"Bắt đầu chạy trực tiếp federated learning với {num_clients} client...")
    
    # Khởi động server trong một thread riêng
    server_thread = threading.Thread(target=run_server, args=(True, args.rounds))
    server_thread.daemon = True
    server_thread.start()
    
    # Đợi server khởi động
    logger.info("Đợi 5 giây để server khởi động...")
    time.sleep(5)
    
    # Khởi động các client trong các thread riêng
    client_threads = []
    for i in range(num_clients):
        logger.info(f"Khởi động client {i}...")
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
            logger.info(f"Federated learning đang chạy... Đã chạy {int(time.time() - start_time)} giây")
        
        logger.info(f"\nĐã hết thời gian chờ tối đa ({max_runtime} giây). Kết thúc chương trình.")
    except KeyboardInterrupt:
        logger.info("\nNhận tín hiệu ngắt, đang kết thúc...")

def main():
    """Hàm chính để chạy federated learning."""
    # Đảm bảo khai báo global biến SERVER_PORT và SERVER_HOST trước khi sử dụng
    global SERVER_PORT, SERVER_HOST, args
    
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
    parser.add_argument('--host', type=str, default='localhost',
                        help='Địa chỉ máy chủ (mặc định: localhost)')
    parser.add_argument('--verbose', action='store_true',
                        help='Hiển thị thông tin chi tiết')
    parser.add_argument('--num-clients', type=int, default=3,
                        help='Số lượng clients cho direct mode và simulation')
    args = parser.parse_args()
    
    # Cập nhật cổng kết nối nếu được chỉ định
    SERVER_PORT = args.port
    SERVER_HOST = args.host
    
    # Xác định sử dụng TLS/SSL hay không (ưu tiên --insecure)
    secure = args.secure and not args.insecure
    
    # Hiển thị thông tin cấu hình
    logger.info(f"Cấu hình chạy:")
    logger.info(f"- Mode: {args.mode}")
    logger.info(f"- Secure: {secure}")
    logger.info(f"- Rounds: {args.rounds}")
    logger.info(f"- Server Host: {SERVER_HOST}")
    logger.info(f"- Server Port: {SERVER_PORT}")
    if args.mode == 'client':
        logger.info(f"- Client ID: {args.client_id}")
    if args.verbose:
        logger.info(f"- Verbose: Bật")
    if args.mode == 'direct':
        logger.info(f"- Số lượng clients: {args.num_clients}")
    
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
            logger.error(f"Mode không hợp lệ: {args.mode}")
            parser.print_help()
    except Exception as e:
        logger.error(f"Lỗi không mong đợi: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nChương trình bị dừng bởi người dùng.")
        sys.exit(0)
