#!/usr/bin/env python3
"""
Flower client strategy implementation cho MNIST.
File này chứa NumPyClient implementation sử dụng với flower-supernode CLI.

To use this with flower-supernode:
    flower-supernode --insecure --superlink='localhost:18443'

Or with secure connection:
    flower-supernode \
        --superlink='localhost:18443' \
        --root-certificates='certs/ca/ca.pem'
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np

import flwr as fl
from flwr.client import NumPyClient

# Cài đặt logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("flower-mnist-supernode")

# Thêm đường dẫn để import các module cần thiết
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

# Import các module từ project
from mnist_flower_datasets import load_mnist_partitions

# Định nghĩa mô hình CNN đơn giản
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class MnistClient(NumPyClient):
    """Client tương thích với SuperNode cho MNIST."""
    
    def __init__(self, node_id, trainloader, valloader, num_examples):
        """Khởi tạo client với ID và tải dữ liệu."""
        self.client_id = node_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.num_examples = num_examples
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Client {node_id}: Sử dụng device {self.device}")
        
        # Khởi tạo mô hình
        logger.info(f"Client {node_id}: Đang khởi tạo mô hình...")
        self.net = Net().to(self.device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)
        
        # Log số lượng mẫu
        logger.info(f"Client {node_id}: Initialized với {num_examples['trainset']} mẫu train, {num_examples['testset']} mẫu val")

    def get_parameters(self, config):
        """Lấy tham số mô hình dưới dạng numpy arrays."""
        logger.info(f"Client {self.client_id}: Được yêu cầu lấy parameters")
        weights = [val.cpu().numpy() for _, val in self.net.state_dict().items()]
        return weights

    def set_parameters(self, parameters):
        """Cập nhật tham số mô hình từ list các numpy arrays."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.Tensor(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Huấn luyện mô hình trên dữ liệu local."""
        logger.info(f"Client {self.client_id}: Bắt đầu quá trình fit")
        self.set_parameters(parameters)
        
        # Lấy thông số huấn luyện từ config
        epochs = config.get("epochs", 1)
        batch_size = config.get("batch_size", 32)
        logger.info(f"Client {self.client_id}: Huấn luyện {epochs} epochs với batch size {batch_size}")
        
        # Huấn luyện mô hình
        self.net.train()
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for batch_idx, batch in enumerate(self.trainloader):
                images, labels = None, None
                if isinstance(batch, tuple) and len(batch) == 2:
                    # Direct tuple format (images, labels)
                    images, labels = batch
                elif isinstance(batch, dict):
                    # Dictionary format with fields
                    if "image" in batch and "label" in batch:
                        images, labels = batch["image"], batch["label"]
                    elif "img" in batch and "label" in batch:
                        images, labels = batch["img"], batch["label"]
                    else:
                        # Try to find appropriate fields
                        img_field = next((key for key in batch.keys() if "image" in key.lower() or "img" in key.lower()), None)
                        label_field = next((key for key in batch.keys() if "label" in key.lower()), None)
                        if img_field and label_field:
                            images, labels = batch[img_field], batch[label_field]
                        else:
                            logger.error(f"Client {self.client_id}: Could not identify image/label fields in batch: {batch.keys()}")
                            continue
                
                # Ensure images and labels are moved to device
                if images is not None and labels is not None:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                else:
                    logger.error(f"Client {self.client_id}: Images or labels are None, skipping batch")
                    continue
                
                # Forward pass, backward pass, và tối ưu
                self.optimizer.zero_grad()
                outputs = self.net(images)
                loss = F.nll_loss(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                # Cập nhật thống kê
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Log tiến trình
                if batch_idx % 10 == 0:
                    logger.info(f"Client {self.client_id}: Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(self.trainloader)}, Loss: {loss.item():.4f}, Acc: {(100 * correct / total):.2f}%")
            
            epoch_loss /= len(self.trainloader) if len(self.trainloader) > 0 else 1
            epoch_acc = correct / total if total > 0 else 0
            logger.info(f"Client {self.client_id}: Epoch {epoch+1}/{epochs} complete. Loss: {epoch_loss:.4f}, Accuracy: {100*epoch_acc:.2f}%")
        
        # Lấy tham số đã được cập nhật
        parameters_updated = self.get_parameters({})
        num_examples = self.num_examples["trainset"]
        return parameters_updated, num_examples, {"loss": epoch_loss, "accuracy": epoch_acc}

    def evaluate(self, parameters, config):
        """Đánh giá mô hình trên tập dữ liệu test."""
        logger.info(f"Client {self.client_id}: Bắt đầu quá trình evaluate")
        self.set_parameters(parameters)
        
        # Chuyển sang chế độ đánh giá
        self.net.eval()
        loss, correct, total = 0.0, 0, 0
        
        # Đánh giá không tính gradient
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valloader):
                images, labels = None, None
                if isinstance(batch, tuple) and len(batch) == 2:
                    # Direct tuple format
                    images, labels = batch
                elif isinstance(batch, dict):
                    # Dictionary format
                    if "image" in batch and "label" in batch:
                        images, labels = batch["image"], batch["label"]
                    elif "img" in batch and "label" in batch:
                        images, labels = batch["img"], batch["label"]
                    else:
                        # Try to find fields
                        img_field = next((key for key in batch.keys() if "image" in key.lower() or "img" in key.lower()), None)
                        label_field = next((key for key in batch.keys() if "label" in key.lower()), None)
                        if img_field and label_field:
                            images, labels = batch[img_field], batch[label_field]
                        else:
                            logger.error(f"Client {self.client_id}: Could not identify image/label fields in batch: {batch.keys()}")
                            continue
                
                # Move to device
                if images is not None and labels is not None:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                else:
                    logger.error(f"Client {self.client_id}: Images or labels are None, skipping batch")
                    continue
                
                # Forward pass
                outputs = self.net(images)
                loss += F.nll_loss(outputs, labels).item()
                
                # Tính accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Tính toán kết quả
        loss /= len(self.valloader) if len(self.valloader) > 0 else 1
        accuracy = correct / total if total > 0 else 0
        logger.info(f"Client {self.client_id}: Evaluate complete. Loss: {loss:.4f}, Accuracy: {100*accuracy:.2f}%")
        
        return loss, self.num_examples["testset"], {"accuracy": accuracy}

# Định nghĩa hàm client_fn để trả về client instance
def client_fn_for_app(cid: str) -> fl.client.Client:
    """Create a Flower client."""
    logger.info(f"[ClientApp] Creating client instance for CID: {cid}")
    
    int_cid = int(cid)
    
    # Load data partitions
    logger.info(f"[ClientApp] Loading data partitions...")
    train_dataloaders, test_dataloader, num_examples = load_data_for_client(int_cid)
    
    logger.info(f"[ClientApp] Using data partition {int_cid} for client CID {cid}")
    logger.info(f"[ClientApp] Client has {num_examples['trainset']} training and {num_examples['testset']} test samples")
    
    return MnistClient(
        node_id=int_cid,
        trainloader=train_dataloaders,
        valloader=test_dataloader,
        num_examples=num_examples
    )

def load_data_for_client(cid: int, num_partitions=3):
    """Load data for a specific client."""
    # Load partitioned data
    train_dataloaders, test_dataloader = load_mnist_partitions(num_partitions=num_partitions)
    
    # Select the appropriate partition for this client
    cid = min(cid, len(train_dataloaders)-1)  # Ensure cid is in bounds
    
    # Create num_examples dict (approximate counting)
    train_samples = sum(len(batch) for batch in train_dataloaders[cid])
    test_samples = sum(len(batch) for batch in test_dataloader)
    
    num_examples = {
        "trainset": train_samples, 
        "testset": test_samples
    }
    
    return train_dataloaders[cid], test_dataloader, num_examples

# Create a ClientApp instance
app = fl.client.ClientApp(client_fn=client_fn_for_app)

# For direct script execution (testing)
if __name__ == "__main__":
    logger.info("Script này được thiết kế để chạy với flower-supernode CLI.")
    logger.info("Khi chạy trực tiếp, nó sẽ chỉ kiểm tra cài đặt.")
    
    # Check environment
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Flower version: {fl.__version__}")
    logger.info(f"Device: {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}")
    
    # Test loading data for client 0
    logger.info("Testing data loading...")
    try:
        train_loader, test_loader, num_ex = load_data_for_client(0)
        logger.info(f"Successfully loaded data with {num_ex['trainset']} training samples and {num_ex['testset']} test samples")
        
        # Try to get a sample batch
        batch = next(iter(train_loader))
        if isinstance(batch, tuple):
            logger.info(f"Batch format: tuple with shapes {[b.shape for b in batch]}")
        elif isinstance(batch, dict):
            logger.info(f"Batch format: dict with keys {batch.keys()}")
        else:
            logger.info(f"Batch format: {type(batch)}")
            
        # Test client creation
        logger.info("Testing client creation...")
        client = client_fn_for_app("0")
        logger.info("Client created successfully")
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
