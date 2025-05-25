#!/usr/bin/env python3
"""
Script để tải MNIST dataset sử dụng flwr-datasets
và chia thành các partition cho federated learning
"""

import os
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# Import flwr-datasets
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner

# Số lượng partition muốn chia
NUM_PARTITIONS = 3

def load_mnist_partitions(num_partitions=NUM_PARTITIONS, batch_size=64, alpha=0.5):
    """
    Tải MNIST dataset và chia thành nhiều partition cho federated learning
    Sử dụng phân phối Dirichlet để tạo sự không đồng nhất giữa các partition
    
    Args:
        num_partitions: Số lượng partition
        batch_size: Kích thước batch cho DataLoader
        alpha: Tham số alpha cho phân phối Dirichlet (càng nhỏ càng không đồng nhất)
        
    Returns:
        train_dataloaders: List các DataLoader cho dữ liệu huấn luyện
        test_dataloader: DataLoader cho dữ liệu kiểm tra
    """
    print(f"Đang tải và chia MNIST dataset thành {num_partitions} partition...")
    
    # Tạo FederatedDataset với MNIST dataset
    try:
        # Sử dụng Dirichlet partitioner để tạo sự không đồng nhất
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={
                "train": DirichletPartitioner(
                    num_partitions=num_partitions,
                    partition_by="label",
                    alpha=alpha
                )
            }
        )
        print("Đã tải dataset thành công!")
    except Exception as e:
        print(f"Lỗi khi tải MNIST dataset: {e}")
        # Thử lại với IID partitioner nếu Dirichlet không hoạt động
        print("Thử lại với IID partitioner...")
        fds = FederatedDataset(
            dataset="ylecun/mnist",
            partitioners={"train": IidPartitioner(num_partitions=num_partitions)}
        )
    
    # Kiểm tra cấu trúc dữ liệu
    print(f"Cấu trúc dữ liệu: {fds.load_partition(0, 'train').features}")
    
    # Chuyển đổi hàm áp dụng transforms
    def apply_transforms(batch):
        # Kiểm tra cấu trúc của batch để xác định đúng tên trường
        if "image" in batch:
            img_field = "image"
        elif "img" in batch:
            img_field = "img"
        else:
            img_field = next(iter(batch.keys()))
            print(f"Warning: Không tìm thấy trường image/img, đang sử dụng '{img_field}'")
            
        batch[img_field] = [ToTensor()(img) for img in batch[img_field]]
        return batch
    
    # Tạo train dataloaders cho từng partition
    train_dataloaders = []
    for i in range(num_partitions):
        partition = fds.load_partition(i, "train")
        partition_torch = partition.with_transform(apply_transforms)
        train_dataloaders.append(DataLoader(partition_torch, batch_size=batch_size, shuffle=True))
        print(f"Partition {i} đã tải với {len(partition)} mẫu")
    
    # Tạo test dataloader từ tập test
    test_split = fds.load_split("test")
    test_torch = test_split.with_transform(apply_transforms)
    test_dataloader = DataLoader(test_torch, batch_size=batch_size)
    print(f"Test set đã tải với {len(test_split)} mẫu")
    
    return train_dataloaders, test_dataloader

if __name__ == "__main__":
    # Test script
    print("Bắt đầu tải MNIST dataset sử dụng flwr-datasets...")
    try:
        # Tạm thời tắt tiến trình chờ để dễ debug
        import os
        os.environ["DATASETS_PROGRESS_BAR"] = "0"
        
        print("Đang tải MNIST partitions...")
        train_dataloaders, test_dataloader = load_mnist_partitions(num_partitions=3)
        print(f"Đã tạo {len(train_dataloaders)} training dataloaders và 1 test dataloader")
        
        # Hiển thị phân phối label trong mỗi partition
        print("\nPhân phối label trong mỗi partition:")
        for i, dataloader in enumerate(train_dataloaders):
            print(f"Đang kiểm tra dataloader {i}...")
            labels = []
            
            # Lấy batch đầu tiên để xem cấu trúc
            first_batch = next(iter(dataloader))
            print(f"Cấu trúc batch: {first_batch.keys()}")
            
            # Lấy labels từ một số batch
            batch_count = 0
            for batch in dataloader:
                batch_count += 1
                if "label" in batch:
                    labels.extend(batch["label"].tolist())
                else:
                    try:
                        # Tìm trường label thích hợp
                        label_field = next(key for key in batch.keys() if key != "image" and key != "img")
                        labels.extend(batch[label_field].tolist())
                    except (StopIteration, KeyError) as e:
                        print(f"Lỗi khi tìm trường label: {e}, batch keys: {batch.keys()}")
                
                if len(labels) >= 100 or batch_count >= 5:  # Chỉ lấy mẫu để hiển thị nhanh
                    break
                    
            print(f"Đã xử lý {batch_count} batches, thu thập {len(labels)} labels")
                    
            # Đếm số lượng mỗi label
            label_counts = {}
            for label in labels:
                if label in label_counts:
                    label_counts[label] += 1
                else:
                    label_counts[label] = 1
                    
            print(f"Partition {i}: {label_counts}")
        
        print("\nScript test thành công!")
    except Exception as e:
        import traceback
        print(f"Lỗi khi chạy script test: {e}")
        traceback.print_exc()
        print("Vui lòng đảm bảo đã cài đặt flwr-datasets: pip install 'flwr-datasets[vision]'")
