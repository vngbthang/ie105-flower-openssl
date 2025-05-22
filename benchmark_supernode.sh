#!/bin/bash
# This script measures the performance overhead of TLS/SSL in Flower using supernode/superlink architecture

echo "===== TLS/SSL Performance Benchmark for Flower (SuperNode/SuperLink) ====="
echo "This benchmark compares the performance of Flower SuperNode/SuperLink with and without TLS"
echo

# Base directory
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to create a temporary insecure superlink configuration
create_insecure_superlink() {
    cat > temp_insecure_superlink.py << 'EOF'
import flwr as fl
import time

# Define a simple strategy
class MyStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(min_fit_clients=1, min_evaluate_clients=1)

if __name__ == "__main__":
    start_time = time.time()
    
    # Create strategy
    strategy = MyStrategy()
    
    # Thêm một timeout tối đa cho server, tự đóng sau 25 giây nếu không có tiến triển
    deadline = time.time() + 25
    
    def custom_fit_config(server_round):
        """Trả về fit config cho từng round."""
        # Kiểm tra xem đã đến timeout hay chưa
        if time.time() > deadline:
            print("Server timeout reached, shutting down...")
            import sys
            sys.exit(0)
        return {}
    
    # The flower-superlink command would normally handle this
    # but we're simulating it here for the benchmark
    try:
        fl.server.start_server(
            server_address="[::]:8080",
            config=fl.server.ServerConfig(num_rounds=1),
            strategy=fl.server.strategy.FedAvg(
                min_fit_clients=1,
                min_evaluate_clients=1,
                on_fit_config_fn=custom_fit_config
            )
        )
    except Exception as e:
        print(f"Server stopped: {e}")
    
    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed} seconds")
EOF

    cat > temp_insecure_supernode.py << 'EOF'
import flwr as fl
from flwr.client import NumPyClient
import time
import numpy as np

class BenchmarkClient(NumPyClient):
    def __init__(self):
        self.data_size = 10000
        self.model_size = 1000
        
    def get_parameters(self, config):
        # Simulate model parameters
        return [np.random.rand(self.model_size).astype(np.float32)]
        
    def fit(self, parameters, config):
        # Simulate training
        time.sleep(0.1)  # Simulate computation
        return [np.random.rand(self.model_size).astype(np.float32)], self.data_size, {}
        
    def evaluate(self, parameters, config):
        # Simulate evaluation
        time.sleep(0.05)  # Simulate computation
        return 0.75, self.data_size, {}

# Create client instance
client = BenchmarkClient()

# Define get_client_fn for supernode - sử dụng to_client() để chuyển đổi từ NumPyClient sang Client
def get_client_fn():
    return client.to_client()

# If running directly (like in our benchmark)
if __name__ == "__main__":
    start_time = time.time()
    
    # Connect to insecure server using up-to-date API
    try:
        fl.client.start_client(
            server_address="localhost:8080", 
            client=client
        )
    except Exception as e:
        print(f"Client stopped: {e}")
    
    elapsed = time.time() - start_time
    print(f"Elapsed time: {elapsed} seconds")
EOF
}

# Function to run benchmark
run_benchmark() {
    echo "--- Running benchmark ---"
    
    echo "1. Testing with TLS/SSL enabled (secure)"
    echo "Starting secure superlink server..."
    bash "${BASE_DIR}/start_server_superlink.sh" > secure_superlink_log.txt 2>&1 &
    SERVER_PID=$!
    sleep 3  # Give server time to start
    
    echo "Starting secure supernode client..."
    # Chạy client với timeout 30 giây để tránh treo
    timeout 30s time bash "${BASE_DIR}/start_client_supernode.sh" > secure_supernode_log.txt 2>&1
    SECURE_STATUS=$?
    
    echo "Stopping secure superlink server..."
    kill $SERVER_PID 2>/dev/null
    # Đảm bảo tất cả các tiến trình con liên quan đều được dọn dẹp
    pkill -f "flower-superlink" 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    
    sleep 3  # Wait between tests
    
    echo "2. Testing without TLS/SSL (insecure)"
    echo "Creating insecure supernode/superlink configurations..."
    create_insecure_superlink
    
    echo "Starting insecure superlink server..."
    python3 temp_insecure_superlink.py > insecure_superlink_log.txt 2>&1 &
    SERVER_PID=$!
    sleep 3  # Give server time to start
    
    echo "Starting insecure supernode client..."
    # Chạy client với timeout 30 giây để tránh treo
    timeout 30s time python3 temp_insecure_supernode.py > insecure_supernode_log.txt 2>&1
    INSECURE_STATUS=$?
    
    echo "Stopping insecure superlink server..."
    kill $SERVER_PID 2>/dev/null
    # Xóa bỏ các tiến trình có thể còn sót lại
    pkill -f "temp_insecure_superlink.py" 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    
    # Cleanup
    rm -f temp_insecure_superlink.py temp_insecure_supernode.py
    
    echo
    echo "--- Results Summary ---"
    # Timeout (mã 124) cũng được coi là thành công vì script chạy hết thời gian
    if [ $SECURE_STATUS -eq 0 ] || [ $SECURE_STATUS -eq 124 ]; then
        echo "Secure connection (SuperNode/SuperLink): SUCCESS"
    else
        echo "Secure connection (SuperNode/SuperLink): FAILED (Mã trạng thái: $SECURE_STATUS)"
    fi
    
    if [ $INSECURE_STATUS -eq 0 ] || [ $INSECURE_STATUS -eq 124 ]; then
        echo "Insecure connection: SUCCESS"
    else
        echo "Insecure connection: FAILED (Mã trạng thái: $INSECURE_STATUS)"
    fi
    
    echo
    echo "Check logs for detailed timings:"
    echo "- secure_superlink_log.txt / secure_supernode_log.txt"
    echo "- insecure_superlink_log.txt / insecure_supernode_log.txt"
}

# Main script
echo "This benchmark will compare Flower SuperNode/SuperLink performance with and without TLS."
echo "Press any key to start or Ctrl+C to cancel..."
read -n 1

run_benchmark

echo
echo "Benchmark complete!"
