#!/usr/bin/env python3
"""
This file defines a Flower Executor for use with flower-superlink.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, cast
import logging
import time

import flwr as fl
import torch
import numpy as np
from flwr.common import Metrics, Parameters, Scalar, FitIns, EvaluateIns, FitRes, EvaluateRes
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy, FedAvg
from flwr.superexec.executor import Executor

# Configure logging with debug level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("flower-mnist-superlink-executor")

class MnistExecutor(Executor):
    """Executor class for MNIST federated learning."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the executor with config parameters."""
        super().__init__()
        self.config = config
        
        # Set parameters for the federated learning process
        self.num_rounds = int(config.get("num_rounds", 3))
        self.min_fit_clients = int(config.get("min_fit_clients", 1))
        self.min_evaluate_clients = int(config.get("min_evaluate_clients", 1))
        self.min_available_clients = int(config.get("min_available_clients", 1))
        self.current_round = 0
        
        # Initialize an empty model (will be populated on first round)
        self.parameters = None
        
        # Create a strategy
        self.strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=self.min_fit_clients,
            min_evaluate_clients=self.min_evaluate_clients,
            min_available_clients=self.min_available_clients,
        )
        
        logger.info(f"Initialized MnistExecutor with {self.num_rounds} rounds")
        logger.info(f"Using min_fit_clients={self.min_fit_clients}, min_evaluate_clients={self.min_evaluate_clients}")
        
    def initialize(self, linkstate_factory=None, ffs_factory=None) -> None:
        """Initialize the executor with required factories."""
        logger.info("Initializing MNIST executor")
        # Store the factories if needed
        self.linkstate_factory = linkstate_factory
        self.ffs_factory = ffs_factory
        
        # Set up a thread that will start training after a delay
        import threading
        import time
        
        def delayed_start():
            # Wait to give client time to connect
            wait_time = 20  # Wait 20 seconds before starting
            logger.info(f"Will auto-start training in {wait_time} seconds...")
            time.sleep(wait_time)
            
            try:
                logger.info("Auto-starting federated learning run")
                # Call start_run directly
                self.start_run()
                logger.info("Federated learning run has been started")
            except Exception as e:
                logger.error(f"Error auto-starting training: {e}", exc_info=True)
                
        # Start the delayed thread
        start_thread = threading.Thread(target=delayed_start, daemon=True)
        start_thread.start()
        logger.info("Auto-start thread initialized")
        
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set executor config."""
        self.config.update(config)
        logger.info(f"Updated config: {self.config}")
        
    def start_run(self) -> None:
        """Start a run."""
        logger.info("Starting MNIST federated learning run")
        # Instead of scheduling task, directly call fit_round
        logger.info("Starting first round directly")
        self.fit_round(1)

    def _create_initial_parameters(self) -> Parameters:
        """Create initial parameters for the model."""
        # Create a simple CNN model to get its parameters
        net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout2d(0.25),
            torch.nn.Flatten(),
            torch.nn.Linear(9216, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout2d(0.5),
            torch.nn.Linear(128, 10),
        )
        
        # Get the model parameters as a list of NumPy arrays
        weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
        return ndarrays_to_parameters(weights)

    def fit_round(self, server_round: int) -> bool:
        """Execute one round of federated learning."""
        logger.info(f"Starting fit_round {server_round}/{self.num_rounds}")
        self.current_round = server_round
        
        # Initialize the parameters if this is the first round
        if self.parameters is None:
            self.parameters = self._create_initial_parameters()
            logger.info("Initialized model parameters for first round")
        
        try:
            # Directly call fit clients
            logger.info(f"Directly calling _fit_clients for round {server_round}")
            self._fit_clients(server_round)
            
            # If we successfully completed this round and there are more rounds,
            # start the next round
            if server_round < self.num_rounds:
                logger.info(f"Scheduling next round: {server_round + 1}")
                
                # Use a separate thread to start the next round
                import threading
                def start_next_round():
                    time.sleep(3)  # Wait a bit before starting next round
                    try:
                        self.fit_round(server_round + 1)
                    except Exception as e:
                        logger.error(f"Error starting round {server_round + 1}: {e}", exc_info=True)
                
                threading.Thread(target=start_next_round, daemon=True).start()
        except Exception as e:
            logger.error(f"Error during fit round {server_round}: {e}", exc_info=True)
        
        # Return True to continue federated learning, False to stop
        return server_round < self.num_rounds
    
    def _fit_clients(self, server_round: int) -> None:
        """Fit clients and update model parameters."""
        try:
            # Try to determine available clients
            logger.debug("Trying to get available client nodes")
            try:
                # This method may not exist or work as expected
                clients = self.get_registered_nodes()
                client_ids = [str(i) for i in range(len(clients))]
                logger.info(f"Round {server_round}: {len(clients)} clients available via get_registered_nodes")
            except Exception as e:
                logger.warning(f"Couldn't get registered nodes: {e}")
                # Fallback: Just assume client ID 0 is available
                client_ids = ["0"]
                logger.info(f"Round {server_round}: Using fallback client_ids={client_ids}")
            
            # Configure client training
            config = {
                "epochs": 1,
                "batch_size": 32,
                "round": server_round,
                "num_rounds": self.num_rounds,
            }
            
            # Fit clients
            logger.info(f"Round {server_round}: Starting client training with {len(client_ids)} clients")
            
            # Create instruction
            try:
                instruction = fl.common.Instruction(
                    label="fit",
                    resumed_from=None,
                    created_at=0.0,
                    parameters=self.parameters,
                    config=config,
                    client_ids=client_ids,
                )
                
                # Execute the instruction
                logger.info(f"Round {server_round}: Executing instruction")
                results = self.execute_instruction(instruction)
                logger.info(f"Round {server_round}: Got {len(results.successes) if results and hasattr(results, 'successes') else 0} successful results")
                
                # Aggregate results
                if results and hasattr(results, 'successes') and results.successes:
                    updated_parameters, _ = self.strategy.aggregate_fit(
                        server_round=server_round,
                        results=[(cid, res.fit) for cid, res in results.successes.items()],
                        failures=[],
                    )
                    
                    if updated_parameters is not None:
                        self.parameters = updated_parameters
                        logger.info(f"Round {server_round}: Updated global model parameters")
                        
                        # Directly evaluate clients
                        self._evaluate_clients(server_round)
                    else:
                        logger.warning(f"Round {server_round}: aggregation returned None parameters")
                else:
                    logger.warning(f"Round {server_round}: No client results to aggregate")
            except Exception as inner_e:
                logger.error(f"Error during instruction execution in round {server_round}: {inner_e}", exc_info=True)
                
        except Exception as e:
            logger.error(f"Error during fit round {server_round}: {e}", exc_info=True)
    
    def _evaluate_clients(self, server_round: int) -> None:
        """Evaluate the model on client data."""
        try:
            # Try to determine available clients
            logger.debug("Trying to get available client nodes for evaluation")
            try:
                # This method may not exist or work as expected
                clients = self.get_registered_nodes()
                client_ids = [str(i) for i in range(len(clients))]
                logger.info(f"Round {server_round}: {len(clients)} clients available for evaluation")
            except Exception as e:
                logger.warning(f"Couldn't get registered nodes for evaluation: {e}")
                # Fallback: Just assume client ID 0 is available
                client_ids = ["0"]
                logger.info(f"Round {server_round}: Using fallback client_ids={client_ids} for evaluation")
            
            # Configure client evaluation
            config = {
                "round": server_round,
            }
            
            # Evaluate clients
            instruction = fl.common.Instruction(
                label="evaluate",
                resumed_from=None,
                created_at=0.0,
                parameters=self.parameters,
                config=config,
                client_ids=client_ids,
            )
            
            # Execute the instruction
            results = self.execute_instruction(instruction)
            logger.info(f"Round {server_round}: Completed evaluation with {len(results.successes)} successful results")
            
            # Aggregate evaluation results
            if results and results.successes:
                # Aggregate the evaluation results
                loss_aggregated = self.strategy.aggregate_evaluate(
                    server_round=server_round,
                    results=[(cid, res.evaluate) for cid, res in results.successes.items()],
                    failures=[],
                )
                
                if loss_aggregated is not None:
                    logger.info(f"Round {server_round}: Aggregated loss: {loss_aggregated:.4f}")
                    
                # Log individual client metrics
                for client_id, result in results.successes.items():
                    metrics = result.evaluate.metrics
                    if metrics:
                        logger.info(f"Client {client_id} metrics: {metrics}")
            else:
                logger.warning(f"Round {server_round}: No evaluation results to aggregate")
                
            # Check if this was the final round
            if server_round >= self.num_rounds:
                logger.info(f"Federated learning completed after {server_round} rounds!")
                
        except Exception as e:
            logger.error(f"Error during evaluation round {server_round}: {e}")
            
    def configure_fit(self) -> Optional[Tuple[Parameters, List]]:
        """Configure fit job."""
        # This is a simplified version - typically you'd return more configuration
        return None
        
    def aggregate_fit(self, metrics: List[Tuple[Metrics, int]]) -> Dict[str, Scalar]:
        """Aggregate fit metrics."""
        aggregated = {}
        
        # Simple averaging of metrics
        if metrics:
            total_samples = sum(num_samples for _, num_samples in metrics)
            
            if total_samples > 0:
                for m, num_samples in metrics:
                    weight = num_samples / total_samples
                    for key, value in m.items():
                        if key not in aggregated:
                            aggregated[key] = 0
                        aggregated[key] += value * weight
        
        return aggregated

    def evaluate(self) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Perform central evaluation."""
        # In this implementation, we don't perform central evaluation
        return None, {}

# Export the executor
executor = MnistExecutor(config={
    "min_fit_clients": 1,
    "min_evaluate_clients": 1,
    "min_available_clients": 1,
    "num_rounds": 3,
})

# For testing
if __name__ == "__main__":
    print(f"Executor initialized: {executor}")
    print("This file defines a Flower executor for use with flower-superlink")
