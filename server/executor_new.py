#!/usr/bin/env python3
"""
This file defines a Flower Executor for MNIST federated learning.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, cast
import logging
import time

import flwr as fl
import numpy as np
import torch
from flwr.common import Metrics, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import Strategy, FedAvg
from flwr.superexec.executor import Executor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("flower-mnist-executor")

class MnistExecutor(Executor):
    """Executor for MNIST federated learning."""

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
        
        # Schedule a task to fit clients asynchronously
        self.schedule_task(
            task_id=f"fit_round_{server_round}",
            task_fn=self._fit_clients,
            task_args=(server_round,),
        )
        
        # Sleep a bit to allow task to be processed
        time.sleep(1)
        
        # Return True to continue federated learning, False to stop
        return server_round < self.num_rounds
    
    def _fit_clients(self, server_round: int) -> None:
        """Fit clients and update model parameters."""
        try:
            # Get available clients
            clients = self.get_registered_nodes()
            client_ids = [str(i) for i in range(len(clients))]
            logger.info(f"Round {server_round}: {len(clients)} clients available")
            
            if not clients:
                logger.warning("No clients available, skipping round")
                return
                
            # Configure client training
            config = {
                "epochs": 1,
                "batch_size": 32,
                "round": server_round,
                "num_rounds": self.num_rounds,
            }
            
            # Fit clients
            logger.info(f"Round {server_round}: Starting client training")
            instruction = fl.common.Instruction(
                label="fit",
                resumed_from=None,
                created_at=0.0,
                parameters=self.parameters,
                config=config,
                client_ids=client_ids,
            )
            
            # Execute the instruction
            results = self.execute_instruction(instruction)
            logger.info(f"Round {server_round}: Got {len(results.successes)} successful results")
            
            # Aggregate results
            if results and results.successes:
                updated_parameters, _ = self.strategy.aggregate_fit(
                    server_round=server_round,
                    results=[(cid, res.fit) for cid, res in results.successes.items()],
                    failures=[],
                )
                
                if updated_parameters is not None:
                    self.parameters = updated_parameters
                    logger.info(f"Round {server_round}: Updated global model parameters")
                    
                    # Schedule evaluation for this round
                    self.schedule_task(
                        task_id=f"evaluate_round_{server_round}",
                        task_fn=self._evaluate_clients,
                        task_args=(server_round,),
                    )
            else:
                logger.warning(f"Round {server_round}: No client results to aggregate")
                
        except Exception as e:
            logger.error(f"Error during fit round {server_round}: {e}")
    
    def _evaluate_clients(self, server_round: int) -> None:
        """Evaluate the model on client data."""
        try:
            # Get available clients
            clients = self.get_registered_nodes()
            client_ids = [str(i) for i in range(len(clients))]
            logger.info(f"Round {server_round}: Starting evaluation with {len(clients)} clients")
            
            if not clients:
                logger.warning("No clients available for evaluation")
                return
                
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

# Create the executor instance
executor = MnistExecutor(config={
    "min_fit_clients": 1,
    "min_evaluate_clients": 1,
    "min_available_clients": 1,
    "num_rounds": 3,
})

if __name__ == "__main__":
    print(f"MnistExecutor initialized with {executor.num_rounds} rounds")
