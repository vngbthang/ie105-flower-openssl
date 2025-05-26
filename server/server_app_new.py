#!/usr/bin/env python3
"""
Server app for Flower SuperLink.

When using flower-superlink, we need to define and register a server app
rather than passing a strategy directly to the command line.
"""

import os
import logging
import flwr as fl
from typing import Dict, List, Optional, Tuple, Union
from flwr.common import Metrics, Parameters, Scalar
from flwr.common.context import Context
from flwr.server import ServerConfig
from flwr.server.serverapp_components import ServerAppComponents

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("flower-mnist-superlink-server")

# Define an enhanced federated learning strategy
class MnistStrategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        num_rounds: int = 3,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
        )
        self.num_rounds = num_rounds
        self.current_round = 0
        logger.info(f"Initialized MnistStrategy with {num_rounds} rounds")
        
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitIns]]: # Ensure ClientProxy is correctly namespaced
        """Configure the next round of training."""
        self.current_round = server_round
        logger.info(f"Starting round {server_round}/{self.num_rounds} of training")
        
        # Configure training parameters for this round
        config = {
            "epochs": 1,  # Local epochs per round
            "batch_size": 32,
            "round": server_round,
            "num_rounds": self.num_rounds,
            "model": "mnist_cnn",
        }
        
        # Let the parent class handle the client selection and configuration
        fit_ins = super().configure_fit(server_round, parameters, client_manager)
        
        # Add our custom config to each client's configuration
        if fit_ins:
            for _, fit_ins_item in fit_ins:
                if hasattr(fit_ins_item, 'config'):
                    fit_ins_item.config.update(config)
            
        return fit_ins
    
    def evaluate(self, server_round: int, parameters: Parameters):
        """Evaluate model on test data after training."""
        logger.info(f"Round {server_round}/{self.num_rounds} completed. Running centralized evaluation...")
        
        # Here you could load a test dataset and evaluate the global model
        # For now, we'll just use the results from client evaluations
        result = super().evaluate(server_round, parameters)
        
        # Check if we've reached the final round
        if server_round >= self.num_rounds:
            logger.info(f"Federated learning completed after {server_round} rounds!")
            
        return result

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]], # Ensure ClientProxy is correctly namespaced
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]], # Ensure ClientProxy is correctly namespaced
    ) -> Optional[float]:
        """Aggregate evaluation results from clients."""
        if not results:
            return None
        
        # Aggregate and log metrics from client evaluations
        loss_aggregated = super().aggregate_evaluate(server_round, results, failures)
        
        # Log metrics for each client
        for i, (client, res) in enumerate(results):
            logger.info(f"Client {client.cid} round {server_round} results: loss={res.loss:.4f}, metrics={res.metrics}")
        
        # Log aggregated metrics
        if loss_aggregated is not None:
            logger.info(f"Round {server_round} aggregated loss: {loss_aggregated:.4f}")
        
        return loss_aggregated
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]], # Ensure ClientProxy is correctly namespaced
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]], # Ensure ClientProxy is correctly namespaced
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results from clients."""
        # Log the number of clients that successfully completed training
        logger.info(f"Round {server_round}: {len(results)} clients successfully completed training")
        logger.info(f"Round {server_round}: {len(failures)} clients failed during training")
        
        # For each successful client, log their metrics
        for i, (client, fit_res) in enumerate(results):
            logger.info(f"Client {client.cid} trained on {fit_res.num_examples} examples")
            if fit_res.metrics:
                for metric_name, metric_value in fit_res.metrics.items():
                    logger.info(f"Client {client.cid} {metric_name}: {metric_value}")
        
        # Use the parent class to aggregate the parameters
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        
        return parameters_aggregated, metrics_aggregated

# Define a server_fn following the pattern required by Flower SuperLink
def server_fn(context: Context) -> ServerAppComponents:
    """Create a ServerAppComponents object that returns strategy and server config."""
    # Create the strategy
    strategy = MnistStrategy(
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        num_rounds=3,
    )
    
    # Set server configuration, using num_rounds from the strategy
    server_config = ServerConfig(num_rounds=strategy.num_rounds)
    
    # Return a ServerAppComponents object with strategy and config
    return ServerAppComponents(strategy=strategy, config=server_config)

# Create and expose the server app object for flower-superlink
app = fl.server.ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    logger.info("Starting Flower ServerApp with MnistStrategy for SuperLink.")
    
    # Create strategy directly
    strategy = MnistStrategy(
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        num_rounds=3,
    )
    
    # Set server configuration
    server_config = fl.server.ServerConfig(num_rounds=3)
    
    # Import and run the server
    from flwr.server import start_server
    
    # Start server with strategy
    start_server(
        server_address="0.0.0.0:9091",
        config=server_config,
        strategy=strategy,
    )
    logger.info("Flower ServerApp has finished.")
