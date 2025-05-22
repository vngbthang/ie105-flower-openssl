"""
This file demonstrates how to configure a Flower server strategy to use with flower-superlink.
To use this, first run:
    pip install "flwr>=1.5.0"
    
Then run the superlink with secure certificates:
    flower-superlink \
        --ssl-certfile=certs/server/server.pem \
        --ssl-keyfile=certs/server/server.key \
        --ssl-ca-certfile=certs/ca/ca.pem \
        --fleet-api-address=[::]:8443
"""

import flwr as fl

# Define your strategy here
class MyStrategy(fl.server.strategy.FedAvg):
    def __init__(self):
        super().__init__(min_fit_clients=2, min_evaluate_clients=2)

# Create the strategy
strategy = MyStrategy()
