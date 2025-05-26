#!/usr/bin/env python3
"""
This file defines a Flower Executor for use with flower-superlink.
For Flower v1.18.0, we use the default executor from flwr.superexec.deployment
"""

import flwr as fl
import logging

# Import the default executor
from flwr.superexec.deployment import executor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("flower-mnist-executor")

# Log that we're using the default executor
logger.info("Using default Flower executor with config from command line")

if __name__ == "__main__":
    print("This file exports the default Flower executor for use with flower-superlink")
    print("The executor is configured using the --executor-config parameter")
