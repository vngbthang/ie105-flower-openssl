#!/usr/bin/env python3
"""
Script to trigger start_run on the Flower server using direct import.
"""

import sys
import os
import time
import logging
import requests
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("flower-mnist-trigger")

# Get the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add current directory to path
sys.path.insert(0, BASE_DIR)

# Import the executor module
logger.info("Importing executor module...")
try:
    from server.executor import executor
    logger.info("Successfully imported executor")
except Exception as e:
    logger.error(f"Error importing executor: {e}")
    sys.exit(1)

# Try to trigger start_run
logger.info("Attempting to trigger start_run...")
try:
    # Call the start_run method
    executor.start_run()
    logger.info("Successfully called start_run")
except AttributeError as e:
    logger.error(f"AttributeError: {str(e)}")
    # Try alternative approaches if method not found
    # 1. Try using __getattribute__
    try:
        method = getattr(executor, "start_run")
        method()
        logger.info("Successfully called start_run using getattr")
    except Exception as e2:
        logger.error(f"Failed using getattr: {str(e2)}")
        
        # 2. Try REST API approach
        logger.info("Trying REST API approach...")
        try:
            response = requests.post("http://localhost:9093/api/v1/run", timeout=5)
            logger.info(f"API response: {response.status_code}")
        except Exception as e3:
            logger.error(f"REST API approach failed: {str(e3)}")
except Exception as e:
    logger.error(f"Error triggering start_run: {e}")

logger.info("Script completed")
