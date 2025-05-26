#!/usr/bin/env python3
"""
Script to trigger training on a running Flower server.
This works with Flower v1.18.0 SuperLink server.
"""

import sys
import time
import argparse
import requests
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("flower-mnist-trigger")

def main():
    """Main function to trigger training."""
    parser = argparse.ArgumentParser(description='Trigger training on a Flower SuperLink server')
    parser.add_argument('--host', type=str, default='localhost', help='Host running the Flower server')
    parser.add_argument('--port', type=int, default=9093, help='API port of the Flower server')
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    
    # Try different potential endpoints
    endpoints = [
        "/api/v1/start-run",
        "/api/v1/start_run",
        "/start-run",
        "/start_run",
        "/run",
        "/"
    ]
    
    logger.info(f"Attempting to trigger training on {base_url}")
    
    for endpoint in endpoints:
        url = f"{base_url}{endpoint}"
        try:
            logger.info(f"Trying endpoint: {url}")
            response = requests.post(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"Success! Training triggered using {endpoint}")
                logger.info(f"Response: {response.text}")
                return 0
            else:
                logger.warning(f"Failed with status {response.status_code}")
        except Exception as e:
            logger.warning(f"Error: {str(e)}")
    
    logger.error("Failed to trigger training on any endpoint")
    return 1

if __name__ == "__main__":
    sys.exit(main())
