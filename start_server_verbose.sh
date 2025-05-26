#!/bin/bash
# Script to run server with verbose output

# Set up logging level
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd):$PYTHONPATH"
export FLOWER_LOGGER=DEBUG

# Start server with verbose output
cd "$(dirname "${BASH_SOURCE[0]}")" && 
    bash start_server_superlink.sh 2>&1 | tee server_output.log
