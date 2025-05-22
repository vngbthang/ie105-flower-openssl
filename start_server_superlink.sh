#!/bin/bash

# Start Flower SuperLink server with mTLS security
# Get the absolute path to the project directory
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

flower-superlink \
    --ssl-certfile="${BASE_DIR}/certs/server/server.pem" \
    --ssl-keyfile="${BASE_DIR}/certs/server/server.key" \
    --ssl-ca-certfile="${BASE_DIR}/certs/ca/ca.pem" \
    --fleet-api-address=[::]:8443
