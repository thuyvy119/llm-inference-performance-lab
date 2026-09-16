#!/bin/bash

set -euo pipefail

echo "Stopping vLLM server..."

if pgrep -f "vllm serve" > /dev/null; then
    pkill -f "vllm serve"
    echo "vLLM server stopped."
else
    echo "No vLLM server process found."
fi