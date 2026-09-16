#!/bin/bash

set -euo pipefail

MODEL="Qwen/Qwen3-0.6B"
HOST="127.0.0.1"
PORT="8000"
MAX_MODEL_LEN="4096"
GPU_MEMORY_UTILIZATION="0.90"

echo "Starting vLLM server with the following configuration:"
echo "Model: $MODEL"
echo "Host: $HOST"
echo "Port: $PORT" 

exec vllm serve "$MODEL" \
    --host $HOST \
    --port $PORT \
    --dtype auto \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION