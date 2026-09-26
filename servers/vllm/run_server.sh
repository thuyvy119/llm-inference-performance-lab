#!/bin/bash

set -euo pipefail

MODEL="Qwen/Qwen3-0.6B"
HOST="127.0.0.1"
PORT="18000"
MAX_MODEL_LEN="4096"
GPU_MEMORY_UTILIZATION="0.90"
TENSOR_PARALLEL_SIZE="1"
ENABLE_PREFIX_CACHING="true"

SNAPSHOT_DIR="servers/vllm/snapshots"
SNAPSHOT_PATH="$SNAPSHOT_DIR/server_config_$(date +%Y%m%d_%H%M%S).json"
LATEST_PATH="servers/vllm/last_launch.json" 

echo "Starting vLLM server with the following configuration:"
echo "Model: $MODEL"
echo "Host: $HOST"
echo "Port: $PORT" 

mkdir -p "$(dirname "$SNAPSHOT_DIR")"
cat > "$SNAPSHOT_PATH" << EOF
{
    "model": "$MODEL",
    "host": "$HOST",
    "port": $PORT,
    "max_model_len": $MAX_MODEL_LEN,
    "gpu_memory_utilization": $GPU_MEMORY_UTILIZATION,
    "tensor_parallel_size": $TENSOR_PARALLEL_SIZE,
    "enable_prefix_caching": $ENABLE_PREFIX_CACHING,
    "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF

cp "$SNAPSHOT_PATH" "$LATEST_PATH"
echo "Server config snapshot written to: $SNAPSHOT_PATH"
echo "Latest snapshot symlinked at: $LATEST_PATH"

exec vllm serve "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --no-enable-prefix-caching \
    --default-chat-template-kwargs '{"enable_thinking": false}'

