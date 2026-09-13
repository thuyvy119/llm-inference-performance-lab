#!/bin/bash

set -e

MODEL="Qwen/Qwen3-0.6B"

docker run \
    --rm \
    --gpus all \
    --ipc=host \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest \
    "$MODEL" \
    --dtype auto \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90