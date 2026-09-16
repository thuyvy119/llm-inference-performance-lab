# Experiment 01 - vLLM Single-GPU Baseline

This experiment establishes a controlled baseline for LLM inference performance using vLLM on a single NVIDIA RTX 3090. Experiment 1 uses a non-reasoning/standard-answer configuration so that the baseline measures ordinary prompt-to-answer serving behavior. Reasoning-enabled inference will be treated as a separate workload variable later.

## Experiments

### 1a - Environment Validation

Validates the software and hardware environment.

### 1b - Single Request

Runs one request to validate the complete measurement pipeline.

### 1c - Repeated Sequential Baseline

Runs 20 sequential requests at concurrency 1 and reports latency distributions.

## Metrics

- TTFT
- Decode duration
- E2E latency
- TPOT
- Output token throughput
- GPU utilization
- GPU memory
- GPU temperature
- GPU power

## Execution

The experiments are designed to run on a Linux NVIDIA GPU instance.

The local development environment is used for code validation and testing. Actual inference measurements are collected on the target GPU environment.