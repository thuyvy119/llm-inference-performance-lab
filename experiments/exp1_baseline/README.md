# Experiment 1 - vLLM Single-GPU Baseline

This experiment establishes a controlled baseline for LLM inference performance using vLLM on a single NVIDIA RTX 3090.
Experiment 1 uses **Qwen3-0.6B with thinking/reasoning explicitly disabled** so that the baseline measures ordinary prompt-to-answer serving behavior without introducing reasoning as an additional workload variable.

Reasoning-enabled Qwen3 inference will be treated as a separate workload condition in a later experiment.

## Experiments

### 1a - Environment Validation

Validates the software and hardware environment.
No inference performance conclusions are drawn from this experiment.

### 1b - Single Request

Runs one request to validate the complete request, latency, token-counting, and GPU-monitoring pipeline.
The resulting request-level measurements and GPU metrics are saved as experiment artifacts.

### 1c - Repeated Sequential Baseline

Runs 20 sequential requests at concurrency 1 and reports latency distributions.
This establishes the repeated-request baseline before introducing concurrency or other workload variables.

## Metrics

### Request-level metrics

* TTFT
* Decode duration
* E2E latency
* TPOT
* Output token throughput
* Input token count
* Output token count

### GPU metrics

* GPU utilization
* GPU memory used
* GPU memory total
* GPU temperature
* GPU power

## Qwen3 Configuration

Qwen3 supports both standard generation and thinking/reasoning generation.

For Experiment 01, thinking is explicitly disabled:

```python
chat_template_kwargs = {
    "enable_thinking": False
}
```

The same configuration is applied at the vLLM server level through the default chat-template configuration.
This is intentional. The purpose of Experiment 01 is to establish a baseline for standard prompt-to-answer inference.
Reasoning-enabled inference is reserved for a later experiment so that its effects on latency, output length, throughput, and GPU behavior can be measured as an independent workload variable.

Therefore, the Experiment 01 baseline should be interpreted as:

> **Qwen3-0.6B, standard generation, single RTX 3090, sequential concurrency-1 serving.**

## Execution

The experiments are designed to run on a Linux NVIDIA GPU instance.
The local development environment is used for code validation and testing. Actual inference measurements are collected on the target GPU environment.

### Target environment

* GPU: NVIDIA GeForce RTX 3090
* GPU memory: approximately 24 GB
* PyTorch: 2.13.0+cu130
* PyTorch CUDA version: 13.0
* vLLM: 0.29.0
* Model: Qwen/Qwen3-0.6B
* Tensor parallelism: 1
* Maximum model length: 4096
* GPU memory utilization target: 0.90

The detailed environment record is produced by Experiment 1a.
