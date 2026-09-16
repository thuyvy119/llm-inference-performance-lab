# Experiment 01 — vLLM Single-GPU Baseline

## Objective

Measure the baseline latency and basic GPU behavior of a vLLM inference server under a sequential single-request workload.

## Experimental Question

What latency and basic GPU behavior does vLLM exhibit when serving Qwen/Qwen3-0.6B on a single NVIDIA RTX 3090 under concurrency 1?

## Hardware

- GPU: NVIDIA GeForce RTX 3090
- GPU count: 1
- VRAM: 24 GB

The exact driver, CUDA, Python, PyTorch, and vLLM versions are recorded during Experiment 1a.

## Model

- Model: Qwen/Qwen3-0.6B
- Maximum model length: 4096
- GPU memory utilization: 0.90
- Tensor parallelism: 1

## Server

- Framework: vLLM
- Host: 127.0.0.1
- Port: 18000

## Workload

Prompt:

> Explain the difference between GPU and CPU in simple terms.

Request parameters:

- max_tokens: 128
- temperature: 0.0
- streaming: enabled

## Experiment 1a — Environment Validation

The environment validation records:

- Python version
- Python executable
- PyTorch version
- CUDA availability
- PyTorch CUDA version
- GPU count
- GPU model
- GPU memory
- vLLM version

The environment validation does not measure inference performance.

## Experiment 1b — Single Request

One request is executed with:

- concurrency = 1
- number of requests = 1

The purpose is to validate the complete request and measurement
pipeline before collecting repeated measurements.

## Experiment 1c — Repeated Sequential Baseline

Twenty requests are executed sequentially.

The next request starts only after the previous request has completed.

- number of requests = 20
- concurrency = 1

This experiment measures request-to-request variability without introducing concurrent scheduling effects.

## Metrics

### Time to First Token (TTFT)

Client-observed elapsed time from request initiation until the first generated output content is received.

### Decode Duration

Elapsed time between the first generated output content and the completion of the streamed response.

### End-to-End Latency

Elapsed time from request initiation until the complete response is received.

### Time Per Output Token (TPOT)

For responses containing at least two generated tokens:

`decode_duration / (output_tokens - 1)`

### Output Throughput

`output_tokens / decode_duration`

### GPU Metrics

GPU metrics are sampled approximately every 50 ms:

- GPU utilization
- GPU memory used
- GPU memory total
- GPU temperature
- GPU power