# Experiment 01 - vLLM Single-GPU Baseline

## Objective

Measure the baseline latency and basic GPU behavior of a vLLM inference server under a sequential single-request workload.

The experiment establishes the reference workload that will be used before introducing additional variables such as concurrency, input length, output length, serving framework, or reasoning mode.

## Experimental Question

What latency and basic GPU behavior does vLLM exhibit when serving Qwen/Qwen3-0.6B on a single NVIDIA RTX 3090 under concurrency 1?

## Hardware

* GPU: NVIDIA GeForce RTX 3090
* GPU count: 1
* VRAM: approximately 24 GB

The detailed software and hardware environment is recorded during Experiment 1a.

## Model

* Model: Qwen/Qwen3-0.6B
* Maximum model length: 4096
* GPU memory utilization target: 0.90
* Tensor parallelism: 1

## Server

* Framework: vLLM
* Host: 127.0.0.1
* Port: 18000

The server is configured for single-GPU inference with tensor parallelism set to 1.

## Qwen3 Reasoning Configuration

Qwen3 supports both thinking/reasoning generation and standard generation.

For Experiment 01, **thinking is explicitly disabled**:

```python
chat_template_kwargs = {
    "enable_thinking": False
}
```

The same setting is applied at the vLLM server level through the default chat-template configuration.

This configuration is intentional. Experiment 01 is designed to establish a baseline for ordinary prompt-to-answer serving behavior without introducing reasoning as an additional workload variable.

Reasoning-enabled Qwen3 inference is reserved for a later experiment.

Therefore, reasoning mode is controlled as follows:

| Configuration      | Experiment 01       |
| ------------------ | ------------------- |
| Qwen3 model        | Qwen/Qwen3-0.6B     |
| Thinking/reasoning | Disabled            |
| `enable_thinking`  | `false`             |
| Generation mode    | Standard generation |

## Workload

The same prompt is used for the baseline requests:

> Explain the difference between GPU and CPU in simple terms.

### Request parameters

* `max_tokens`: 128
* `temperature`: 0.7
* `top_p`: 0.8
* `top_k`: 20
* `min_p`: 0.0
* `enable_thinking`: false
* streaming: enabled

The request configuration is kept fixed across Experiments 1b and 1c.

Streaming is enabled so that the client can measure the time at which the first generated output is received.

When streaming, usage information is requested from the server so that input and output token counts are obtained from the API rather than inferred from the number of streaming chunks.

## Experiment 1a - Environment Validation

Experiment 1a validates the execution environment before collecting inference measurements.

The environment validation records:

* Python version
* Python executable
* PyTorch version
* CUDA availability
* PyTorch CUDA version
* GPU count
* GPU model
* GPU memory
* vLLM version

The environment validation does not measure inference performance.

The recorded environment for the completed experiment includes:

* Python: 3.12.14
* PyTorch: 2.13.0+cu130
* PyTorch CUDA: 13.0
* GPU: NVIDIA GeForce RTX 3090
* GPU count: 1
* GPU memory: approximately 24 GB
* vLLM: 0.29.0

## Experiment 1b — Single Request

One request is executed with:

* concurrency = 1
* number of requests = 1

The purpose is to validate the complete request and measurement pipeline before collecting repeated measurements.

The experiment records:

* request start time
* first output time
* request completion time
* TTFT
* decode duration
* E2E latency
* input token count
* output token count
* TPOT
* output token throughput
* response text

GPU metrics are sampled concurrently during the request.

The results are saved as:

```text
results/raw/exp1b_single_request.json
results/raw/exp1b_gpu_metrics.csv
```

## Experiment 1c — Repeated Sequential Baseline

Twenty requests are executed sequentially.

The next request starts only after the previous request has completed.

* number of requests = 20
* concurrency = 1

The purpose is to measure request-to-request variability without introducing concurrent scheduling effects.

Each request uses the same prompt and generation configuration as Experiment 1b.

The experiment reports distribution statistics for:

* TTFT
* TPOT
* E2E latency

The distribution statistics include:

* mean
* minimum
* maximum
* p50
* p90
* p95
* p99

The request-level results and GPU monitoring trace are saved as:

```text
results/raw/exp1c_repeated_baseline.json
results/raw/exp1c_gpu_metrics.csv
```

## Metrics

### Time to First Token (TTFT)

TTFT = first_output_time - request_start
Client-observed elapsed time from request initiation until the first generated output content is received.
TTFT captures the latency experienced before the first streamed generated content reaches the client.

### Decode Duration

decode_duration = request_end - first_output_time
Elapsed time between the first generated output content and completion of the streamed response.

### End-to-End Latency

E2E = request_end - request_start
Elapsed time from request initiation until the complete response is received.

### Time Per Output Token (TPOT)

For responses containing at least two generated tokens:

```text
TPOT = decode_duration / (output_tokens - 1)
```

TPOT represents the average time between generated output tokens after the first output token.

### Output Token Throughput

Output token throughput is calculated as:

```text
output_tokens / decode_duration
```

It represents the observed generation rate during the measured decode interval.

### Input Token Count

The number of input tokens reported by the inference API for the completed request.

### Output Token Count

The number of generated output tokens reported by the inference API.

Streaming chunk count is not used as a substitute for output token count because a streaming response chunk does not necessarily correspond to exactly one generated token.

## GPU Metrics

GPU metrics are sampled approximately every 50 ms using `nvidia-smi`.

The following measurements are collected:

* GPU utilization
* GPU memory used
* GPU memory total
* GPU temperature
* GPU power

GPU monitoring is performed independently of request-level timing.
The sampling timestamps therefore provide a coarse time series of GPU behavior rather than phase-synchronized measurements of prefill and decode.
In particular, a GPU utilization sample should not be interpreted as the exact utilization of a specific request or decode step.

## Measurement Scope

Experiment 1 is intentionally limited to a sequential concurrency-1 workload.
It is designed to establish a reference point before changing additional workload variables.

The experiment does not evaluate:
* concurrent request scheduling
* throughput under load
* batching efficiency
* different input lengths
* different output lengths
* multi-GPU inference
* vLLM versus SGLang
* kernel-level performance
* Nsight Systems traces
* Nsight Compute kernel analysis
* reasoning-enabled Qwen3 generation

These conditions are reserved for later experiments.
