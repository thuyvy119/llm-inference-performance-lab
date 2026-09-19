# Experiment 1: vLLM Single-GPU Baseline — Results

## 1. Run Summary

Experiment 1 was successfully executed on the target GPU environment.

### Environment

| Parameter              | Result                       |
| ---------------------- | ---------------------------- |
| GPU                    | NVIDIA GeForce RTX 3090      |
| GPU count              | 1                            |
| GPU memory             | 23.56 GB reported by PyTorch |
| PyTorch                | 2.13.0+cu130                 |
| CUDA                   | 13.0                         |
| vLLM                   | 0.29.0                       |
| Model                  | Qwen/Qwen3-0.6B              |
| Tensor parallelism     | 1                            |
| Max model length       | 4096                         |
| GPU memory utilization | 0.90                         |
| Thinking               | Disabled                     |

Environment validation passed all experiment assumptions.

---

# 2. Workload

The repeated baseline used:

| Parameter         | Value                                                       |
| ----------------- | ----------------------------------------------------------- |
| Requests          | 20                                                          |
| Concurrency       | 1                                                           |
| Prompt            | Explain the difference between GPU and CPU in simple terms. |
| Max output tokens | 128                                                         |
| Temperature       | 0.7                                                         |
| Top-p             | 0.8                                                         |
| Top-k             | 20                                                          |
| Min-p             | 0.0                                                         |
| Thinking          | Disabled                                                    |
| Streaming         | Enabled                                                     |

The workload therefore represents sequential single-request serving.
The complete 1c configuration is recorded in the raw result file.

---

# 3. Experiment 1b — Single Request

The single-request run produced:

| Metric            |          Result |
| ----------------- | --------------: |
| TTFT              |       108.13 ms |
| Decode duration   |       372.22 ms |
| E2E latency       |       480.35 ms |
| Input tokens      |              24 |
| Output tokens     |             128 |
| TPOT              |  2.931 ms/token |
| Output throughput | 343.88 tokens/s |

The request reached the configured maximum of 128 output tokens.
The single-request run also generated a GPU monitoring trace.

---

# 4. Experiment 1c — Repeated Sequential Baseline

Twenty requests were executed sequentially.

The complete request-level measurements are stored in:

```text
results/raw/exp1c_repeated_baseline.json
```

The experiment recorded 20 requests at concurrency 1.

## 4.1 TTFT

| Statistic |     TTFT |
| --------- | -------: |
| Mean      | 46.87 ms |
| Minimum   | 35.66 ms |
| P50       | 43.21 ms |
| P90       | 53.71 ms |
| P95       | 58.03 ms |
| P99       | 87.03 ms |
| Maximum   | 94.28 ms |

The first request had the largest TTFT at approximately 94.28 ms. Subsequent requests generally exhibited lower TTFT.

---

## 4.2 TPOT

| Statistic |           TPOT |
| --------- | -------------: |
| Mean      | 2.764 ms/token |
| Minimum   | 2.637 ms/token |
| P50       | 2.741 ms/token |
| P90       | 2.812 ms/token |
| P95       | 2.882 ms/token |
| P99       | 3.056 ms/token |
| Maximum   | 3.099 ms/token |

TPOT showed relatively little variation across the 20 sequential requests.

---

## 4.3 End-to-End Latency

| Statistic | E2E latency |
| --------- | ----------: |
| Mean      |   396.11 ms |
| Minimum   |   378.52 ms |
| P50       |   390.77 ms |
| P90       |   402.32 ms |
| P95       |   409.81 ms |
| P99       |   472.24 ms |
| Maximum   |   487.84 ms |

The first request had an E2E latency of approximately 487.84 ms, while most subsequent requests were approximately 380–402 ms.

---

# 5. Output Length

Most requests generated the maximum 128 output tokens.

The observed output lengths were:

* 17 requests with 128 tokens
* 2 requests with 125 tokens
* 1 request with 121 tokens

Therefore, `max_tokens=128` should be interpreted as a maximum generation limit rather than a fixed output length.

## The raw request-level measurements record the exact output count for each request. For example, requests 1–3 generated 128 tokens, while request 4 generated 125 tokens.

---

# 6. GPU Monitoring

The Experiment 1c GPU trace contains 92 samples collected at an approximately 50-ms sampling interval.

Observed ranges across the complete trace:

| Metric          |  Minimum |  Maximum |     Mean |
| --------------- | -------: | -------: | -------: |
| GPU utilization |       0% |      96% |   79.01% |
| GPU memory used | 22534 MB | 22534 MB | 22534 MB |
| GPU temperature |     42°C |     62°C |  58.08°C |
| GPU power       |  28.54 W | 296.02 W | 250.95 W |

The GPU trace shows periods of low utilization followed by sustained high utilization during the repeated inference workload.

The all-sample mean GPU utilization of 79.01% should not be interpreted as the average utilization of decode alone. The monitoring interval includes periods before, between, and after request activity, and the sampling is not synchronized with inference phases.

Similarly, the GPU memory measurement remained approximately constant at 22.5 GB during the trace, reflecting the serving process's GPU memory footprint rather than per-request memory consumption.

---

# 7. Raw Artifacts

The raw Experiment 1 artifacts are stored under:

results/raw/
├── exp1a_environment.json
├── exp1b_single_request.json
├── exp1b_gpu_metrics.csv
├── exp1c_repeated_baseline.json
└── exp1c_gpu_metrics.csv

These raw files contain the request-level measurements, environment information, and GPU monitoring samples used to produce this summary.

Interpretation Scope

The results establish a baseline for:

single-GPU inference
Qwen3-0.6B
standard generation without thinking
sequential concurrency-1 serving
client-observed latency
coarse GPU behavior

The results do not establish performance under concurrent load, high-throughput serving, multi-GPU execution, or kernel-level optimization.
Those conditions are intentionally left for subsequent experiments.

---

# 8. Observations

### Observation 1 — The single request is not representative of the repeated baseline

The 1b request produced:

```text
TTFT = 108.13 ms
E2E = 480.35 ms
```

while the 1c median values were:

```text
TTFT = 43.21 ms
E2E = 390.77 ms
```

This demonstrates why the repeated sequential baseline is useful for characterizing normal request-to-request behavior.
The difference should not by itself be interpreted as a general performance characteristic without additional controlled investigation.

---

### Observation 2 — Decode behavior is relatively stable

TPOT remained within approximately:

```text
2.64–3.10 ms/token
```

across the 20 requests.
The P95 TPOT was 2.882 ms/token, while the P99 was 3.056 ms/token.
This provides a useful reference point for later experiments that change concurrency or output length.

---

### Observation 3 — GPU utilization reaches a high level during active generation

The GPU trace shows the GPU transitioning from low utilization to sustained utilization above 90% during substantial portions of the workload. This provides basic evidence that the GPU is actively executing the inference workload.
However, `nvidia-smi` utilization alone cannot identify which kernels are responsible for execution or whether the workload is limited by compute, memory movement, synchronization, scheduling, or another factor.
Kernel-level investigation is therefore deferred to the Nsight Systems and Nsight Compute phases.

---

### Observation 4 — GPU memory allocation remains stable

GPU memory remained approximately:

```text
22534 MB / 24576 MB
```

throughout the sampled interval.
This is consistent with the model/server maintaining a large persistent GPU allocation during serving.
The measurement alone does not establish how the allocated memory is divided between model weights, KV cache, CUDA graphs, workspace, or other allocations.

---

### Observation 5 — Reasoning is explicitly disabled

The baseline uses:

```text
enable_thinking = false
```

This makes the Experiment 01 workload an ordinary answer-generation baseline.
Reasoning-enabled Qwen3 inference should be evaluated separately so that changes in generation behavior and output characteristics can be attributed to the reasoning configuration rather than being mixed into the initial baseline.

---

# 9. Limitations

Experiment 1 has several important limitations.

### Fixed prompt

Only one prompt is used.

Therefore, the results do not characterize the effect of different prompt lengths, semantic complexity, or input distributions.

### Fixed maximum output length

The workload uses `max_tokens=128`.

Actual output length varies slightly between requests. A separate experiment will vary output length explicitly.

### Sequential workload

Concurrency is fixed at 1. Therefore, the experiment does not characterize batching, request scheduling under contention, or maximum serving throughput.

### Coarse GPU monitoring

GPU metrics are collected using `nvidia-smi` at approximately 50 ms intervals.
This provides useful coarse-grained runtime information but cannot expose individual CUDA kernels, memory transfers, synchronization operations, or kernel-level bottlenecks.

### Client-observed measurements

Latency measurements represent the behavior observed by the benchmark client. They therefore include relevant client/network/protocol overhead and are not equivalent to an internal vLLM kernel timeline.

---

# 10. Baseline Reference

Experiment 1 establishes the following reference values for subsequent experiments:

```text
Model:              Qwen/Qwen3-0.6B
GPU:                RTX 3090
Concurrency:        1
Requests:           20
Thinking:           disabled

TTFT P50:           43.21 ms
TTFT P95:           58.03 ms

TPOT P50:           2.741 ms/token
TPOT P95:           2.882 ms/token

E2E P50:            390.77 ms
E2E P95:            409.81 ms
```

These values should be treated as the Experiment 1 baseline rather than as general performance claims about vLLM or the Qwen3-0.6B model.

```
```





