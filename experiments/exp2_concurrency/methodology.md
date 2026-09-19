# Experiment 2 — Methodology

## 1. Objective

The objective of Experiment 2 is to characterize how request concurrency changes inference behavior on a single GPU.
Experiment 1 measured the baseline behavior of Qwen/Qwen3-0.6B under sequential execution. Experiment 02 increases the number of simultaneously active requests while holding the workload and serving environment constant.
The experiment therefore treats concurrency as the primary independent variable.

## 2. Research Question

> How does increasing request concurrency affect per-request latency, system throughput, and GPU utilization when serving Qwen/Qwen3-0.6B with vLLM on a single NVIDIA RTX 3090?

## 3. Hardware and Serving Configuration

The experiment uses:

* NVIDIA GeForce RTX 3090
* one GPU
* tensor parallelism = 1
* Qwen/Qwen3-0.6B
* vLLM 0.29.0
* maximum model length = 4096
* GPU memory utilization target = 0.90

The vLLM server remains running for the entire experiment.
The server is not restarted between concurrency conditions.

## 4. Request Configuration

Every request uses the same workload:

```text
Prompt:
Explain the difference between GPU and CPU in simple terms.

max_tokens = 128
temperature = 0.7
top_p = 0.8
top_k = 20
min_p = 0.0
stream = true
enable_thinking = false
```

The Qwen3 reasoning mode remains disabled so that Experiment 02 measures standard generation behavior consistently with Experiment 01.

## 5. Independent Variable

The independent variable is request concurrency.

The tested levels are:

```text
C = {1, 2, 4, 8, 16, 32}
```

## 6. Definition of Concurrency

Concurrency `N` is defined as the maximum number of simultaneously in-flight requests.
The benchmark uses a rolling scheduler.

At concurrency 4, for example, the scheduler may operate as:

```text
R1 ────────────────
R2 ───────────
R3 ───────────────
R4 ──────────
          R5 ───────────────
           R6 ──────────
             ...
```

When one request completes, another request becomes eligible to start.

The scheduler therefore attempts to maintain:

```text
number of active requests <= target concurrency
```

until all requested benchmark work has completed.
This is intentionally different from launching fixed batches of N requests and waiting for every request in the batch to finish.

## 7. Warm-Up Procedure

Each concurrency condition begins with exactly 8 warm-up requests.

The warm-up requests use:

* the same model,
* the same prompt,
* the same generation parameters, and
* the same target concurrency.

Warm-up results are discarded. No warm-up latency or throughput measurement is included in the final experiment.

### Warm-Up Limitation

The number of warm-up requests is fixed at 8.
Therefore, for concurrency levels greater than 8, the warm-up phase cannot fill every possible concurrency slot simultaneously.

For example:

```text
C = 32
warm-up requests = 8
maximum warm-up active requests = 8
```

This is acceptable because the purpose of the warm-up is to stabilize the serving path before measurement, while the official 64-request workload is responsible for measuring the target concurrency condition.

## 8. Measurement Procedure

After the 8 warm-up requests complete:

1. Discard all warm-up results.
2. Clear/reset the benchmark's request-level measurement state.
3. Start the GPU sampler.
4. Start the benchmark timer.
5. Schedule 64 measured requests using the target concurrency.
6. Continue scheduling replacement requests whenever an active request completes.
7. Wait until all 64 measured requests finish.
8. Stop the benchmark timer.
9. Stop the GPU sampler.
10. Save the measured results.

The warm-up period is therefore outside the official measurement window.

## 9. Benchmark Duration

For each concurrency condition:

```text
benchmark_duration =
measurement_end - measurement_start
```

where:

```text
measurement_start =
immediately before the first measured request is scheduled

measurement_end =
when the final measured request completes
```

Warm-up duration is never included.

For example, if:

```text
warm-up = 2.0 seconds
measured workload = 8.0 seconds
```

then:

```text
benchmark duration = 8.0 seconds
```

not 10.0 seconds.

## 10. Request-Level Timing

Each measured request independently records:

```text
request_start
first_token
request_end
```

From these timestamps:

```text
TTFT = first_token - request_start

E2E = request_end - request_start

decode_duration = request_end - first_token
```

For requests with more than one output token:

```text
TPOT =
decode_duration / (output_tokens - 1)
```

The benchmark obtains input and output token counts from the server's API usage information rather than counting streaming chunks.

## 11. System Throughput

Request throughput is calculated over the entire measured workload:

```text
request_throughput =
successful_requests / benchmark_duration
```

Output token throughput is:

```text
output_token_throughput =
total_output_tokens / benchmark_duration
```

These metrics describe system-level serving capacity rather than the latency of an individual request.

## 12. Latency Statistics

For each concurrency level, latency distributions are calculated separately.

Reported statistics include:

```text
min
mean
p50
p90
p95
p99
max
```

for:

* TTFT
* TPOT
* E2E latency

The experiment does not replace the distributions with only averages because increasing concurrency can change tail latency substantially.

## 13. GPU Monitoring

GPU telemetry is sampled approximately every 50 ms during the measured workload.

The sampler records:

* timestamp
* GPU index
* GPU name
* GPU utilization
* memory used
* memory total
* temperature
* power draw

Warm-up GPU samples are discarded. The saved GPU data therefore corresponds only to the official 64-request measurement period.
GPU sampling is asynchronous and may not capture every short-lived utilization transition. In particular, GPU samples should not be interpreted as exact boundaries between prefill and decode.

## 14. Repetitions

Each concurrency condition is measured once.

Therefore:

```text
repetitions per condition = 1
```

The experiment prioritizes a controlled first characterization of concurrency behavior while limiting GPU rental time.
If a later run reveals an obvious measurement anomaly, the condition may be rerun as a diagnostic, but such a rerun is not part of the planned primary experiment.

## 15. Experimental Order

Conditions are executed in ascending order:

```text
1 → 2 → 4 → 8 → 16 → 32
```

The same vLLM server remains active throughout.
Between conditions, the previous workload must fully complete before the next warm-up begins.

## 16. Controlled Variables

The following remain fixed:

* model
* GPU
* GPU count
* tensor parallelism
* server configuration
* prompt
* maximum output tokens
* temperature
* top-p
* top-k
* min-p
* reasoning mode
* streaming mode
* maximum model length

Only concurrency changes between conditions.

## 17. Expected Analysis

The main analysis will compare concurrency against:

* TTFT distribution
* TPOT distribution
* E2E latency distribution
* request throughput
* output token throughput
* GPU utilization
* GPU memory usage
* GPU temperature
* GPU power

The purpose is to observe how additional concurrent work changes serving behavior rather than to assume in advance that any particular metric must increase or decrease.
