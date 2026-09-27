# Experiment 2 — Concurrency Scaling

## Overview

Experiment 2 measures how increasing request concurrency affects inference performance when serving `Qwen/Qwen3-0.6B` with vLLM on a single NVIDIA RTX 3090.

The experiment focuses on the relationship between:

* request concurrency,
* per-request latency,
* system request throughput,
* output token throughput, and
* GPU utilization.

Experiment 1 established the single-request and repeated sequential baseline at concurrency 1. Experiment 2 extends that baseline by increasing the number of simultaneously active requests while keeping the model, hardware, request configuration, and server configuration fixed.

The primary independent variable is **request concurrency**.

## Research Question

> How does increasing request concurrency affect per-request latency, system throughput, and GPU utilization when serving Qwen/Qwen3-0.6B with vLLM on a single NVIDIA RTX 3090?

## Relationship to Experiment 1

The concurrency = 1 condition in this experiment uses the same workload,
same server configuration, and — as of this experiment — the same
absence of per-condition warm-up as Experiment 1c's repeated baseline.
The two should be directly comparable. A material discrepancy between
Experiment 1c's summary statistics and this experiment's concurrency = 1
condition is a signal to investigate before trusting the rest of the
sweep.

## Experimental Conditions

The benchmark evaluates six concurrency levels:

```text
1
2
4
8
16
32
```

For each concurrency level:

1. Start the GPU sampler.
2. Start the benchmark timer.
3. Execute 64 requests using a rolling scheduler at the target concurrency.
4. Record all request-level and system-level metrics.
5. Stop measurement after the final request completes.
6. Proceed to the next concurrency level.

There is no per-condition warm-up (see design.md, "One-Time
Server-Readiness Warm-Up"). A single warm-up is run manually once, after
the server starts and before any experiment — including this one —
begins. It is not part of this experiment's measured protocol.

The vLLM server is not restarted between concurrency levels.

## Request Counts

Each concurrency condition contains 64 measured requests.

```text
6 concurrency levels × 64 measured requests = 384 measured requests
```

All 384 requests contribute to the experiment's reported results — there
is no separate warm-up count to exclude.

## Target Environment

| Component                     | Configuration           |
| ------------------------------ | ------------------------ |
| Model                          | Qwen/Qwen3-0.6B          |
| GPU                             | NVIDIA GeForce RTX 3090  |
| GPU count                       | 1                        |
| Tensor parallelism              | 1                        |
| vLLM                             | 0.29.0                  |
| Max model length                 | 4096                    |
| GPU memory utilization target    | 0.90                     |
| Automatic prefix caching         | Disabled                 |
| Reasoning                        | Disabled                 |
| Concurrency levels                | 1, 2, 4, 8, 16, 32      |
| Measured requests per condition   | 64                       |
| Warm-up                          | One-time, server-level, outside this experiment |
| Streaming                        | Enabled                  |

## Fixed Request Configuration

The request workload remains identical to Experiment 1:

```text
Prompt:
Explain the difference between GPU and CPU in simple terms.

max_tokens: 128
temperature: 0.7
top_p: 0.8
top_k: 20
min_p: 0.0
enable_thinking: false
stream: true
ignore_eos: true
```

`ignore_eos: true` ensures every successful request generates exactly
128 output tokens, so output length does not vary across requests or
concurrency levels as an uncontrolled confound — see Experiment 1
design.md for the full rationale.

Automatic prefix caching is disabled server-side, consistent with
Experiment 1, so that every request's prefill is measured cold rather
than as a cache hit against a previous request's identical prompt.

The purpose is to isolate the effect of concurrency rather than
simultaneously changing the workload.

## Concurrency Definition

For this experiment, concurrency `N` means:
> At most `N` requests are simultaneously in flight.

The benchmark uses a rolling concurrency model rather than fixed request
batches, implemented as an `asyncio.Semaphore(N)` guarding a fixed set of
request tasks (`benchmarks/runner.py::run_concurrent_workload`). At
concurrency 4, four requests may initially be active. When one request
completes, the benchmark immediately schedules another, maintaining up
to four active requests until all 64 measured requests have completed.

Thus, concurrency controls the maximum number of simultaneously active
requests rather than defining fixed request waves.

## Measurement Window

```text
start GPU sampling
        ↓
start benchmark timer
        ↓
64 measured requests (rolling scheduler)
        ↓
last measured request completes
        ↓
stop benchmark timer
        ↓
stop GPU sampling
```

## Primary Metrics

### Per-request metrics

For each measured request:

* Time to First Token (TTFT)
* decode duration
* end-to-end latency
* Time Per Output Token (TPOT)
* input token count
* output token count
* request start time
* first-token time
* completion time
* request success/failure

### System-level metrics

For each concurrency level:

* benchmark duration
* completed request count
* failed request count
* request throughput
* total output token count
* output token throughput

Request throughput:

```text
request throughput =
successful completed requests / benchmark duration
```

Output token throughput:

```text
output token throughput =
total output tokens / benchmark duration
```

### Latency distributions

For each concurrency level, report:

* minimum
* mean
* p50
* p90
* p95
* p99
* maximum

for:

* TTFT
* TPOT
* end-to-end latency

### GPU metrics

GPU telemetry is sampled approximately every 50 ms during the measured workload:

* GPU utilization
* GPU memory used
* GPU memory total
* GPU temperature
* GPU power draw

GPU sampling is descriptive and asynchronous. It is not synchronized exactly to individual prefill or decode phases.

## Experimental Order

Concurrency levels are executed in ascending order:

```text
1 → 2 → 4 → 8 → 16 → 32
```

The server remains running throughout the experiment.
This ordering keeps the experiment simple and reproducible. Potential order effects, such as GPU temperature or power-state changes, are recorded rather than hidden.

## Scope

Experiment 2 isolates concurrency scaling.

It does not intentionally vary:

* input length,
* output length,
* model,
* GPU,
* tensor parallelism,
* sampling parameters,
* reasoning mode, or
* serving framework.

Input-length scaling and output-length scaling are investigated separately in Experiments 3 and 4.
