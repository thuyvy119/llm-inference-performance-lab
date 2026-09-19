# Experiment 2 — Design Decisions

## Decision 1 — Six Concurrency Levels

The experiment uses:

```text
1, 2, 4, 8, 16, 32
```

These levels provide progressively larger serving loads while keeping the experiment small enough to execute on a single RTX 3090.
The powers-of-two progression also makes scaling behavior easier to inspect.

## Decision 2 — 64 Measured Requests Per Condition

Each concurrency condition contains exactly:

```text
64 measured requests
```

Therefore the official dataset contains:

```text
6 × 64 = 384 measured requests
```

The same request count is used at every concurrency level so that the conditions have a consistent measurement budget.

## Decision 3 — One Official Repetition

Each concurrency condition is run once.

```text
repetitions = 1
```

The experiment is intended as the first controlled characterization of concurrency scaling. A second run is not part of the primary protocol because the goal is to complete the full set of experiments before renting GPU time again.
If a later run reveals an obvious measurement anomaly, the condition may be rerun as a diagnostic, but such a rerun is not part of the planned primary experiment.

## Decision 4 — Eight Warm-Up Requests

Each concurrency condition begins with:

```text
8 warm-up requests
```

The warm-up workload uses the same request configuration as the measured workload.
Warm-up results are discarded completely.

### Why do we need warm-up?

The first requests sent to an inference server may not represent the steady behavior we want to measure.
For example, the serving system may be transitioning from an initially idle state, establishing reusable connections, initializing runtime components, or moving into its normal GPU execution state. These effects are part of operating the serving system, but they are not the concurrency behavior that Experiment 02 is designed to study.

Therefore, a short warm-up phase separates:

```text
initial / transition behavior
```

from:

```text
official measurement behavior
```

The warm-up is not intended to "make the GPU faster." Its purpose is to reduce the influence of initial transient behavior on the measured workload.

The warm-up requests therefore:

* use the same model,
* use the same prompt,
* use the same generation parameters,
* use the same target concurrency,
* are allowed to complete normally, and
* have their latency, throughput, and GPU measurements discarded.

### Why exactly 8 requests?

Eight requests provide a simple fixed warm-up budget for every condition without making the GPU experiment unnecessarily long. This is deliberately kept separate from the 64-request measurement workload. At high concurrency levels, such as 16 and 32, eight warm-up requests cannot fill every possible concurrency slot. Therefore, this warm-up should be interpreted as a **short stabilization phase**, not as a full-load warm-up.
The official 64-request workload is responsible for exercising and measuring the target concurrency.

## Decision 5 — No Server Restart Between Conditions

The vLLM server is started once and remains running while all six concurrency conditions are executed. Restarting vLLM between every condition would introduce additional startup and model-loading behavior that is unrelated to the concurrency variable.

The benchmark instead performs:

```text
warm-up
→ measurement
→ finish completely
→ next warm-up
→ next measurement
```

Before beginning the next condition, all requests from the previous condition must have completed.

## Decision 6 — Rolling Concurrency Rather Than Fixed Waves

Concurrency `N` means:

> At most N requests are simultaneously in flight.

The scheduler continuously replaces completed requests until all 64 measured requests have completed.

For example, at `N = 4`:

```text
initially:
R1 R2 R3 R4

when R2 completes:
R5 starts

when R4 completes:
R6 starts

...
```

The benchmark does not wait for all four requests to finish before scheduling replacements.
This prevents the benchmark driver from introducing artificial idle gaps between request groups.

## Decision 7 — Benchmark Timer Excludes Warm-Up

The timer begins only after warm-up has completed.

Conceptually:

```text
warm-up
    ↓
discard warm-up
    ↓
start GPU sampler
    ↓
start benchmark timer
    ↓
64 measured requests
    ↓
last request completes
    ↓
stop benchmark timer
    ↓
stop GPU sampler
```

This makes the benchmark duration represent only the official measured workload.

## Decision 8 — GPU Sampling Excludes Warm-Up

The GPU sampler follows the same measurement boundary.

Warm-up:

```text
GPU sampling: ignored
```

Measured workload:

```text
GPU sampling: recorded
```

This keeps the GPU CSV aligned with the official measurement window.

## Decision 9 — Request-Level and System-Level Metrics Are Separate

Request-level metrics answer:

> How long does an individual request take?

System-level metrics answer:

> How much work can the serving system complete during the measured interval?

For example:

```text
TTFT / E2E / TPOT
```

are request-level metrics, while:

```text
requests/sec
output tokens/sec
```

are system-level metrics.
Both are required because increasing concurrency can improve overall throughput while simultaneously changing individual-request latency.

## Decision 10 — Use API Token Counts

The benchmark records input and output token counts from the API usage information.
Streaming chunks are not treated as tokens. This avoids incorrectly estimating token counts from the number of SSE events received.

## Decision 11 — Preserve Failed Requests

A request that fails is recorded rather than silently ignored.

The experiment should report:

```text
total_requests
successful_requests
failed_requests
```

Throughput is based on successfully completed requests.
Failures remain visible because silently dropping them could make a heavily loaded condition appear artificially faster.

## Decision 12 — Ascending Execution Order

The conditions run:

```text
1 → 2 → 4 → 8 → 16 → 32
```

This provides a simple and deterministic execution protocol.
Potential order effects such as GPU temperature or power-state changes are recorded through GPU telemetry.
Randomized condition order is intentionally not used for this first experiment because reproducibility and implementation simplicity are prioritized.
