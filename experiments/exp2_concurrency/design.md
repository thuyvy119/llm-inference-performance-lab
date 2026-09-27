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

## Decision 4 — No Server Restart Between Conditions

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

## Decision 5 — Rolling Concurrency Rather Than Fixed Waves

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
Implementation: an `asyncio.Semaphore(N)` wraps a fixed list of request coroutines, dispatched via `asyncio.gather`. This gives the rolling behavior described above directly — no separate scheduling logic was needed beyond the semaphore itself.

## Decision 6 — Request-Level and System-Level Metrics Are Separate

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

## Decision 7 — Use API Token Counts

The benchmark records input and output token counts from the API usage information.
Streaming chunks are not treated as tokens. This avoids incorrectly estimating token counts from the number of SSE events received.

## Decision 8 — Preserve Failed Requests

A request that fails is recorded rather than silently ignored.

The experiment should report:

```text
total_requests
successful_requests
failed_requests
```

Throughput is based on successfully completed requests.
Failures remain visible because silently dropping them could make a heavily loaded condition appear artificially faster.

## Decision 9 — Ascending Execution Order

The conditions run:

```text
1 → 2 → 4 → 8 → 16 → 32
```

This provides a simple and deterministic execution protocol.
Potential order effects such as GPU temperature or power-state changes are recorded through GPU telemetry.
Randomized condition order is intentionally not used for this first experiment because reproducibility and implementation simplicity are prioritized.

## Decision 10 - Async Concurrent Client Instead of a Thread Pool

Options considered:
A synchronous client dispatched across a ThreadPoolExecutor.
An async client (httpx.AsyncClient) dispatched via asyncio.

What I chose:
AsyncInferenceClient in client/request.py, using httpx.AsyncClient and asyncio.Semaphore/asyncio.gather for concurrent dispatch.

Why:
Request-level work here is I/O-bound (waiting on network/streaming response), which asyncio handles with lower overhead per concurrent unit than OS threads, and makes the rolling-replacement semaphore pattern (Decision 5) a natural, direct expression rather than something built on top of thread-pool primitives. It also keeps InferenceClient (sync, Exp1/3/4) and AsyncInferenceClient (async, Exp2/5) side by side in one file sharing result-construction logic (build_result, build_payload), rather than requiring a second client file.

Tradeoff / what I gave up:
Async code is somewhat less familiar to reason about than a thread pool for beginner to Python concurrency; failure modes (an exception inside a gathered task) need explicit handling rather than propagating naturally, which is why run_concurrent_workload wraps each request in its own try/except (see Decision 8, Preserve Failed Requests).

