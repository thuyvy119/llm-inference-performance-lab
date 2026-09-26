# Experiment 1 — Design Decisions

## Purpose

Experiment 1 is designed to establish a controlled and reproducible reference point for LLM inference performance.
The experiment intentionally changes as few workload variables as possible. Later experiments can then introduce one major variable at a time and compare the resulting measurements against this baseline.

## 1. Why Qwen3-0.6B?

Qwen/Qwen3-0.6B was selected as the initial model because it is small enough to make iteration practical while still providing a realistic autoregressive LLM inference workload.
The model also provides a useful foundation for studying both standard generation and reasoning-enabled generation in later experiments.

## 2. Why Disable Qwen3 Thinking?

Qwen3 supports a thinking/reasoning mode that can introduce additional generated tokens and different generation behavior. If reasoning were enabled during the baseline, differences in output length and generation behavior could become an additional source of variation. Therefore, Experiment 01 explicitly disables thinking:

```python
chat_template_kwargs = {
    "enable_thinking": False
}
```

This makes the baseline represent standard prompt-to-answer generation. Reasoning-enabled inference will be introduced later as a separate workload variable.
This separation allows the project to ask a clearer experimental question later:

> How does enabling Qwen3 reasoning change latency, output length, throughput, and GPU behavior relative to the standard-generation baseline?

## 3. Why One GPU?

Experiment 01 uses a single RTX 3090 with:

```text
tensor_parallel_size = 1
```

The purpose is to remove multi-GPU communication from the initial measurement.
Multi-GPU execution introduces additional variables such as tensor-parallel communication and NCCL behavior, which are outside the scope of the first baseline.

## 4. Why Concurrency 1?

The baseline uses:

```text
concurrency = 1
```

and each request is started only after the previous request completes.
This isolates single-request behavior before introducing concurrent scheduling and batching effects.
Later experiments can compare concurrency levels such as:

```text
1, 2, 4, 8, 16, 32
```

against the Experiment 1 baseline.

## 5. Why Run a Single Request Before Repeated Requests?

Experiment 1b is a pipeline validation experiment.

It verifies that:
* the server is reachable
* requests are correctly formed
* streaming works
* TTFT can be measured
* completion can be detected
* token usage can be collected
* GPU sampling works
* result artifacts are written correctly

Only after this pipeline is validated is the repeated workload executed in Experiment 1c.

## 6. Why 20 Sequential Requests?

Experiment 1c uses 20 requests to observe request-to-request variability while keeping concurrency fixed at 1.
The goal is not to characterize production-scale throughput. Instead, the repeated workload provides enough observations to calculate latency distribution statistics and identify unusually slow or fast requests.

## 7. Why Streaming?

Streaming is enabled because the experiment measures TTFT.
Without streaming, the client would only observe the completed response and could not directly measure when the first generated output became available.

Streaming also makes it possible to separate the first-output interval from the subsequent decode interval.

## 8. Why API Token Usage?

The benchmark uses the token counts reported by the API rather than counting streaming response chunks.
A streaming chunk is a transport-level event and does not necessarily represent exactly one generated token.
Using the server-reported usage therefore provides a more appropriate basis for calculating TPOT and output token throughput.

## 9. Why Fixed Sampling Parameters?

The baseline uses fixed generation parameters:
```text
temperature = 0.7
top_p       = 0.8
top_k       = 20
min_p       = 0.0
```

Keeping these parameters constant prevents generation configuration from becoming an uncontrolled workload variable.

## 10. Why `max_tokens = 128` and why `ignore_eos = true`??

The baseline limits generated output to 128 tokens.Combined with `ignore_eos = true` (see Section 11 below), this makes output length a genuinely controlled variable rather than a ceiling — every successful request generates exactly 128 tokens. This matters for later experiments (particularly the planned output-length sweep) that need `output_tokens` to be the value actually set, not a sampling-dependent outcome.

## 11. Why `ignore_eos = true`?

At `temperature = 0.7` with `top_p = 0.8, top_k = 20`, the token sampled at each decode step — including whether it happens to be the end-of-sequence token — is stochastic. Without `ignore_eos`, two requests with identical config can legitimately produce very different `output_tokens` counts (e.g. 40 vs 115), purely from where sampling hit
EOS. Since `decode_duration`, `E2E latency`, and (to a lesser extent, since it's per-token-normalized) `TPOT` are direct functions of how many tokens were actually generated, this would make output length an uncontrolled confound riding along with every other comparison.

Setting `ignore_eos = true` forces decoding to continue to `max_tokens` regardless of EOS, so `output_tokens == max_tokens` deterministically.
Note this does not disable sampling generally, temperature/top_p/top_k still govern which token is chosen at each step; `ignore_eos` only prevents the EOS token specifically from ending generation early.

Reference: https://docs.vllm.ai/api/vllm/sampling_params.htm

## 12. Why GPU Sampling at Approximately 50 ms?

GPU monitoring is performed as a time series rather than as a single snapshot.

Approximately 50-ms sampling provides a coarse view of:
* GPU utilization
* memory usage
* temperature
* power

The sampling interval is not intended to provide kernel-level timing.
More detailed GPU behavior will be studied later using Nsight Systems and Nsight Compute.

## 13. Controlled Variables

The following variables remain fixed across Experiments 1b and 1c:
* model
* GPU
* tensor parallelism
* maximum model length
* GPU memory utilization target
* prompt
* max output tokens
* sampling parameters
* Qwen3 reasoning configuration
* server endpoint

The primary difference between 1b and 1c is the number of requests.

