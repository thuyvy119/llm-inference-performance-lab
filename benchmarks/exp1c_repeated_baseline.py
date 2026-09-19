from __future__ import annotations
import json
from pathlib import Path
from client.request import InferenceClient
from configs.loader import load_yaml
from metrics.gpu import GPUSampler
from metrics.latency import summarize


CONFIG_PATH = Path("configs/experiments/exp1c_repeated_baseline.yaml")
SERVER_CONFIG_PATH = Path("configs/experiments/exp1a_environment.yaml")

GPU_METRICS_PATH = Path("results/raw/exp1c_gpu_metrics.csv")
GPU_SAMPLING_INTERVAL_SECONDS = 0.05
OUTPUT_PATH = Path("results/raw/exp1c_repeated_baseline.json")

def main():
    config = load_yaml(CONFIG_PATH)
    server_config = load_yaml(SERVER_CONFIG_PATH)

    host = server_config["server"]["host"]
    port = server_config["server"]["port"]
    base_url = f"http://{host}:{port}"

    model = config["model"]
    workload = config["workload"]
    request_config = config["request"]

    num_requests = workload["num_requests"]
    concurrency = workload["concurrency"]

    if concurrency != 1:
        raise ValueError("Experiment 1c requires concurrency=1")

    client = InferenceClient(base_url=base_url)

    print("Experiment 1c: Repeated Sequential Baseline")
    print("=" * 50)
    print(f"Server: {base_url}")
    print(f"Model: {model}")
    print(f"Requests: {num_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"Max tokens: {request_config['max_tokens']}")
    print(f"Temperature: {request_config['temperature']}")
    print(f"Thinking enabled: {request_config['enable_thinking']}")
    print(f"Top-p: {request_config['top_p']}")
    print(f"Top-k: {request_config['top_k']}")
    print(f"Min-p: {request_config['min_p']}")

    gpu_sampler = GPUSampler(
        output_path=GPU_METRICS_PATH,
        interval_seconds=GPU_SAMPLING_INTERVAL_SECONDS,
    )

    results = []

    gpu_sampler.start()

    try:
        for request_id in range(1, num_requests + 1):
            print(
                f"\nRunning request "
                f"{request_id}/{num_requests}..."
            )

            result = client.generate(
                model=model,
                prompt=request_config["prompt"],
                max_tokens=request_config["max_tokens"],
                temperature=request_config["temperature"],
                top_p=request_config["top_p"],
                top_k=request_config["top_k"],
                min_p=request_config["min_p"],
                enable_thinking=request_config["enable_thinking"],
                stream=request_config["stream"],
            )

            result["request_id"] = request_id
            results.append(result)

            print(
                f"  TTFT: "
                f"{result['ttft_seconds']:.4f}s"
            )
            print(
                f"  E2E: "
                f"{result['e2e_latency_seconds']:.4f}s"
            )
            print(
                f"  Output tokens: "
                f"{result['output_tokens']}"
            )

    finally:
        gpu_samples = gpu_sampler.stop()

    metrics_to_summarize = ["ttft_seconds", "tpot_seconds", "e2e_latency_seconds"]
    summary = {}

    for metric in metrics_to_summarize:
        valid_results = [
            result
            for result in results
            if result.get(metric) is not None
        ]

        if valid_results:
            summary[metric] = summarize(
                valid_results,
                metric,
            )

    output = {
        "experiment": {
            "name": config["name"],
            "model": model,
            "num_requests": num_requests,
            "concurrency": concurrency,
        },
        "workload": {
            "prompt": request_config["prompt"],
            "max_tokens": request_config["max_tokens"],
            "temperature": request_config["temperature"],
            "top_p": request_config["top_p"],
            "top_k": request_config["top_k"],
            "min_p": request_config["min_p"],
            "enable_thinking": request_config["enable_thinking"],
            "stream": request_config["stream"],
        },
        "server": {
            "base_url": base_url,
        },
        "gpu_metrics": {
            "path": str(GPU_METRICS_PATH),
            "sampling_interval_seconds": (
                GPU_SAMPLING_INTERVAL_SECONDS
            ),
            "num_samples": len(gpu_samples),
        },
        "results": results,
        "summary": summary,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)
    print("\n=== Summary ===")
    for metric, statistics in summary.items():
        print(f"\n{metric}:")
        for name, value in statistics.items():
            print(f"  {name}: "
                f"{value:.6f}s")

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(f"GPU metrics saved to: {GPU_METRICS_PATH}")
    
if __name__ == "__main__":
    main()
