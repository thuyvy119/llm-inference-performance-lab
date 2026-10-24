from __future__ import annotations
from pathlib import Path
from client.request import InferenceClient
from configs.loader import load_yaml
from metrics.gpu import GPUSampler
from metrics.latency import summarize_latency
from benchmarks.common import build_generate_kwargs, build_gpu_metrics_record, get_base_url, print_experiment_header, save_json, print_latency_summary


CONFIG_PATH = Path("configs/experiments/exp1c_repeated_baseline.yaml")
SERVER_CONFIG_PATH = Path("configs/experiments/exp1a_environment.yaml")

GPU_METRICS_PATH = Path("results/raw/exp1c_gpu_metrics.csv")
GPU_SAMPLING_INTERVAL_SECONDS = 0.05
OUTPUT_PATH = Path("results/raw/exp1c_repeated_baseline.json")

def main():
    config = load_yaml(CONFIG_PATH)
    server_config = load_yaml(SERVER_CONFIG_PATH)

    base_url = get_base_url(server_config)
    model = config["model"]
    workload = config["workload"]
    request_config = config["request"]

    num_requests = workload["num_requests"]
    concurrency = workload["concurrency"]

    if concurrency != 1:
        raise ValueError("Experiment 1c requires concurrency=1")

    print_experiment_header("Experiment 1c: Repeated Sequential Baseline", base_url, model, request_config, workload)

    gpu_sampler = GPUSampler(
        output_path=GPU_METRICS_PATH,
        interval_seconds=GPU_SAMPLING_INTERVAL_SECONDS,
    )

    results = []

    with InferenceClient(base_url=base_url) as client:
        gpu_sampler.start()
        try:
            for request_id in range(1, num_requests + 1):
                print(f"\nRunning request {request_id}/{num_requests}...")
                
                result = client.generate(
                    model=model,
                    **build_generate_kwargs(request_config),
                )
                result["request_id"] = request_id
                result["success"] = True
                results.append(result)

                print(f"  TTFT: {result['ttft_seconds']:.4f}s")
                print(f"  E2E: {result['e2e_latency_seconds']:.4f}s")
                print(f"  Output tokens: {result['output_tokens']}")

        finally:
            gpu_samples = gpu_sampler.stop()

    summary = summarize_latency(results)

    output = {
            "experiment": {
                "name": config["name"],
                "model": model
            },
            "workload": workload,
            "request_config": request_config,
            "server": {"base_url": base_url},
            "gpu_metrics": build_gpu_metrics_record(gpu_sampler, gpu_samples, GPU_METRICS_PATH),
            "results": [result],
            "summary": summary,
        }

    save_json(OUTPUT_PATH, output)
    
    print("\n=== Summary ===")
    print_latency_summary(summary)

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(f"GPU metrics saved to: {GPU_METRICS_PATH}")
    
if __name__ == "__main__":
    main()
