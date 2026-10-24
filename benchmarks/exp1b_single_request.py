from __future__ import annotations
from pathlib import Path
from client.request import InferenceClient
from configs.loader import load_yaml
from metrics.gpu import GPUSampler
from benchmarks.common import build_generate_kwargs, get_base_url, build_gpu_metrics_record, print_experiment_header, save_json

CONFIG_PATH = Path("configs/experiments/exp1b_single_request.yaml")
SERVER_CONFIG_PATH = Path("configs/experiments/exp1a_environment.yaml")

OUTPUT_PATH = Path("results/raw/exp1b_single_request.json")
GPU_METRICS_PATH = Path("results/raw/exp1b_gpu_metrics.csv")


def main():
    config = load_yaml(CONFIG_PATH)
    server_config = load_yaml(SERVER_CONFIG_PATH)

    base_url = get_base_url(server_config)
    model = config["model"]
    workload = config["workload"]
    request_config = config["request"]

    print_experiment_header("Experiment 1b: Single Request", base_url, model, request_config, workload)

    gpu_sampler = GPUSampler(output_path=GPU_METRICS_PATH)

    with InferenceClient(base_url=base_url) as client:
        gpu_sampler.start()
        try:
            result = client.generate(
                model=model,
                **build_generate_kwargs(request_config),)
        finally:
            gpu_samples = gpu_sampler.stop()

    
    print("\n=== Results ===")
    if result["ttft_seconds"] is not None:
        print(f"TTFT: {result['ttft_seconds']:.4f} seconds")
    else:
        print("TTFT: unavailable")

    if result["decode_duration_seconds"] is not None:
        print(f"Decode duration: {result['decode_duration_seconds']:.4f} seconds")
    else:
        print("Decode duration: unavailable")

    print(f"E2E latency: {result['e2e_latency_seconds']:.4f} seconds")
    print(f"Input tokens: {result['input_tokens']}")
    print(f"Output tokens: {result['output_tokens']}")

    if result["tpot_seconds"] is not None:
        print(f"TPOT: {result['tpot_seconds']:.4f} seconds/token")
    else:
        print("TPOT: unavailable")

    if result["output_tokens_per_second"] is not None:
        print(
            f"Output throughput: {result['output_tokens_per_second']:.2f} tokens/s")
    else:
        print("Output throughput: unavailable")

    print("\n=== Response ===")
    print(result["response_text"])


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
    }

    save_json(OUTPUT_PATH, output)
    
    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(f"GPU metrics saved to: {GPU_METRICS_PATH}")


if __name__ == "__main__":
    main()
