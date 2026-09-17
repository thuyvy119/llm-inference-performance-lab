from __future__ import annotations

import json
from pathlib import Path

from client.request import InferenceClient
from configs.loader import load_yaml
from metrics.gpu import GPUSampler

CONFIG_PATH = Path("configs/experiments/exp1b_single_request.yaml")
SERVER_CONFIG_PATH = Path("configs/experiments/exp1a_environment.yaml")

OUTPUT_PATH = Path("results/raw/exp1b_single_request.json")
GPU_METRICS_PATH = Path("results/raw/exp1b_gpu_metrics.csv")
GPU_SAMPLING_INTERVAL_SECONDS = 0.05


def main():
    config = load_yaml(CONFIG_PATH)
    server_config = load_yaml(SERVER_CONFIG_PATH)

    host = server_config["server"]["host"]
    port = server_config["server"]["port"]
    base_url = f"http://{host}:{port}"

    model = config["model"]
    request_config = config["request"]

    client = InferenceClient(base_url=base_url)

    print("Experiment 1b: Single Request")
    print("=" * 40)
    print(f"Server: {base_url}")
    print(f"Model: {model}")
    print(f"Prompt: {request_config['prompt']}")
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

    gpu_sampler.start()
    try:
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
    finally:
        gpu_samples = gpu_sampler.stop()

    
    print("\n=== Results ===")
    if result["ttft_seconds"] is not None:
        print(f"TTFT: {result['ttft_seconds']:.4f} seconds")
    else:
        print("TTFT: unavailable")

    if result["decode_duration_seconds"] is not None:
        print(
            f"Decode duration: "
            f"{result['decode_duration_seconds']:.4f} seconds"
        )
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
            f"Output throughput: "
            f"{result['output_tokens_per_second']:.2f} tokens/s"
        )
    else:
        print("Output throughput: unavailable")

    print("\n=== Response ===")
    print(result["response_text"])


    output = {
        "experiment": {
            "name": config["name"],
            "model": model,
            "num_requests": 1,
            "concurrency": 1,
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
            "sampling_interval_seconds": GPU_SAMPLING_INTERVAL_SECONDS,
            "num_samples": len(gpu_samples),
        },
        "results": [result],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2)

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(f"GPU metrics saved to: {GPU_METRICS_PATH}")


if __name__ == "__main__":
    main()
