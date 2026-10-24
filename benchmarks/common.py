from __future__ import annotations
import json
from pathlib import Path

def get_base_url(server_config):
    host = server_config["server"]["host"]
    port = server_config["server"]["port"]
    return f"http://{host}:{port}"

def build_generate_kwargs(request_config):
    return {
        "prompt": request_config["prompt"],
        "max_tokens": request_config["max_tokens"],
        "temperature": request_config["temperature"],
        "top_p": request_config["top_p"],
        "top_k": request_config["top_k"],
        "min_p": request_config["min_p"],
        "enable_thinking": request_config.get("enable_thinking", False),
        "stream": request_config.get("stream", True),
        "ignore_eos": request_config.get("ignore_eos", False),
    }
    
def build_gpu_metrics_record(gpu_sampler, gpu_samples, path):
    return {
        "path": str(path),
        "sampling_interval_seconds": gpu_sampler.interval_seconds,
        "num_samples": len(gpu_samples),
    }

def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def print_experiment_header(title, base_url, model, request_config, workload):
    print(title)
    print("=" * 50)
    print(f"Server: {base_url}")
    print(f"Model: {model}")
    print(f"Prompt: {request_config['prompt']}")
    print(f"Request Config: {request_config}")
    for label, value in workload.items():
        print(f"{label}: {value}")
    print(f"Max tokens: {request_config['max_tokens']}")
    print(f"Temperature: {request_config['temperature']}")
    print(f"Thinking enabled: {request_config['enable_thinking']}")
    print(f"Top-p: {request_config['top_p']}")
    print(f"Top-k: {request_config['top_k']}")
    print(f"Min-p: {request_config['min_p']}")
    print(f"Ignore EOS: {request_config.get('ignore_eos', False)}")
    
def print_latency_summary(summary):
    print("\n=== Summary ===")
    for metric, statistics in summary.items():
        print(f"\n{metric}:")
        for name, value in statistics.items():
            print(f"  {name}: {value:.6f}s")