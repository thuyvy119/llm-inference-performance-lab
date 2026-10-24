from __futture__ import annotations
import argparse
import asyncio
from pathlib import Path
from client.request import AsyncInferenceClient
from configs.loader import load_yaml
from benchmarks.runner import uniform_requests, run_concurrent_workload

SERVER_CONFIG_PATH = Path("configs/experiments/exp1a_environment.yaml")

async def main_async(num_requests: int, concurrency: int):
    server_config = load_yaml(SERVER_CONFIG_PATH)
    base_url = f"http://{server_config['server']['host']}:{server_config['server']['port']}"
    model = server_config.get("model", "Qwen/Qwen3-0.6B")
    request_config = {"prompt": "Explain the difference between supervised and unsupervised learning.",
                      "max_tokens": 32,
                      "temperature": 0.7,
                      "top_p": 0.8,
                      "top_k": 20,
                      "min_p": 0.0,
                      "enable_thinking": False,
                      "stream": True,
                      "ignore_eos": False,}
    
    async with AsyncInferenceClient(base_url=base_url) as client:
        request_configs = uniform_requests(request_config, num_requests)
        results, wall_lock = await run_concurrent_workload(client, model, request_configs, num_requests, concurrency)
    successful_requests = sum(1 for result in results if result.get('success'))
    print(f"Warm-up: {successful_requests}/{num_requests} requests succeeded in {wall_lock:.2f}s.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=32)
    args = parser.parse_args()
    asyncio.run(main_async(args.num_requests, args.concurrency))
    