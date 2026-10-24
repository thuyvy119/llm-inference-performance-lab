from __future__ import annotations
import asyncio
# from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
from pathlib import Path
from client.request import AsyncInferenceClient
from configs.loader import load_yaml
from benchmarks.common import build_generate_kwargs, build_gpu_metrics_record, get_base_url, print_experiment_header, save_json
from benchmarks.runner import uniform_requests, run_concurrent_workload
from metrics.gpu import GPUSampler
from metrics.latency import summarize_latency, print_latency_summary
from metrics.throughput import compute_throughput_metrics


CONFIG_PATH = Path("configs/experiments/exp2_concurrency.yaml")
SERVER_CONFIG_PATH = Path("configs/experiments/exp1a_environment.yaml")
OUTPUT_PATH = Path("results/raw/exp2_concurrency.json")

def gpu_metrics_path(concurrency: int):
    return Path(f"results/raw/exp2_gpu_metrics_c{concurrency}.csv")

async def run_condition(client, model, request_config, num_requests, concurrency):
    gpu_sampler = GPUSampler(output_path=gpu_metrics_path(concurrency))
    request_configs = uniform_requests(request_config, num_requests)
    gpu_sampler.start()
    try:
        results, wall_clock = await run_concurrent_workload(client, model, request_configs, concurrency)
    finally:
        gpu_samples= gpu_sampler.stop()
    successful_requests = [result for result in results if result['success']]
    failed_requests = [result for result in results if not result['success']]
    return {"concurrency": concurrency,
            "num_requests": len(results),
            "wall_clock": wall_clock,
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            **compute_throughput_metrics(successful_requests, wall_clock),
            "summary": summarize_latency(successful_requests),
            "gpu_metrics": build_gpu_metrics_record(gpu_sampler, gpu_samples, gpu_metrics_path(concurrency)),
            "results": results,}

async def main_async():
    config = load_yaml(CONFIG_PATH)
    server_config = load_yaml(SERVER_CONFIG_PATH)
    base_url = get_base_url(server_config)
    model = config["model"]
    workload = config["workload"]
    request_config = config["request"]
    
    concurrency_levels = workload["concurrency_levels"]
    requests_per_condition = workload["requests_per_condition"]
    
    print("Experiment 2: Concurrency Sweep")
    print_experiment_header("Experiment 2: Concurrency Sweep", base_url, model, request_config, workload)
    
    conditions = []
    
    async with AsyncInferenceClient(base_url=base_url) as client:
        for concurrency in concurrency_levels:
            print(f"\n---Concurrency:{concurrency}---")
            condition = await run_condition(client, model, request_config, requests_per_condition, concurrency)
            conditions.append(condition)
            
            print(f"Successful requests: {condition['successful_requests']}/{condition['num_requests']}"
                  f"Failed requests: {condition['failed_requests']}")
            print(f"Wall clock time: {condition['wall_clock']:.2f}s")
            if condition['request_throughput'] is not None:
                print(f"Request throughput: {condition['request_throughput']:.2f} requests/s")
                print(f"Output tokens throughput: {condition['output_tokens_throughput']:.2f} tokens/s")
            print_latency_summary(condition['summary'])
            
    output = {"experiment": {"name": config["name"], "model": model},
              "workload": {"concurrency_levels": concurrency_levels,
                           "requests_per_condition": requests_per_condition,},
              "request": request_config,
              "server": {"base_url": base_url},
              "conditions": conditions,}
    
    save_json(OUTPUT_PATH, output)
    
    print(f"\nResults saved to: {OUTPUT_PATH}")

def main():
    asyncio.run(main_async())
    
if __name__ == "__main__":
    main()
    
            