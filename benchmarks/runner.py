from __future__ import annotations
import asyncio
import time
from client.request import AsyncInferenceClient
from typing import Any

# duplicate 1 request config N times
def uniform_requests(request_configs: dict[str, Any], num_requests: int):
    return [dict(request_configs) for _ in range(num_requests)]


async def run_concurrent_workload(client: AsyncInferenceClient, model: str, request_configs: list[dict[str, Any]], concurrency: int, ):
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    
    async def _run_one(request_id: int, request_config: dict[str, Any]):
        async with semaphore:
            try:
                res = await client.generate(model=model,
                                            prompt=request_config["prompt"],
                                            max_tokens=request_config["max_tokens"],
                                            temperature=request_config["temperature"],
                                            top_p=request_config["top_p"],
                                            top_k=request_config["top_k"],
                                            min_p=request_config["min_p"],
                                            enable_thinking=request_config.get("enable_thinking", False),
                                            stream=request_config.get("stream", True),
                                            ignore_eos=request_config.get("ignore_eos", False),)
                res["request_id"] = request_id
                res["success"] = True
            except Exception as e:
                res = {"request_id": request_id, "success": False, "error": str(e)}
            results.append(res)
            
    start = time.perf_counter()
    asyncio.gather(
        *(_run_one(i, cfg) for i, cfg in enumerate(request_configs, start=1))
    )
    wall_lock = time.perf_counter() - start
    results.sort(key=lambda r: r["request_id"])
    return results, wall_lock
    