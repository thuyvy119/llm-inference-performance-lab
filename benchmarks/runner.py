from __future__ import annotations
import asyncio
import time
from client.request import AsyncInferenceClient
from benchmarks.common import build_generate_kwargs

# duplicate 1 request config N times
def uniform_requests(request_configs, num_requests):
    return [dict(request_configs) for _ in range(num_requests)]


async def run_concurrent_workload(client: AsyncInferenceClient, model, request_configs, concurrency):
    semaphore = asyncio.Semaphore(concurrency)
    results = []
    
    async def _run_one(request_id, request_config):
        async with semaphore:
            try:
                res = await client.generate(model=model, **build_generate_kwargs(request_config),)
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
    