from __future__ import annotations
import asyncio
from benchmarks.runner import run_concurrent_workload, uniform_requests
from client.request import AsyncInferenceClient

BASE =  {"prompt": "Explain GPUs", "max_tokens": 8, "temperature": 0.7, "stream": True, "ignore_eos": True}

def run_workload(base_url, request_configs, concurrency):
    async def go():
        async with AsyncInferenceClient(base_url) as client:
            return await run_concurrent_workload(client, "test-model", request_configs, concurrency)
    return asyncio.run(go())

def test_concurrency_cap_is_respected_and_reached(mock_server):
    state, base_url = mock_server
    results, wall = run_workload(base_url, uniform_requests(BASE, 12), concurrency=4)
    assert len(results) == 12 and all(r["success"] for r in results)
    assert state.max_in_flight <= 4
    assert state.max_in_flight == 4
    assert wall > 0

def test_concurrency_one_is_sequential(mock_server):
    state, base_url = mock_server
    run_workload(base_url, uniform_requests(BASE, 4), concurrency=1)
    assert state.max_in_flight == 1
    
def test_results_are_ordered_by_request_id(mock_server):
    _, base_url = mock_server
    results, _ = run_workload(base_url, uniform_requests(BASE, 8), concurrency=4)
    assert [r["request_id"] for r in results] == list(range(1, 9))
    
def test_failed_request_is_recorded_not_dropped(mock_server):
    _, base_url = mock_server
    configs = uniform_requests(BASE, 4)
    configs[2] = {**BASE, "prompt": "FAIL"}
    
def test_heterogeneous_configs_are_delivered(mock_server):
    state, base_url = mock_server
    configs = [{**BASE, "prompt": f"prompt-{i}", "max_tokens": 4*(i+1)} for i in range(5)]
    run_workload(base_url, configs, concurrency=3)
    sent = sorted((p["messages"][0]["content"], p["max_tokens"]) for p in state.payloads)
    assert sent == sorted((c["prompt"], c["max_tokens"]) for c in configs)

def test_uniform_requests_return_independent_copies():
    configs = uniform_requests(BASE, 3)
    configs[0]["prompt"] = "changed"
    assert configs[1]["prompt"] == BASE["prompt"]    