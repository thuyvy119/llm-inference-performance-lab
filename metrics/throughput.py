from __future__ import annotations

def compute_throughput_metrics(successful_requests, wall_clock):
    total_output_tokens = sum(result.get("output_tokens", 0) for result in successful_requests)
    if wall_clock<= 0:
        return {"total_output_tokens": total_output_tokens, "request_throughput": None, "output_tokens_throughput": None}
    
    return {
        "total_output_tokens": total_output_tokens,
        "request_throughput": len(successful_requests) / wall_clock,
        "output_tokens_throughput": total_output_tokens / wall_clock,
    }
    