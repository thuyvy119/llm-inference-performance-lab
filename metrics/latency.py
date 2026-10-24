from __future__ import annotations
from statistics import mean

def percentile(values, percentile_value):
    if not values:
        raise ValueError("Cannot calculate percentile of empty data.")

    sorted_values = sorted(values)
    position = ((len(sorted_values) - 1)* percentile_value/ 100)

    lower = int(position)
    upper = min(lower + 1, len(sorted_values)-1)
    weight = position - lower

    return (sorted_values[lower] + weight * (sorted_values[upper] - sorted_values[lower]))

def summarize(results, metric_name):
    values = [
        result[metric_name]
        for result in results
        if result.get(metric_name) is not None
    ]

    if not values:
        raise ValueError(f"No valid values found for metric: {metric_name}")

    return {
        "mean": mean(values),
        "min": min(values),
        "max": max(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
    }

LATENCY_METRICS = ["ttft_seconds", "tpot_seconds", "e2e_latency_seconds"]
def summarize_latency(results):
    summary = {}
    for metric in LATENCY_METRICS:
        valid_results = [result for result in results if result.get(metric) is not None]
        if valid_results:
            summary[metric] = summarize(valid_results, metric)
    return summary