from __future__ import annotations
from statistics import mean
from typing import Any

def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        raise ValueError("Cannot calculate percentile of empty data.")

    sorted_values = sorted(values)
    position = ((len(sorted_values) - 1)* percentile_value/ 100)

    lower = int(position)
    upper = min(lower + 1, len(sorted_values))
    weight = position - lower

    return (sorted_values[lower] + weight * (sorted_values[upper] - sorted_values[lower]))

def summarize(results: list[dict[str, Any]], metric_name: str) -> dict[str, float]:
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