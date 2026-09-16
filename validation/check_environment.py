"""
Validate the software and GPU environment for Experiment 1.

This script does not benchmark performance. It collects environment
information and checks it against the assumptions defined in the
Experiment 1a configuration file
"""

from __future__ import annotations
import platform
import sys
from pathlib import Path
import torch
import json
from configs.loader import load_yaml


CONFIG_PATH = Path("configs/experiments/exp1a_environment.yaml")


def collect_environment(config: dict) -> dict:
    """
    Collect environment information and validation results.
    Args:
        config: Experiment 1a configuration.

    Returns:
        Dictionary containing environment information and checks.
    """
    environment = {
        "python": {"version": platform.python_version(), "executable": sys.executable,},
        "pytorch": {"version": torch.__version__, 
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_version": torch.version.cuda,},
        "gpus": [],
        "vllm": {"installed": False, "version": None,},
        "validation": {},
    }

    # collect GPU info
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()

        for index in range(gpu_count):
            properties = torch.cuda.get_device_properties(index)
            environment["gpus"].append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "memory_gb": round(
                        properties.total_memory / (1024**3),
                        2,
                    ),
                }
            )

    # check vLLM installation
    try:
        import vllm

        environment["vllm"] = {
            "installed": True,
            "version": vllm.__version__,
        }
    except ImportError:
        pass

    # validate experiment assumptions
    expected_gpu_count = config["validation"]["expected_gpu_count"]
    expected_gpu_name = config["validation"]["expected_gpu_name"]

    actual_gpu_count = len(environment["gpus"])

    environment["validation"]["cuda_available"] = (
        torch.cuda.is_available()
    )

    environment["validation"]["gpu_count"] = {
        "expected": expected_gpu_count,
        "actual": actual_gpu_count,
        "passed": actual_gpu_count == expected_gpu_count,
    }

    gpu_name_matches = False

    if environment["gpus"]:
        actual_gpu_name = environment["gpus"][0]["name"]

        gpu_name_matches = (
            expected_gpu_name.lower() in actual_gpu_name.lower()
        )

    environment["validation"]["gpu_name"] = {
        "expected": expected_gpu_name,
        "actual": (
            environment["gpus"][0]["name"]
            if environment["gpus"]
            else None
        ),
        "passed": gpu_name_matches,
    }

    environment["validation"]["vllm_installed"] = (
        environment["vllm"]["installed"]
    )

    return environment

# print the collected environment info
def print_environment(environment: dict) -> None:
    print("Experiment 01 Environment Validation")
    print("=" * 40)
    print("\n=== Python ===")
    print(f"Python version: {environment['python']['version']}")
    print(f"Python executable: {environment['python']['executable']}")
    print("\n=== PyTorch ===")
    print(f"PyTorch version: {environment['pytorch']['version']}")
    print(
        f"CUDA available: "
        f"{environment['pytorch']['cuda_available']}"
    )
    print(
        f"PyTorch CUDA version: "
        f"{environment['pytorch']['cuda_version']}"
    )

    print("\n=== GPUs ===")
    if not environment["gpus"]:
        print("No CUDA GPUs detected.")

    for gpu in environment["gpus"]:
        print(
            f"GPU {gpu['index']}: "
            f"{gpu['name']} "
            f"({gpu['memory_gb']} GB)"
        )

    print("\n=== vLLM ===")

    if environment["vllm"]["installed"]:
        print(
            f"vLLM version: "
            f"{environment['vllm']['version']}"
        )
    else:
        print("vLLM: not installed")

    print("\n=== Experiment Assumptions ===")
    validation = environment["validation"]
    print(
        f"CUDA available: "
        f"{'PASS' if validation['cuda_available'] else 'FAIL'}"
    )
    gpu_count = validation["gpu_count"]
    print(
        f"GPU count: "
        f"{'PASS' if gpu_count['passed'] else 'FAIL'} "
        f"(expected={gpu_count['expected']}, "
        f"actual={gpu_count['actual']})"
    )

    gpu_name = validation["gpu_name"]
    if gpu_name["passed"]:
        print(f"GPU name: PASS ({gpu_name['actual']})")
    else:
        print(
            f"GPU name: WARNING "
            f"(expected={gpu_name['expected']}, "
            f"actual={gpu_name['actual']})"
        )
    print(
        f"vLLM installed: "
        f"{'PASS' if validation['vllm_installed'] else 'FAIL'}"
    )

# Load configuration, collect environment and print results
def main() -> None:
    config = load_yaml(CONFIG_PATH)
    environment = collect_environment(config)
    print_environment(environment)
    
    output_path = Path("results/raw/exp1a_environment.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(environment, f, indent=2)
    print(f"\nEnvironment information saved to: {output_path}")

if __name__ == "__main__":
    main()