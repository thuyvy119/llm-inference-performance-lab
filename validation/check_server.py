"""
Validate the vLLM server for Experiment 1

This script checks server connectivity and verifies that the expected
model is available through the OpenAI-compatible API
"""

from __future__ import annotations
from pathlib import Path
import httpx
from configs.loader import load_yaml

CONFIG_PATH = Path("configs/experiments/exp1a_environment.yaml")


def check_server(config: dict) -> None:
    """
    Check whether the vLLM server is reachable and serving the expected model

    Args:
        config: Experiment configuration
    """
    host = config["server"]["host"]
    port = config["server"]["port"]
    expected_model = config["model"]

    base_url = f"http://{host}:{port}"

    print("Experiment 1 Server Validation")
    print("=" * 40)
    print(f"Server URL: {base_url}")
    print(f"Expected model: {expected_model}")

    try:
        response = httpx.get(
            f"{base_url}/v1/models",
            timeout=5.0,)
    except httpx.RequestError as error:
        print("\nFAIL: Could not connect to vLLM server.")
        print(f"Error: {error}")
        return

    print(f"\nHTTP status: {response.status_code}")

    if response.status_code != 200:
        print("FAIL: Server returned a non-200 response.")
        print(response.text)
        return

    data = response.json()
    models = data.get("data", [])
    model_names = [
        model.get("id")
        for model in models
    ]

    print(f"Available models: {model_names}")

    if expected_model in model_names:
        print(f"PASS: Expected model is available: {expected_model}")
    else:
        print(
            "FAIL: Expected model was not found: "
            f"{expected_model}"
        )

if __name__ == "__main__":
    config = load_yaml(CONFIG_PATH)
    check_server(config)