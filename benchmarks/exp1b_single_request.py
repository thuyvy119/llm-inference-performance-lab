from __future__ import annotations
from pathlib import Path
from configs.loader import load_yaml
from client.request import InferenceClient

CONFIG_PATH = Path("configs/experiments/exp1b_single_request.yaml")

SERVER_CONFIG_PATH = Path("configs/experiments/exp1a_environment.yaml")

def main():
    config = load_yaml(CONFIG_PATH)
    server_config = load_yaml(SERVER_CONFIG_PATH)
    host = server_config["server"]["host"]
    port = server_config["server"]["port"]
    base_url = f"http://{host}:{port}"

    model = config["model"]
    request_config = config["request"]
    client = InferenceClient(base_url=base_url)

    print("Experiment 1b: Single Request")
    print("=" * 40)
    print(f"Server: {base_url}")
    print(f"Model: {model}")
    print(f"Prompt: {request_config['prompt']}")
    print(f"Max tokens: {request_config['max_tokens']}")
    print(f"Temperature: {request_config['temperature']}")

    result = client.generate(
        model=model,
        prompt=request_config["prompt"],
        max_tokens=request_config["max_tokens"],
        temperature=request_config["temperature"],
        stream=request_config["stream"],
    )

    print("\n=== Results ===")
    print(f"TTFT: "
        f"{result['ttft_seconds']:.4f} seconds"
        if result["ttft_seconds"] is not None
        else "TTFT: unavailable")

    print(f"Decode duration: "
        f"{result['decode_duration_seconds']:.4f} seconds"
        if result["decode_duration_seconds"] is not None
        else "Decode duration: unavailable")

    print(f"E2E latency: "
        f"{result['e2e_latency_seconds']:.4f} seconds")

    print(f"Input tokens: {result['input_tokens']}")
    print(f"Output tokens: {result['output_tokens']}")

    if result["tpot_seconds"] is not None:
        print(f"TPOT: "
            f"{result['tpot_seconds']:.4f} seconds/token")
    else:
        print("TPOT: unavailable")

    if result["output_tokens_per_second"] is not None:
        print(f"Output throughput: "
            f"{result['output_tokens_per_second']:.2f} tokens/s")
    else:
        print("Output throughput: unavailable")

    print("\n=== Response ===")
    print(result["response_text"])

if __name__ == "__main__":
    main()