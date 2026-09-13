import json
import time
import httpx
from pathlib import Path

SERVER_URL = "http://localhost:8080"
MODEL = "Qwen/Qwen3-0.6B"

PROMPT = "Explain the difference between GPU and CPU in simple terms."
MAX_TOKENS = 128
TEMPERATURE = 0.0

def send_request():
    url = f"{SERVER_URL}/v1/chat/completions"
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": PROMPT
            }
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
    }

    request_start = time.perf_counter()
    first_token_time = None
    # token_count = 0
    last_token_time = None
    generated_chunk = []

    with httpx.stream("POST", url, json=payload, timeout=300) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            data = line[len("data: "):].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content is None:
                continue
            now = time.perf_counter()
            if first_token_time is None:
                first_token_time = now
            # token_count += 1
            last_token_time = now
            generated_chunk.append(content)
            
    request_end = time.perf_counter()

    ttft = (first_token_time - request_start) if first_token_time else None
    e2e_latency = request_end - request_start
    decode_time = (last_token_time - first_token_time) if first_token_time and last_token_time else None
    
    result = {
        "model": MODEL,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "ttft_s": ttft,
        "decode_time_seconds": decode_time,
        "e2e_latency_seconds": e2e_latency,
        "num_streamed_chunks": len(generated_chunk),
        "output_text": "".join(generated_chunk),
    }
    
    return result

def main():
    result = send_request()
    print("\n=== Experiment 01: Single Request ===")
    print(f"TTFT:          {result['ttft_seconds']:.4f} s")
    print(f"E2E latency:   {result['e2e_latency_seconds']:.4f} s")
    print(f"Decode time:   {result['decode_time_seconds']:.4f} s")
    print(f"Chunks:        {result['num_stream_chunks']}")
    print(f"\nOutput:\n{result['output_text']}")
    
if __name__ == "__main__":
    main()