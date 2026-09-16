# Client for sending streaming completion requests to a vLLM server
from __future__ import annotations
import json
import time
from typing import Any
import httpx

class InferenceClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, model: str, prompt: str, max_tokens: int, temperature: float, stream: bool = True) -> dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }

        if stream:
            payload["stream_options"] = {
                "include_usage": True,
            }

        request_start = time.perf_counter()

        first_output_time: float | None = None
        response_text_parts: list[str] = []
        usage: dict[str, Any] | None = None

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", url, json=payload) as response:
                response.raise_for_status()

                if not stream:
                    data = response.json()
                    request_end = time.perf_counter()
                    choice = data["choices"][0]
                    message = choice.get("message", {})
                    response_text = message.get("content", "")
                    usage = data.get("usage")

                    return self._build_result(
                        request_start=request_start,
                        first_output_time=None,
                        request_end=request_end,
                        response_text=response_text,
                        usage=usage,
                    )

                for line in response.iter_lines():
                    if not line:
                        continue

                    if not line.startswith("data:"):
                        continue

                    data = line[len("data:"):].strip()

                    if data == "[DONE]":
                        break

                    chunk = json.loads(data)

                    # the final chunk may contain usage information
                    if chunk.get("usage") is not None:
                        usage = chunk["usage"]

                    choices = chunk.get("choices", [])

                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})

                    content = delta.get("content")

                    if content:
                        if first_output_time is None:
                            first_output_time = time.perf_counter()

                        response_text_parts.append(content)

                request_end = time.perf_counter()

        response_text = "".join(response_text_parts)

        return self._build_result(
            request_start=request_start,
            first_output_time=first_output_time,
            request_end=request_end,
            response_text=response_text,
            usage=usage,
        )

    @staticmethod
    def _build_result(
        request_start: float,
        first_output_time: float | None,
        request_end: float,
        response_text: str,
        usage: dict[str, Any] | None,
    ) -> dict[str, Any]:

        ttft_seconds: float | None = None
        decode_duration_seconds: float | None = None
        tpot_seconds: float | None = None
        output_tokens_per_second: float | None = None

        if first_output_time is not None:
            ttft_seconds = first_output_time - request_start
            decode_duration_seconds = request_end - first_output_time

        input_tokens = None
        output_tokens = None

        if usage is not None:
            input_tokens = usage.get("prompt_tokens")
            output_tokens = usage.get("completion_tokens")

        if (
            decode_duration_seconds is not None
            and output_tokens is not None
            and output_tokens > 1
        ):
            tpot_seconds = (decode_duration_seconds / (output_tokens - 1))

        if (
            decode_duration_seconds is not None
            and output_tokens is not None
            and output_tokens > 0
        ):
            output_tokens_per_second = (output_tokens / decode_duration_seconds)

        return {
            "response_text": response_text,
            "request_start": request_start,
            "first_output_time": first_output_time,
            "request_end": request_end,
            "ttft_seconds": ttft_seconds,
            "decode_duration_seconds": decode_duration_seconds,
            "e2e_latency_seconds": request_end - request_start,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tpot_seconds": tpot_seconds,
            "output_tokens_per_second": output_tokens_per_second,
        }
        