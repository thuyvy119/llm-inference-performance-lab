from __future__ import annotations
import asyncio
from client.request import AsyncInferenceClient, InferenceClient

ARGS = dict(model="test-model", prompt="Explain GPUs", max_tokens=8, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, enable_thinking=False, stream=True, ignore_eos=True)

def assert_complete_streaming_result(result):
    assert result["response_text"] == "GPU is faster"
    assert result["ttft_seconds"] > 0
    assert result["decode_duration_seconds"] > 0
    assert result["e2e_latency_seconds"] >= result["ttft_seconds"]
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 3
    assert result["tpot_seconds"] is not None
    
def test_sync_streaming_result_and_payload(mock_server):
    state, base_url = mock_server
    with InferenceClient(base_url) as client:
        result = client.generate(**ARGS, ignore_eos=True)

    assert_complete_streaming_result(result)
    payload = state.payload[0]
    assert payload["messages"] == [{"role": "user", "content": "Explain GPUs"}]
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["ignore_eos"] is True
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}

def test_sync_streaming_result_and_payload(mock_server):
    state, base_url = mock_server
    with InferenceClient(base_url) as client:
        result = client.generate(**ARGS, ignore_eos=True)

    assert_complete_streaming_result(result)
    payload = state.payload[0]
    assert payload["messages"] == [{"role": "user", "content": "Explain GPUs"}]
    assert payload["stream"] is True
    assert payload["stream_options"] == {"include_usage": True}
    assert payload["ignore_eos"] is True
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}

def test_ignore_eos_omitted_when_false(mock_server):
    state, base_url = mock_server
    with InferenceClient(base_url) as client:
        result = client.generate(**ARGS)
    
    assert "ignore_eos" not in state.payload[0]

def test_non_streamin_path(mock_server):
    state, base_url = mock_server
    with InferenceClient(base_url) as client:
        result = client.generate(**ARGS, stream=False)
    
    assert result["ttft_seconds"] is None
    assert result["output_tokens"] == 3
    assert "stream_options" not in state.payload[0]
    
def test_async_client_matches_sync(mock_server):
    state, base_url = mock_server
    async def go():
        async with AsyncInferenceClient(base_url) as client:
            return await client.generate(**ARGS, ignore_eos=True)
    
    assert_complete_streaming_result(asyncio.run(go()))
    assert state.payload[0]["stream_options"] == {"include_usage": True}
    