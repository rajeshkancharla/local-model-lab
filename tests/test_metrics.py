"""Tests for benchmark metrics and capture."""

import json

import pytest

from local_model_lab.benchmarks.metrics import BenchmarkResult, capture_streaming_metrics
from local_model_lab.client import OllamaClient


def test_benchmark_result_serialization():
    result = BenchmarkResult(
        model="test-model",
        prompt_id="test_01",
        prompt_text="Hello",
        temperature=0.0,
        ttft_ms=50.0,
        total_latency_ms=1000.0,
        tokens_generated=20,
        eval_duration_ms=800.0,
        tokens_per_second=25.0,
        prompt_eval_ms=150.0,
        load_duration_ms=50.0,
        response_text="Hello there!",
    )

    json_str = result.to_json()
    parsed = json.loads(json_str)

    assert parsed["model"] == "test-model"
    assert parsed["tokens_per_second"] == 25.0
    assert parsed["is_warmup"] is False

    # Round-trip
    restored = BenchmarkResult.from_json(json_str)
    assert restored.model == result.model
    assert restored.tokens_generated == result.tokens_generated


TEST_MODEL = "qwen3:4b"  # already pulled; swap to any available model


@pytest.mark.integration
@pytest.mark.asyncio
async def test_capture_streaming_metrics():
    client = OllamaClient()
    result = await capture_streaming_metrics(
        client,
        model=TEST_MODEL,
        prompt_id="test_quick",
        prompt_text="What is 2+2? Answer with just the number.",
        temperature=0.0,
    )

    assert result.model == TEST_MODEL
    assert result.tokens_generated > 0
    assert result.tokens_per_second > 0
    assert result.ttft_ms > 0
    assert result.total_latency_ms > result.ttft_ms
    assert len(result.response_text) > 0
    assert result.is_warmup is False
