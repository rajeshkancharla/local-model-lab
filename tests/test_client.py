"""Tests for the Ollama HTTP client."""

import pytest

from local_model_lab.client import OllamaClient


@pytest.mark.integration
@pytest.mark.asyncio
async def test_health_check():
    client = OllamaClient()
    result = await client.health_check()
    assert result is True, "Ollama must be running for integration tests"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_models():
    client = OllamaClient()
    models = await client.list_models()
    assert isinstance(models, list)
    assert len(models) > 0, "At least one model should be available"
    assert "name" in models[0]


TEST_MODEL = "qwen3:4b"  # already pulled; swap to any available model


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generate_streaming():
    client = OllamaClient()
    chunks = []
    async for chunk in client.generate(TEST_MODEL, "Say hello in one word.", temperature=0.0):
        chunks.append(chunk)

    assert len(chunks) > 0, "Should receive at least one chunk"
    assert chunks[-1].done is True, "Last chunk should have done=True"

    # Final chunk should contain server-side metrics
    final = chunks[-1].raw
    assert "eval_count" in final
    assert "eval_duration" in final
    assert final["eval_count"] > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generate_full():
    client = OllamaClient()
    result = await client.generate_full(TEST_MODEL, "Say hello in one word.", temperature=0.0)

    assert "response" in result
    assert len(result["response"]) > 0
    assert result["eval_count"] > 0
    assert result["eval_duration"] > 0
