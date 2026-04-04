"""Benchmark result data model and streaming metrics capture."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from local_model_lab.client import OllamaClient
from local_model_lab.config import settings


@dataclass
class BenchmarkResult:
    model: str
    prompt_id: str
    prompt_text: str
    temperature: float

    # Client-side timing
    ttft_ms: float  # time to first token
    total_latency_ms: float  # wall-clock end-to-end

    # Server-side metrics (from Ollama response)
    tokens_generated: int
    eval_duration_ms: float
    tokens_per_second: float
    prompt_eval_ms: float
    load_duration_ms: float

    response_text: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_warmup: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> BenchmarkResult:
        return cls(**json.loads(line))


def _prepare_prompt(model: str, prompt_text: str) -> str:
    """Apply model-specific prompt adjustments.

    Qwen3 models default to 'thinking' mode which generates a lengthy internal
    chain-of-thought before answering — can add minutes per prompt. /no_think
    disables this, giving direct answers comparable to other models.
    """
    if "qwen3" in model.lower():
        return prompt_text + " /no_think"
    return prompt_text


async def _stream_inference(
    client: OllamaClient,
    model: str,
    prompt: str,
    temperature: float,
) -> tuple[list[str], dict, float | None]:
    """Inner coroutine that streams one inference. Wrapped by timeout below."""
    response_parts: list[str] = []
    final_raw: dict = {}
    first_token_time: float | None = None

    async for chunk in client.generate(model, prompt, temperature=temperature):
        if first_token_time is None and chunk.text:
            first_token_time = time.perf_counter()
        response_parts.append(chunk.text)
        if chunk.done:
            final_raw = chunk.raw

    return response_parts, final_raw, first_token_time


async def capture_streaming_metrics(
    client: OllamaClient,
    model: str,
    prompt_id: str,
    prompt_text: str,
    temperature: float = 0.0,
    is_warmup: bool = False,
) -> BenchmarkResult:
    """Run a single streaming inference and capture all timing metrics.

    If the inference exceeds ``settings.inference_timeout_s``, it is cancelled
    and a result with timed_out=True and zero token counts is returned so the
    benchmark run continues rather than hanging indefinitely.
    """
    actual_prompt = _prepare_prompt(model, prompt_text)
    start = time.perf_counter()
    timed_out = False

    try:
        response_parts, final_raw, first_token_time = await asyncio.wait_for(
            _stream_inference(client, model, actual_prompt, temperature),
            timeout=settings.inference_timeout_s,
        )
    except asyncio.TimeoutError:
        timed_out = True
        response_parts, final_raw, first_token_time = [], {}, None

    end = time.perf_counter()

    # Extract server-side metrics (nanoseconds -> milliseconds)
    eval_count = final_raw.get("eval_count", 0)
    eval_duration_ns = final_raw.get("eval_duration", 0)
    prompt_eval_ns = final_raw.get("prompt_eval_duration", 0)
    load_duration_ns = final_raw.get("load_duration", 0)

    eval_duration_ms = eval_duration_ns / 1_000_000
    tokens_per_second = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0.0

    ttft_ms = ((first_token_time - start) * 1000) if first_token_time else 0.0
    total_latency_ms = (end - start) * 1000

    response_text = "".join(response_parts)
    if timed_out:
        response_text = f"[TIMED OUT after {settings.inference_timeout_s}s]"

    return BenchmarkResult(
        model=model,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        temperature=temperature,
        ttft_ms=round(ttft_ms, 2),
        total_latency_ms=round(total_latency_ms, 2),
        tokens_generated=eval_count,
        eval_duration_ms=round(eval_duration_ms, 2),
        tokens_per_second=round(tokens_per_second, 2),
        prompt_eval_ms=round(prompt_eval_ns / 1_000_000, 2),
        load_duration_ms=round(load_duration_ns / 1_000_000, 2),
        response_text=response_text,
        is_warmup=is_warmup,
    )


def save_result(result: BenchmarkResult, output_path: Path) -> None:
    """Append a single result as a JSONL line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(result.to_json() + "\n")


def load_results(path: Path) -> list[BenchmarkResult]:
    """Load all results from a JSONL file."""
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(BenchmarkResult.from_json(line))
    return results
