"""Thin async wrapper around the Ollama REST API using httpx."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx

from local_model_lab.config import settings


@dataclass
class StreamChunk:
    """A single chunk from a streaming Ollama response."""

    text: str
    done: bool
    raw: dict = field(repr=False)


class OllamaClient:
    """Async HTTP client for the Ollama /api/generate endpoint."""

    def __init__(self, base_url: str | None = None, timeout: float = 300.0):
        self.base_url = (base_url or settings.ollama_base_url).rstrip("/")
        self.timeout = timeout

    async def generate(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream tokens from /api/generate. Yields StreamChunk objects.

        The final chunk (done=True) contains server-side timing metrics in its
        ``raw`` dict: eval_count, eval_duration, prompt_eval_duration, load_duration.
        """
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_gpu": settings.num_gpu,
                "num_predict": settings.max_tokens,
            },
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST", f"{self.base_url}/api/generate", json=payload
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    import json

                    data = json.loads(line)
                    yield StreamChunk(
                        text=data.get("response", ""),
                        done=data.get("done", False),
                        raw=data,
                    )

    async def generate_full(
        self,
        model: str,
        prompt: str,
        *,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> dict:
        """Non-streaming call to /api/generate. Returns the full response dict."""
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_gpu": settings.num_gpu,
                "num_predict": settings.max_tokens,
            },
        }
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def list_models(self) -> list[dict]:
        """Return the list of locally available models from /api/tags."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            return resp.json().get("models", [])

    async def verify_gpu(self, model: str) -> dict:
        """Query /api/ps to check whether a loaded model is using GPU.

        Returns a dict with keys:
          processor  — e.g. "100% GPU", "100% CPU", or "unknown"
          vram_gb    — VRAM used in GB (0.0 if CPU-only)
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/api/ps")
                resp.raise_for_status()
                models = resp.json().get("models", [])
                for m in models:
                    if model in m.get("name", ""):
                        processor = m.get("details", {}).get("processor", "unknown")
                        vram_gb = m.get("size_vram", 0) / 1e9
                        return {"processor": processor, "vram_gb": round(vram_gb, 2)}
        except httpx.HTTPError:
            pass
        return {"processor": "unknown", "vram_gb": 0.0}

    async def unload_model(self, model: str) -> None:
        """Explicitly unload a model from memory using keep_alive=0.

        Ollama keeps models warm for 5 minutes by default. On memory-constrained
        machines, calling this between benchmarks ensures the previous model is
        fully evicted before the next one loads.
        """
        payload = {"model": model, "keep_alive": 0, "prompt": "", "stream": False}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(f"{self.base_url}/api/generate", json=payload)
        except httpx.HTTPError:
            pass  # Best-effort; don't fail the benchmark run over this

    async def health_check(self) -> bool:
        """Return True if Ollama is reachable."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(self.base_url)
                return resp.status_code == 200
        except httpx.HTTPError:
            return False
