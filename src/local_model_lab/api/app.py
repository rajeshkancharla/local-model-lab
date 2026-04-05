"""FastAPI application for the Local Model Lab.

Endpoints:
    GET  /health               — Ollama reachability check
    GET  /models               — list locally available models
    POST /generate             — free-form text generation
    POST /generate/structured  — structured output with schema validation

Start with::

    lab serve
    # or directly:
    uvicorn local_model_lab.api.app:app --reload
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from local_model_lab.client import OllamaClient
from local_model_lab.config import settings
from local_model_lab.structured.enforcer import enforce_schema, parse_enforce_result
from local_model_lab.structured.schemas import SCHEMA_REGISTRY

app = FastAPI(
    title="Local Model Lab API",
    description="REST interface for benchmarking and structured-output experiments with local LLMs.",
    version="0.2.0",
)

_client = OllamaClient()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    model: str = Field(default="llama3.2:3b", description="Ollama model tag")
    prompt: str = Field(description="User prompt text")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    system: str | None = Field(default=None, description="Optional system prompt")


class GenerateResponse(BaseModel):
    model: str
    response: str
    tokens_generated: int
    tokens_per_second: float
    latency_ms: float


class StructuredRequest(BaseModel):
    model: str = Field(default="llama3.2:3b", description="Ollama model tag")
    prompt: str = Field(description="User prompt text (schema injected automatically)")
    schema_name: str = Field(
        description=f"Schema to enforce. One of: {list(SCHEMA_REGISTRY.keys())}"
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class StructuredResponse(BaseModel):
    model: str
    schema_name: str
    success: bool
    result: dict[str, Any] | None
    raw_response: str
    attempts: int
    latency_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Ollama health check")
async def health() -> dict:
    ok = await _client.health_check()
    if not ok:
        raise HTTPException(status_code=503, detail="Ollama is not reachable")
    return {"status": "ok", "ollama_url": settings.ollama_base_url}


@app.get("/models", summary="List available local models")
async def list_models() -> dict:
    try:
        model_list = await _client.list_models()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"models": [m.get("name") for m in model_list]}


@app.post("/generate", response_model=GenerateResponse, summary="Free-form generation")
async def generate(req: GenerateRequest) -> GenerateResponse:
    start = time.perf_counter()
    try:
        resp = await _client.generate_full(
            req.model,
            req.prompt,
            temperature=req.temperature,
            system=req.system,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    latency_ms = (time.perf_counter() - start) * 1000
    eval_count = resp.get("eval_count", 0)
    eval_duration_ns = resp.get("eval_duration", 0)
    tps = (eval_count / (eval_duration_ns / 1e9)) if eval_duration_ns > 0 else 0.0

    return GenerateResponse(
        model=req.model,
        response=resp.get("response", ""),
        tokens_generated=eval_count,
        tokens_per_second=round(tps, 2),
        latency_ms=round(latency_ms, 2),
    )


@app.post(
    "/generate/structured",
    response_model=StructuredResponse,
    summary="Structured output with schema enforcement",
)
async def generate_structured(req: StructuredRequest) -> StructuredResponse:
    if req.schema_name not in SCHEMA_REGISTRY:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown schema '{req.schema_name}'. Available: {list(SCHEMA_REGISTRY.keys())}",
        )

    schema_class = SCHEMA_REGISTRY[req.schema_name]
    try:
        enforce_result = await enforce_schema(
            _client, req.model, req.prompt, schema_class, temperature=req.temperature
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    parsed = parse_enforce_result(enforce_result)
    return StructuredResponse(
        model=req.model,
        schema_name=req.schema_name,
        **parsed,
    )
