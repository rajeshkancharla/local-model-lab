# local-model-lab

A portfolio-grade benchmarking and evaluation framework for running Small Language Models (SLMs) entirely offline using Ollama. Demonstrates practical engineering trade-offs of local inference: privacy, latency, cost, and edge deployment on constrained hardware.

---

## Project Overview

This project systematically evaluates four open-source SLMs from four companies on the same hardware under identical conditions — producing a data-driven technical report rather than following trends.

**Models evaluated:**
| Model | Company | Size |
|-------|---------|------|
| llama3.2:3b | Meta | 1.9 GB |
| phi3.5:3.8b | Microsoft | 2.0 GB |
| gemma3:4b | Google | 3.1 GB |
| qwen3:4b | Alibaba | 2.3 GB |

**Hardware:** Lenovo Yoga laptop, Intel Arc Graphics, CPU-only inference (Intel Arc not supported by Ollama on Windows — a real-world edge deployment constraint).

---

## Phases

### Phase 1: Inference Performance Benchmarking ✅
- Ollama HTTP client built with `httpx` (direct REST, not the ollama SDK — for timing precision)
- Tracks: tokens/sec, time to first token (TTFT), total latency
- Warmup runs excluded from stats; models explicitly unloaded between benchmarks
- Per-inference timeout and token cap to handle runaway generation
- Results saved incrementally as JSONL — crash resilient

**Phase 1 Results** (CPU-only, temperature=0.0, 8 prompts × 3 repeats):

| Model | Avg Tokens/s | Avg TTFT (ms) | Avg Latency (ms) | Avg Tokens |
|-------|-------------|---------------|------------------|------------|
| llama3.2:3b | 13.4 | 1,261 | 13,072 | 147 |
| phi3.5:3.8b | 9.8 | 1,467 | 32,814 | 301 |
| gemma3:4b | 10.6 | 1,647 | 23,066 | 216 |
| qwen3:4b | 10.2 | 4,023 | 36,434 | 353 |

Key findings:
- **llama3.2:3b wins every speed metric** — fastest tokens/sec, lowest TTFT, lowest latency
- Token counts identical across independent runs — confirms temperature=0 determinism
- qwen3:4b TTFT (4s) does not improve with warmup — model characteristic, not cold-start
- Thermal throttling observed under sustained CPU load (tokens/sec degrades ~10-15% over long runs)

### Phase 2: Structured Output and Determinism 🔜
- Pydantic schema enforcement (SentimentResult, EntityList, CodeReview, StructuredSummary)
- Retry mechanism with error feedback on validation failure
- Temperature variance study (0.0 → 1.0) documenting output consistency
- FastAPI wrapper: `/generate`, `/generate/structured`, `/models`, `/health`

### Phase 3: Model Comparison Study 🔜
- 30-50 standardised prompts across: factual, reasoning, summarisation, code generation, creative, structured output, multi-step
- Heuristic quality scoring (relevance, completeness, format compliance, coherence)
- Memory footprint measurement per model via psutil
- Generated Markdown technical report with full data tables and per-category breakdowns
- Quantized variant comparison (extra mile)

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Model orchestration | Ollama |
| HTTP client | httpx (direct REST) |
| Validation | Pydantic v2 |
| API | FastAPI + uvicorn |
| CLI | Typer + rich |
| Results storage | JSONL |
| Prompts | YAML |
| Memory measurement | psutil |
| Tests | pytest + pytest-asyncio |

---

## Setup

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) installed and running.

```bash
# Pull models
ollama pull llama3.2:3b
ollama pull phi3.5:3.8b
ollama pull gemma3:4b
ollama pull qwen3:4b

# Install project
pip install -e ".[dev]"

# Verify
lab health
lab models
```

---

## Usage

```bash
# Run benchmark (Phase 1)
lab benchmark --models llama3.2:3b,phi3.5:3.8b,gemma3:4b,qwen3:4b --prompts benchmark_quick --repeats 3

# Test structured output (Phase 2)
lab structured --model llama3.2:3b --schema sentiment --temperature 0.0

# Full model comparison (Phase 3)
lab compare --models all --prompts full

# Generate report
lab report --output reports/comparison.md

# Start API server
lab serve

# Check Ollama status
lab health
lab models
```

---

## Key Design Decisions

- **`httpx` over `ollama` SDK** — direct HTTP control gives precise timing and demonstrates understanding of the underlying REST protocol
- **JSONL for results** — append-friendly, crash-resilient; partial runs are never lost
- **Explicit model unloading** — `keep_alive=0` between models prevents memory contention on constrained hardware
- **Warmup excluded from stats** — first inference includes model-load time; standard benchmarking practice
- **Heuristic quality scoring** — no LLM-as-judge since the system runs offline; transparent methodology over circular self-evaluation
- **`pathlib.Path` throughout** — cross-platform path handling
