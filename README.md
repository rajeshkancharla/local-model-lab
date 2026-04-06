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

**Results** (CPU-only, temperature=0.0, 8 prompts × 3 repeats):

| Model | Avg Tokens/s | Avg TTFT (ms) | Avg Latency (ms) |
|-------|-------------:|-------------:|-----------------:|
| llama3.2:3b | 13.4 | 1,261 | 13,072 |
| phi3.5:3.8b | 9.8 | 1,467 | 32,814 |
| gemma3:4b | 10.6 | 1,647 | 23,066 |
| qwen3:4b | 10.2 | 4,023 | 36,434 |

Key findings:
- **llama3.2:3b wins every speed metric** — fastest tokens/sec, lowest TTFT, lowest latency
- Token counts identical across independent runs — confirms temperature=0 determinism
- qwen3:4b TTFT (4s) does not improve with warmup — model characteristic, not cold-start
- Thermal throttling observed under sustained CPU load (tokens/sec degrades ~10-15% over long runs)

---

### Phase 2: Structured Output and Determinism ✅

- Pydantic schema enforcement for four schemas: `SentimentResult`, `EntityList`, `CodeReview`, `StructuredSummary`
- JSON extraction handles code fences, inline comments (phi3.5), and embedded prose
- Retry mechanism with validation error feedback — model self-corrects on second attempt
- Temperature variance study (0.0, 0.3, 0.7, 1.0 × 5 repeats) documenting output consistency
- FastAPI wrapper: `POST /generate`, `POST /generate/structured`, `GET /models`, `GET /health`

**Structured output success rates** (all temperatures, 5 repeats per cell):

| Model | sentiment | entities | code_review | summary |
|-------|:---------:|:--------:|:-----------:|:-------:|
| gemma3:4b | 100% | 100% | 100% | 100% |
| phi3.5:3.8b | 100% | 100% | 100% | 100% |
| llama3.2:3b | 100% | 100% | 95%* | 100% |
| qwen3:4b | 100% | 0% | partial | 0% |

\* llama degrades to 80% at temperature=0.7; 100% at 0.0

Key findings:
- **gemma3:4b is the most reliable structured output model** — zero retries needed
- Temperature 0.0 is essential for production structured output use
- qwen3:4b is unsuitable — verbose generation exhausts token budget before completing JSON
- Small models require plain-English field descriptions in the prompt, not JSON Schema definitions

---

### Phase 3: Model Comparison Study ✅

- 40 standardised prompts across 7 categories: factual recall, reasoning, summarization, code generation, creative writing, multi-step instructions, structured output
- Heuristic quality scoring: 4 dimensions × 0–5 = 20 point scale (no LLM-as-judge — offline constraint)
- Memory footprint measurement per model via psutil RSS delta
- Full technical report: [reports/comparison.md](reports/comparison.md)

**Results** (CPU-only, temperature=0.0, 40 prompts × 3 repeats = 120 inferences per model):

| Model | Avg T/s | TTFT (ms) | Latency (ms) | Memory (MB) | Quality/20 |
|-------|--------:|----------:|-------------:|------------:|-----------:|
| llama3.2:3b | **13.1** | 6,467 | **26,638** | 30 | **16.8** |
| phi3.5:3.8b | 10.4 | **5,739** | 37,428 | ~2,000* | 16.8 |
| gemma3:4b | 10.4 | 7,156 | 37,640 | 36 | 16.2 |

\* phi3.5 RSS delta was 0 MB due to OS memory page reuse after llama unload; actual disk size is 2.0 GB

**Quality by category:**

| Category | gemma3:4b | llama3.2:3b | phi3.5:3.8b |
|----------|----------:|------------:|------------:|
| Factual Recall | 16.5 | 16.3 | 16.0 |
| Reasoning | 17.2 | 17.5 | 17.3 |
| Summarization | 16.2 | **17.8** | 17.2 |
| Code Generation | 16.5 | 16.2 | **18.0** |
| Creative Writing | 15.5 | 15.8 | 16.0 |
| Multi-step | 15.3 | 16.5 | 16.0 |
| Structured Output | 16.3 | 17.1 | 16.6 |

**Recommendation by use case:**

| Use case | Model |
|----------|-------|
| Latency-sensitive / real-time | `llama3.2:3b` |
| General quality | `llama3.2:3b` |
| Code generation | `phi3.5:3.8b` |
| Structured JSON output | `gemma3:4b` |

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
# Phase 1 — inference benchmarks
lab benchmark --models llama3.2:3b,phi3.5:3.8b,gemma3:4b,qwen3:4b --prompts benchmark_quick --repeats 3

# Phase 2 — structured output temperature sweep
lab structured --models llama3.2:3b,phi3.5:3.8b,gemma3:4b --schemas sentiment,entities,code_review,summary --temperatures 0.0,0.3,0.7,1.0 --repeats 5

# Phase 2 — start FastAPI server (http://localhost:8000/docs)
lab serve

# Phase 3 — full model comparison (~2-3 hours on CPU, keep laptop unlocked)
lab compare --models llama3.2:3b,phi3.5:3.8b,gemma3:4b --prompts comparison_full --repeats 3

# Phase 3 — generate Markdown report from latest results
lab report

# Utilities
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
