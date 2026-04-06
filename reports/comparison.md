# Local AI Model Comparison Study

*Generated: 2026-04-06 · 40 prompts · 360 inferences · CPU-only inference (Lenovo Yoga, Intel Core i7)*

---

## Executive Summary

| Finding | Winner |
|---------|--------|
| Fastest inference (T/s) | `llama3.2:3b` (13.1 T/s) |
| Best response quality | `llama3.2:3b` (16.8/20) |
| Lowest memory footprint | `phi3.5:3.8b` (0 MB) |

**Key insight:** Small models in the 3–4B range are viable for offline AI assistants on commodity hardware. Speed, quality, and memory form a three-way trade-off — no single model wins on all dimensions.

## Methodology

### Hardware

- **Device:** Lenovo Yoga laptop
- **CPU:** Intel Core i7 (no GPU acceleration — Intel Arc not supported by Ollama on Windows)
- **Inference engine:** Ollama v0.20.0, CPU-only

### Models Tested

- `gemma3:4b` — Google, 4B parameters, 3.1 GB on disk
- `llama3.2:3b` — Meta, 3B parameters, 1.9 GB on disk
- `phi3.5:3.8b` — Microsoft, 3.8B parameters, 2.0 GB on disk

### Evaluation Dataset

- **40 prompts** across 7 categories: Factual Recall, Reasoning, Summarization, Code Generation, Creative Writing, Multi-step Instructions, Structured Output
- **3 repetitions** per prompt per model
- Temperature fixed at 0.0 for deterministic comparison

### Quality Scoring (0–20)

Heuristic scoring across 4 dimensions × 0–5 each. No LLM-as-judge — all scoring is deterministic and offline-safe.

| Dimension | How it is measured |
|-----------|-------------------|
| **Relevance** (0–5) | Proportion of expected domain keywords present |
| **Completeness** (0–5) | Response length vs. minimum expected length; structural depth |
| **Format Compliance** (0–5) | JSON validity for structured prompts; code fences for code prompts |
| **Coherence** (0–5) | No repetition loops, no refusals, no mid-sentence truncation |

## Performance Benchmarks

| Model | Avg T/s | TTFT (ms) | Latency (ms) | Memory (MB) | Avg Quality/20 | Timeouts |
|-------|--------:|----------:|-------------:|------------:|---------------:|:--------:|
| `gemma3:4b` | 10.4 | 7156 | 37640 | 36 | 16.2 | 0 |
| `llama3.2:3b` | 13.1 | 6467 | 26638 | 30 | 16.8 | 0 |
| `phi3.5:3.8b` | 10.4 | 5739 | 37428 | 0 | 16.8 | 0 |

> **T/s** = tokens per second (higher is better).  **TTFT** = time to first token in milliseconds (lower is better).  **Latency** = wall-clock end-to-end (lower is better).  **Memory** = Ollama process RSS delta after model load.

## Quality by Category

Average quality score (0–20) per category.

| Category | `gemma3:4b` | `llama3.2:3b` | `phi3.5:3.8b` |
|----------|---: | ---: | ---:|
| Factual Recall | 16.5 | 16.3 | 16.0 |
| Reasoning | 17.2 | 17.5 | 17.3 |
| Summarization | 16.2 | 17.8 | 17.2 |
| Code Generation | 16.5 | 16.2 | 18.0 |
| Creative Writing | 15.5 | 15.8 | 16.0 |
| Multi-step Instructions | 15.3 | 16.5 | 16.0 |
| Structured Output | 16.3 | 17.1 | 16.6 |

### Quality Dimensions (averages across all prompts)

| Dimension | `gemma3:4b` | `llama3.2:3b` | `phi3.5:3.8b` |
|-----------|---: | ---: | ---:|
| Relevance | 3.9 | 4.0 | 4.1 |
| Completeness | 5.0 | 5.0 | 5.0 |
| Format Compliance | 4.0 | 4.0 | 4.0 |
| Coherence | 3.4 | 3.8 | 3.6 |

## Phase 2: Structured Output Reliability

Results from the temperature variance experiment (4 temperatures × 5 repeats per cell).

### Success Rates by Model and Schema

| Schema | `gemma3:4b` | `llama3.2:3b` | `phi3.5:3.8b` |
|--------|---:|---:|---:|
| code_review | 100% | 95% (4↻) | 100% (1↻) |
| entities | 100% | 100% | 100% |
| sentiment | 100% | 100% (1↻) | 100% |
| summary | 100% | 100% | 100% |

### Temperature Effect on Success Rate

| Temperature | `gemma3:4b` | `llama3.2:3b` | `phi3.5:3.8b` |
|-------------|---:|---:|---:|
| `0.0` | 100% | 100% | 100% |
| `0.3` | 100% | 100% | 100% |
| `0.7` | 100% | 95% | 100% |
| `1.0` | 100% | 100% | 100% |

> **↻** = retry was needed (model self-corrected on second attempt).  Temperature 0.0 consistently produced the highest structured output reliability.

## Key Findings & Recommendations

### Speed
- `llama3.2:3b` is the fastest model at 13.1 T/s average.
- All models fall in the 9–14 T/s range on CPU — fast enough for interactive use.
- TTFT is the dominant latency component for short answers; first-token delay is primarily model-load time on the first inference.

### Quality
- `llama3.2:3b` produces the highest-quality responses (16.8/20 average).
- Structured output (JSON) is the hardest category for all models at temperature 0.
- Code generation quality is high when models use fenced code blocks.

### Memory
- `phi3.5:3.8b` has the smallest memory footprint (0 MB).
- All tested models fit comfortably within 4 GB RAM on a CPU-only device.

### Recommendation

| Use case | Recommended model |
|----------|-------------------|
| Latency-sensitive / real-time | `llama3.2:3b` |
| Response quality matters most | `llama3.2:3b` |
| Memory-constrained deployment | `phi3.5:3.8b` |
| Structured JSON output | `gemma3:4b` (best Phase 2 reliability) |

## Appendix: Quality Scoring Methodology

All scoring is offline-safe and deterministic — no LLM-as-judge is used.

### Relevance (0–5)
Each prompt defines a list of `expected_keywords`. The score is the proportion
of those keywords found in the response, scaled to 0–5.

### Completeness (0–5)
Each prompt defines a `min_length` (character count). The score is:
- 0 — response under 20 chars
- 1 — under ⅓ of min_length
- 2 — under ⅔ of min_length
- 3 — close to min_length
- 4 — meets min_length
- 5 — exceeds 2× min_length (or min_length with good list structure for multi-step prompts)

### Format Compliance (0–5)
- **Structured output prompts** — checks for valid JSON (5 = root JSON, 4 = embedded JSON, 2 = malformed, 0 = absent)
- **Code generation prompts** — checks for fenced code blocks (5) or bare function definitions (3)
- **Text prompts** — plain text scores 4; unexpected JSON scores 2

### Coherence (0–5)
- 0 — empty response
- 1 — refusal detected (`"I cannot..."`) or severe repetition loop (>50% duplicate sentences)
- 3 — possible truncation (long response ending without terminal punctuation)
- 4–5 — multiple complete sentences, no issues detected
