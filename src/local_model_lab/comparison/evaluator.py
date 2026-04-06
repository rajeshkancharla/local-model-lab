"""Phase 3 multi-model comparison harness.

Extends the Phase 1 benchmark runner with three additions:
  1. Memory measurement  — psutil captures Ollama's RSS delta after model load.
  2. Quality scoring     — heuristic 0–20 score per response (see quality.py).
  3. Category tracking  — prompt category and difficulty recorded per result.

ComparisonResult is a separate dataclass (not a subclass of BenchmarkResult)
so it can be serialised independently as JSONL.

Usage::

    results = await run_comparison(
        models=["llama3.2:3b", "phi3.5:3.8b", "gemma3:4b"],
        prompt_set="comparison_full",
        repeats=3,
    )
    print_comparison_summary(results)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import psutil
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from local_model_lab.benchmarks.metrics import capture_streaming_metrics
from local_model_lab.benchmarks.prompts import load_prompts
from local_model_lab.client import OllamaClient
from local_model_lab.comparison.quality import ZERO_SCORE, score_response
from local_model_lab.config import settings

console = Console()


@dataclass
class ComparisonResult:
    model: str
    prompt_id: str
    category: str
    difficulty: str
    temperature: float
    repeat: int                 # 0-indexed repeat number
    ttft_ms: float
    total_latency_ms: float
    tokens_generated: int
    tokens_per_second: float
    memory_mb: float            # Ollama process RSS delta after model load
    quality_relevance: int      # 0-5
    quality_completeness: int   # 0-5
    quality_format: int         # 0-5
    quality_coherence: int      # 0-5
    quality_total: int          # 0-20
    response_text: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_warmup: bool = False
    timed_out: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "ComparisonResult":
        return cls(**json.loads(line))


# ── I/O helpers ───────────────────────────────────────────────────────────────


def save_comparison_result(result: ComparisonResult, path: Path) -> None:
    """Append a single result as a JSONL line (crash-resilient)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(result.to_json() + "\n")


def load_comparison_results(path: Path) -> list[ComparisonResult]:
    """Load all ComparisonResult records from a JSONL file."""
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(ComparisonResult.from_json(line))
    return results


# ── Memory measurement ────────────────────────────────────────────────────────


def _measure_ollama_memory_mb() -> float:
    """Return the RSS of the Ollama server process in MB.

    Iterates psutil process list looking for a process whose name contains
    'ollama'. Returns 0.0 if no matching process is found or access is denied.
    """
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            name = proc.info["name"] or ""
            if "ollama" in name.lower():
                mem = proc.info["memory_info"]
                return mem.rss / (1024 * 1024) if mem else 0.0
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return 0.0


# ── Main comparison runner ────────────────────────────────────────────────────


async def run_comparison(
    models: list[str],
    prompt_set: str = "comparison_full",
    temperature: float = 0.0,
    repeats: int = 3,
    output_dir: Path | None = None,
) -> list[ComparisonResult]:
    """Run the Phase 3 comparison study.

    For each model:
      1. Measure Ollama RSS before warmup (no model loaded).
      2. Run a warmup inference (loads the model into Ollama's memory).
      3. Measure Ollama RSS after warmup → delta = model memory footprint.
      4. Run every prompt ``repeats`` times with quality scoring.
      5. Save each result incrementally to JSONL (crash-resilient).
      6. Unload the model before moving to the next one.

    Args:
        models:     Model tags to evaluate.
        prompt_set: Name of the YAML prompt file (without extension).
        temperature: Sampling temperature for all inferences.
        repeats:    Number of repetitions per prompt.
        output_dir: Where to write JSONL files. Defaults to data/results/.

    Returns:
        All ComparisonResult objects (including warmups, which are marked).
    """
    output_dir = output_dir or settings.results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    client = OllamaClient()
    if not await client.health_check():
        raise RuntimeError(
            "Ollama is not running. Start it from the system tray or run 'ollama serve', "
            "then retry."
        )

    prompts = load_prompts(prompt_set)
    all_results: list[ComparisonResult] = []
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    total_steps = len(models) * (1 + len(prompts) * repeats)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Comparing models", total=total_steps)

        for model in models:
            output_path = output_dir / f"comparison_{model.replace(':', '_')}_{ts}.jsonl"
            console.print(f"\n[bold green]Model: {model}[/bold green]")

            # Baseline: measure Ollama RSS before the model is loaded into memory.
            # Previous model should already be unloaded by this point.
            memory_before = _measure_ollama_memory_mb()

            # Warmup — also loads the model, so we can measure the RSS delta.
            progress.update(task, description=f"Warmup: {model}")
            warmup_bm = await capture_streaming_metrics(
                client, model, prompts[0].id, prompts[0].text,
                temperature=temperature, is_warmup=True,
            )
            progress.advance(task)

            memory_after = _measure_ollama_memory_mb()
            memory_delta = max(0.0, memory_after - memory_before)
            console.print(f"  Memory footprint: {memory_delta:.0f} MB")

            # Save warmup record (excluded from quality aggregates)
            warmup_cr = ComparisonResult(
                model=model,
                prompt_id=warmup_bm.prompt_id,
                category="warmup",
                difficulty="easy",
                temperature=temperature,
                repeat=0,
                ttft_ms=warmup_bm.ttft_ms,
                total_latency_ms=warmup_bm.total_latency_ms,
                tokens_generated=warmup_bm.tokens_generated,
                tokens_per_second=warmup_bm.tokens_per_second,
                memory_mb=memory_delta,
                quality_relevance=0,
                quality_completeness=0,
                quality_format=0,
                quality_coherence=0,
                quality_total=0,
                response_text=warmup_bm.response_text,
                is_warmup=True,
            )
            save_comparison_result(warmup_cr, output_path)

            # Main benchmark loop
            for prompt in prompts:
                for rep in range(repeats):
                    progress.update(
                        task,
                        description=f"{model} | {prompt.id} [{rep + 1}/{repeats}]",
                    )

                    bm = await capture_streaming_metrics(
                        client, model, prompt.id, prompt.text,
                        temperature=temperature,
                    )

                    timed_out = bm.response_text.startswith("[TIMED OUT")
                    qs = ZERO_SCORE if timed_out else score_response(prompt, bm.response_text)

                    result = ComparisonResult(
                        model=model,
                        prompt_id=prompt.id,
                        category=prompt.category,
                        difficulty=prompt.difficulty,
                        temperature=temperature,
                        repeat=rep,
                        ttft_ms=bm.ttft_ms,
                        total_latency_ms=bm.total_latency_ms,
                        tokens_generated=bm.tokens_generated,
                        tokens_per_second=bm.tokens_per_second,
                        memory_mb=memory_delta,
                        quality_relevance=qs.relevance,
                        quality_completeness=qs.completeness,
                        quality_format=qs.format_compliance,
                        quality_coherence=qs.coherence,
                        quality_total=qs.total,
                        response_text=bm.response_text,
                        timed_out=timed_out,
                    )
                    save_comparison_result(result, output_path)
                    all_results.append(result)
                    progress.advance(task)

                    if timed_out:
                        console.print(f"  [yellow]Timed out:[/yellow] {model} | {prompt.id}")

            # Unload before the next model to free RAM on constrained hardware.
            progress.update(task, description=f"Unloading {model}...")
            await client.unload_model(model)
            console.print(f"[dim]Unloaded {model}[/dim]")

    return all_results


# ── CLI summary table ─────────────────────────────────────────────────────────


def print_comparison_summary(results: list[ComparisonResult]) -> None:
    """Print a rich summary table of comparison results (excluding warmups)."""
    actual = [r for r in results if not r.is_warmup]
    if not actual:
        console.print("[yellow]No comparison results to summarise.[/yellow]")
        return

    by_model: dict[str, list[ComparisonResult]] = {}
    for r in actual:
        by_model.setdefault(r.model, []).append(r)

    table = Table(title="Phase 3 Comparison Summary", show_lines=True)
    table.add_column("Model", style="bold cyan")
    table.add_column("Prompts", justify="right")
    table.add_column("Avg T/s", justify="right")
    table.add_column("Avg TTFT (ms)", justify="right")
    table.add_column("Avg Latency (ms)", justify="right")
    table.add_column("Memory (MB)", justify="right")
    table.add_column("Avg Quality/20", justify="right")

    for model, model_results in by_model.items():
        valid = [r for r in model_results if not r.timed_out]
        n = len(valid)
        if n == 0:
            continue
        avg_tps = sum(r.tokens_per_second for r in valid) / n
        avg_ttft = sum(r.ttft_ms for r in valid) / n
        avg_latency = sum(r.total_latency_ms for r in valid) / n
        memory = model_results[0].memory_mb
        avg_quality = sum(r.quality_total for r in valid) / n
        timeouts = sum(1 for r in model_results if r.timed_out)
        count_str = str(n) + (f" [yellow]({timeouts} TO)[/yellow]" if timeouts else "")

        table.add_row(
            model,
            count_str,
            f"{avg_tps:.1f}",
            f"{avg_ttft:.0f}",
            f"{avg_latency:.0f}",
            f"{memory:.0f}",
            f"{avg_quality:.1f}",
        )

    console.print(table)
