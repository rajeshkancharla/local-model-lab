"""Benchmark orchestrator: runs prompts against models with warmup and repeats."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from local_model_lab.benchmarks.metrics import BenchmarkResult, capture_streaming_metrics, save_result
from local_model_lab.benchmarks.prompts import Prompt, load_prompts
from local_model_lab.client import OllamaClient
from local_model_lab.config import settings

console = Console()


async def run_benchmark(
    models: list[str],
    prompt_set: str = "benchmark_quick",
    temperature: float = 0.0,
    repeats: int = 3,
    output_dir: Path | None = None,
) -> list[BenchmarkResult]:
    """Run a full benchmark suite across models and prompts.

    For each model:
      1. A warmup inference is run (first prompt, excluded from statistics).
      2. Each prompt is run ``repeats`` times.
      3. Results are saved incrementally as JSONL.
    """
    output_dir = output_dir or settings.results_dir
    prompts = load_prompts(prompt_set)
    client = OllamaClient()

    all_results: list[BenchmarkResult] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    total_inferences = len(models) * (1 + len(prompts) * repeats)  # warmup + actual

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Benchmarking", total=total_inferences)

        for model in models:
            output_path = output_dir / f"{model.replace(':', '_')}_{timestamp}.jsonl"
            console.print(f"\n[bold green]Model: {model}[/bold green]")

            # Warmup run — also loads the model into memory so we can verify GPU
            progress.update(task, description=f"Warmup: {model}")
            warmup = await capture_streaming_metrics(
                client, model, prompts[0].id, prompts[0].text,
                temperature=temperature, is_warmup=True,
            )
            save_result(warmup, output_path)
            all_results.append(warmup)
            progress.advance(task)

            # Verify GPU usage after model is loaded
            gpu_info = await client.verify_gpu(model)
            processor = gpu_info["processor"]
            vram = gpu_info["vram_gb"]
            if "gpu" in processor.lower():
                console.print(f"  [green]GPU active[/green]: {processor} | VRAM: {vram:.1f} GB")
            else:
                console.print(f"  [yellow]CPU only[/yellow]: {processor} (no GPU acceleration)")

            # Actual benchmark runs
            for prompt in prompts:
                for rep in range(repeats):
                    progress.update(
                        task,
                        description=f"{model} | {prompt.id} [{rep + 1}/{repeats}]",
                    )
                    result = await capture_streaming_metrics(
                        client, model, prompt.id, prompt.text,
                        temperature=temperature,
                    )
                    save_result(result, output_path)
                    all_results.append(result)
                    progress.advance(task)
                    if result.response_text.startswith("[TIMED OUT"):
                        console.print(f"  [yellow]Timed out:[/yellow] {model} | {prompt.id} (skipped)")

            # Explicitly unload the model before moving to the next one.
            # Ollama keeps models warm for 5 minutes by default; on memory-
            # constrained hardware this would block the next model from loading.
            progress.update(task, description=f"Unloading {model} from memory...")
            await client.unload_model(model)
            console.print(f"[dim]Unloaded {model}[/dim]")

    return all_results


def print_summary(results: list[BenchmarkResult]) -> None:
    """Print a rich summary table of benchmark results (excluding warmups)."""
    actual = [r for r in results if not r.is_warmup]
    if not actual:
        console.print("[yellow]No benchmark results to summarize.[/yellow]")
        return

    # Group by model
    models: dict[str, list[BenchmarkResult]] = {}
    for r in actual:
        models.setdefault(r.model, []).append(r)

    table = Table(title="Benchmark Summary", show_lines=True)
    table.add_column("Model", style="bold cyan")
    table.add_column("Prompts", justify="right")
    table.add_column("Avg Tokens/s", justify="right")
    table.add_column("Avg TTFT (ms)", justify="right")
    table.add_column("Avg Latency (ms)", justify="right")
    table.add_column("Avg Tokens", justify="right")

    for model, model_results in models.items():
        n = len(model_results)
        avg_tps = sum(r.tokens_per_second for r in model_results) / n
        avg_ttft = sum(r.ttft_ms for r in model_results) / n
        avg_latency = sum(r.total_latency_ms for r in model_results) / n
        avg_tokens = sum(r.tokens_generated for r in model_results) / n

        table.add_row(
            model,
            str(n),
            f"{avg_tps:.1f}",
            f"{avg_ttft:.0f}",
            f"{avg_latency:.0f}",
            f"{avg_tokens:.0f}",
        )

    console.print(table)
