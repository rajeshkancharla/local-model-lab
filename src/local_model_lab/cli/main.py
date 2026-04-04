"""CLI entry point for local-model-lab."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from local_model_lab.config import settings

app = typer.Typer(
    name="lab",
    help="Local Model Lab - benchmark and evaluate local LLMs via Ollama.",
)
console = Console()


@app.command()
def benchmark(
    models: str = typer.Option(
        ",".join(settings.models),
        "--models", "-m",
        help="Comma-separated model tags to benchmark.",
    ),
    prompts: str = typer.Option(
        "benchmark_quick",
        "--prompts", "-p",
        help="Prompt set name (YAML file in data/prompts/ without extension).",
    ),
    repeats: int = typer.Option(
        settings.benchmark_repeats,
        "--repeats", "-r",
        help="Number of repeats per prompt.",
    ),
    temperature: float = typer.Option(
        settings.default_temperature,
        "--temperature", "-t",
        help="Sampling temperature.",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir", "-o",
        help="Directory for result JSONL files (default: data/results/).",
    ),
    restart_server: bool = typer.Option(
        False,
        "--restart-server",
        help=(
            "Kill existing Ollama and restart it with OLLAMA_VULKAN=1 and "
            "OLLAMA_INTEL_GPU=1 for Intel Arc GPU acceleration. "
            "Has no effect if Ollama already started with these flags."
        ),
    ),
):
    """Run inference benchmarks against one or more local models."""
    from local_model_lab.benchmarks.runner import run_benchmark, print_summary

    model_list = [m.strip() for m in models.split(",")]
    console.print(f"[bold]Models:[/bold] {model_list}")
    console.print(f"[bold]Prompt set:[/bold] {prompts}")
    console.print(f"[bold]Repeats:[/bold] {repeats}")
    console.print(f"[bold]Temperature:[/bold] {temperature}\n")

    server_proc = None
    if restart_server:
        from local_model_lab.server import start_with_gpu, stop_server
        console.print("[bold yellow]Restarting Ollama with GPU flags (Vulkan + Intel GPU)...[/bold yellow]")
        server_proc = start_with_gpu()
        console.print("[bold green]Ollama server ready.[/bold green]\n")

    try:
        results = asyncio.run(
            run_benchmark(
                models=model_list,
                prompt_set=prompts,
                temperature=temperature,
                repeats=repeats,
                output_dir=output_dir,
            )
        )
    finally:
        if server_proc is not None:
            from local_model_lab.server import stop_server
            stop_server(server_proc)
            console.print("\n[dim]Ollama server stopped.[/dim]")

    print_summary(results)


@app.command()
def models():
    """List models available in the local Ollama instance."""
    from local_model_lab.client import OllamaClient
    from rich.table import Table

    async def _list():
        client = OllamaClient()
        return await client.list_models()

    model_list = asyncio.run(_list())

    table = Table(title="Available Ollama Models")
    table.add_column("Name", style="bold cyan")
    table.add_column("Size", justify="right")
    table.add_column("Modified")

    for m in model_list:
        size_gb = m.get("size", 0) / (1024**3)
        table.add_row(
            m.get("name", "?"),
            f"{size_gb:.1f} GB",
            m.get("modified_at", "?")[:19],
        )

    console.print(table)


@app.command()
def health():
    """Check if Ollama is running and reachable."""
    from local_model_lab.client import OllamaClient

    async def _check():
        client = OllamaClient()
        return await client.health_check()

    ok = asyncio.run(_check())
    if ok:
        console.print("[bold green]Ollama is running.[/bold green]")
    else:
        console.print("[bold red]Ollama is not reachable.[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
