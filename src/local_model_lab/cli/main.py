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


@app.command()
def structured(
    models: str = typer.Option(
        ",".join(settings.models),
        "--models", "-m",
        help="Comma-separated model tags to test.",
    ),
    schemas: str = typer.Option(
        "sentiment,entities,code_review,summary",
        "--schemas", "-s",
        help="Comma-separated schema names to test.",
    ),
    temperatures: str = typer.Option(
        "0.0,0.3,0.7,1.0",
        "--temperatures", "-t",
        help="Comma-separated temperature values to sweep.",
    ),
    repeats: int = typer.Option(
        5,
        "--repeats", "-r",
        help="Repeats per (model, schema, temperature) cell.",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir", "-o",
        help="Directory for result JSONL files (default: data/results/).",
    ),
):
    """Run structured output temperature variance experiments."""
    from local_model_lab.structured.experiments import (
        run_temperature_experiment,
        print_experiment_summary,
    )

    model_list = [m.strip() for m in models.split(",")]
    schema_list = [s.strip() for s in schemas.split(",")]
    temp_list = [float(t.strip()) for t in temperatures.split(",")]

    console.print(f"[bold]Models:[/bold] {model_list}")
    console.print(f"[bold]Schemas:[/bold] {schema_list}")
    console.print(f"[bold]Temperatures:[/bold] {temp_list}")
    console.print(f"[bold]Repeats per cell:[/bold] {repeats}\n")

    results = asyncio.run(
        run_temperature_experiment(
            models=model_list,
            schema_names=schema_list,
            temperatures=temp_list,
            repeats=repeats,
            output_dir=output_dir,
        )
    )
    print_experiment_summary(results)


@app.command()
def compare(
    models: str = typer.Option(
        "llama3.2:3b,phi3.5:3.8b,gemma3:4b",
        "--models", "-m",
        help="Comma-separated model tags to compare (qwen3:4b excluded by default — too slow for structured output).",
    ),
    prompts: str = typer.Option(
        "comparison_full",
        "--prompts", "-p",
        help="Prompt set name (YAML file in data/prompts/ without extension).",
    ),
    repeats: int = typer.Option(
        3,
        "--repeats", "-r",
        help="Number of repeats per prompt.",
    ),
    temperature: float = typer.Option(
        0.0,
        "--temperature", "-t",
        help="Sampling temperature (0.0 = deterministic).",
    ),
    output_dir: Path = typer.Option(
        None,
        "--output-dir", "-o",
        help="Directory for result JSONL files (default: data/results/).",
    ),
):
    """Run the Phase 3 model comparison study.

    Evaluates each model across all prompts with quality scoring and memory
    measurement. Results are saved incrementally as JSONL files.

    Estimated run time on CPU: ~2-3 hours for 3 models × 40 prompts × 3 repeats.
    Do not lock the laptop screen — Windows throttles CPU and causes timeouts.
    """
    from local_model_lab.comparison.evaluator import run_comparison, print_comparison_summary

    model_list = [m.strip() for m in models.split(",")]
    console.print(f"[bold]Models:[/bold] {model_list}")
    console.print(f"[bold]Prompt set:[/bold] {prompts}")
    console.print(f"[bold]Repeats:[/bold] {repeats}")
    console.print(f"[bold]Temperature:[/bold] {temperature}")
    total = len(model_list) * 40 * repeats  # approx
    console.print(f"[bold yellow]Estimated inferences:[/bold yellow] ~{total} "
                  "(keep laptop unlocked to prevent CPU throttling)\n")

    results = asyncio.run(
        run_comparison(
            models=model_list,
            prompt_set=prompts,
            temperature=temperature,
            repeats=repeats,
            output_dir=output_dir,
        )
    )
    print_comparison_summary(results)


@app.command()
def report(
    output: Path = typer.Option(
        None,
        "--output", "-o",
        help="Output path for the Markdown report (default: reports/comparison.md).",
    ),
    results_dir: Path = typer.Option(
        None,
        "--results-dir",
        help="Directory containing comparison_*.jsonl files (default: data/results/).",
    ),
):
    """Generate the Phase 3 comparison Markdown report.

    Automatically discovers the latest comparison_*.jsonl file per model and
    the latest structured_experiment_*.jsonl (Phase 2 data) in data/results/.
    Writes reports/comparison.md.
    """
    from local_model_lab.comparison.report import (
        find_latest_comparison_files,
        find_latest_phase2_file,
        generate_report,
    )

    search_dir = results_dir or settings.results_dir
    comparison_files = find_latest_comparison_files(search_dir)

    if not comparison_files:
        console.print(
            "[bold red]No comparison_*.jsonl files found.[/bold red]\n"
            "Run [bold]lab compare[/bold] first to generate data."
        )
        raise typer.Exit(code=1)

    phase2_file = find_latest_phase2_file(search_dir)
    if phase2_file:
        console.print(f"[dim]Phase 2 data: {phase2_file.name}[/dim]")
    else:
        console.print("[dim]No Phase 2 experiment data found — structured output section will be omitted.[/dim]")

    console.print(f"[dim]Comparison files: {[f.name for f in comparison_files]}[/dim]\n")

    md = generate_report(
        comparison_files=comparison_files,
        phase2_file=phase2_file,
        output_path=output,
    )

    out_path = output or (settings.reports_dir / "comparison.md")
    console.print(f"[bold green]Report written to:[/bold green] {out_path}")
    console.print(f"[dim]({len(md.splitlines())} lines, {len(md):,} chars)[/dim]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host."),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development."),
):
    """Start the FastAPI server (local_model_lab.api.app)."""
    import uvicorn

    console.print(f"[bold green]Starting API server at http://{host}:{port}[/bold green]")
    console.print(f"[dim]Docs: http://{host}:{port}/docs[/dim]\n")
    uvicorn.run(
        "local_model_lab.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
