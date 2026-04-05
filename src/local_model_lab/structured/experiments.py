"""Temperature variance study for structured output reliability.

For each (model, schema, temperature) combination, runs N repeats and records:
- Whether the response validated successfully
- Raw response text
- Latency

Results are saved as JSONL and a summary table is printed via rich.

Usage::

    results = await run_temperature_experiment(
        models=["llama3.2:3b", "gemma3:4b"],
        temperatures=[0.0, 0.3, 0.7, 1.0],
        repeats=5,
    )
    print_experiment_summary(results)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from local_model_lab.client import OllamaClient
from local_model_lab.config import settings
from local_model_lab.structured.enforcer import enforce_schema
from local_model_lab.structured.schemas import SCHEMA_REGISTRY

console = Console()


@dataclass
class ExperimentResult:
    model: str
    schema_name: str
    temperature: float
    repeat: int
    success: bool
    attempts: int           # 1 = passed first try, 2 = needed retry
    latency_ms: float
    raw_response: str
    parsed_result: dict | None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "ExperimentResult":
        return cls(**json.loads(line))


def _save_result(result: ExperimentResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(result.to_json() + "\n")


async def run_temperature_experiment(
    models: list[str] | None = None,
    schema_names: list[str] | None = None,
    temperatures: list[float] | None = None,
    repeats: int = 5,
    output_dir: Path | None = None,
) -> list[ExperimentResult]:
    """Run the temperature variance experiment.

    For every (model × schema × temperature) combination, calls enforce_schema
    ``repeats`` times and records success/failure + latency.

    Args:
        models: Model tags to test. Defaults to settings.models.
        schema_names: Keys from SCHEMA_REGISTRY to test. Defaults to all four.
        temperatures: Temperature values to sweep. Defaults to [0.0, 0.3, 0.7, 1.0].
        repeats: Number of times to repeat each (model, schema, temperature) cell.
        output_dir: Where to write JSONL results. Defaults to data/results/.
    """
    models = models or settings.models
    schema_names = schema_names or list(SCHEMA_REGISTRY.keys())
    temperatures = temperatures if temperatures is not None else [0.0, 0.3, 0.7, 1.0]
    output_dir = output_dir or settings.results_dir

    client = OllamaClient()
    all_results: list[ExperimentResult] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"structured_experiment_{timestamp}.jsonl"

    total = len(models) * len(schema_names) * len(temperatures) * repeats

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Structured experiment", total=total)

        for model in models:
            for schema_name in schema_names:
                schema_class = SCHEMA_REGISTRY[schema_name]
                # Access the class-level PROMPT attribute
                prompt = schema_class.PROMPT  # type: ignore[attr-defined]

                for temperature in temperatures:
                    for rep in range(1, repeats + 1):
                        progress.update(
                            task,
                            description=(
                                f"{model} | {schema_name} | t={temperature:.1f} [{rep}/{repeats}]"
                            ),
                        )

                        validated, raw, attempts, latency_ms = await enforce_schema(
                            client, model, prompt, schema_class, temperature=temperature
                        )

                        result = ExperimentResult(
                            model=model,
                            schema_name=schema_name,
                            temperature=temperature,
                            repeat=rep,
                            success=validated is not None,
                            attempts=attempts,
                            latency_ms=latency_ms,
                            raw_response=raw,
                            parsed_result=validated.model_dump() if validated is not None else None,
                        )
                        _save_result(result, output_path)
                        all_results.append(result)
                        progress.advance(task)

            # Unload model between models to free RAM
            progress.update(task, description=f"Unloading {model}...")
            await client.unload_model(model)
            console.print(f"[dim]Unloaded {model}[/dim]")

    console.print(f"\n[dim]Results saved to {output_path}[/dim]")
    return all_results


def print_experiment_summary(results: list[ExperimentResult]) -> None:
    """Print per-(model, schema, temperature) success rate and avg latency."""
    if not results:
        console.print("[yellow]No experiment results to summarise.[/yellow]")
        return

    # Aggregate: (model, schema, temperature) -> {successes, total, latencies}
    from collections import defaultdict

    Cell = dict  # {successes: int, total: int, latency_sum: float, retry_count: int}
    agg: dict[tuple, Cell] = defaultdict(lambda: {"successes": 0, "total": 0, "latency_sum": 0.0, "retry_count": 0})

    for r in results:
        key = (r.model, r.schema_name, r.temperature)
        agg[key]["total"] += 1
        agg[key]["latency_sum"] += r.latency_ms
        if r.success:
            agg[key]["successes"] += 1
        if r.attempts == 2:
            agg[key]["retry_count"] += 1

    table = Table(title="Structured Output Experiment Summary", show_lines=True)
    table.add_column("Model", style="bold cyan")
    table.add_column("Schema")
    table.add_column("Temp", justify="right")
    table.add_column("Success Rate", justify="right")
    table.add_column("Retries", justify="right")
    table.add_column("Avg Latency (ms)", justify="right")

    for (model, schema_name, temp), cell in sorted(agg.items()):
        n = cell["total"]
        rate = cell["successes"] / n * 100
        avg_lat = cell["latency_sum"] / n
        rate_str = f"{rate:.0f}% ({cell['successes']}/{n})"
        colour = "green" if rate == 100 else ("yellow" if rate >= 50 else "red")
        table.add_row(
            model,
            schema_name,
            f"{temp:.1f}",
            f"[{colour}]{rate_str}[/{colour}]",
            str(cell["retry_count"]),
            f"{avg_lat:.0f}",
        )

    console.print(table)


def load_experiment_results(path: Path) -> list[ExperimentResult]:
    results = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(ExperimentResult.from_json(line))
    return results
