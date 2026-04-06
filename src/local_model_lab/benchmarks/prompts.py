"""Load and manage prompt sets from YAML files."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from local_model_lab.config import settings


@dataclass
class Prompt:
    id: str
    category: str
    text: str
    expected_format: str = "free_text"
    difficulty: str = "medium"
    # Phase 3: quality scoring metadata
    expected_keywords: list[str] = field(default_factory=list)
    min_length: int = 100  # minimum response length for a "complete" answer


def load_prompts(name: str) -> list[Prompt]:
    """Load a prompt set by name (e.g. 'benchmark_quick' or 'comparison_full').

    Looks for ``{name}.yaml`` inside the configured prompts directory.
    """
    path = settings.prompts_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return [Prompt(**item) for item in data]
