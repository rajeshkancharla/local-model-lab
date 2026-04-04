from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"

    models: list[str] = Field(
        default=["llama3.2:3b", "phi3.5:3.8b", "gemma3:4b", "qwen3:4b"],
        description="Model tags available in Ollama",
    )

    default_temperature: float = 0.0
    max_retries: int = 1
    benchmark_repeats: int = 3

    # Max seconds to wait for a single inference before skipping it.
    # CPU inference on 4B models can be slow; 90s is generous but prevents
    # the benchmark from hanging indefinitely on complex prompts.
    inference_timeout_s: int = 120

    # Max tokens to generate per inference. Caps runaway generation from
    # models with thinking/reasoning modes (e.g. qwen3.5). 400 is enough
    # for meaningful benchmark responses without multi-minute waits on CPU.
    max_tokens: int = 400

    # num_gpu: 0 = CPU-only default. When Ollama is started with OLLAMA_VULKAN=1
    # and OLLAMA_INTEL_GPU=1 (see server.py), set to -1 to offload all layers.
    # Override via LAB_NUM_GPU env var.
    num_gpu: int = 0

    # Whether the benchmark CLI should kill and restart Ollama with GPU flags
    # before running. Set to true when you want Intel Arc / Vulkan acceleration.
    restart_server_with_gpu: bool = False

    results_dir: Path = PROJECT_ROOT / "data" / "results"
    prompts_dir: Path = PROJECT_ROOT / "data" / "prompts"
    reports_dir: Path = PROJECT_ROOT / "reports"

    model_config = {"env_prefix": "LAB_"}


settings = Settings()
