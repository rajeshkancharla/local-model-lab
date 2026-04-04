"""Ollama server lifecycle management.

Ollama reads GPU-related environment variables at startup, not per-request.
If it is already running without GPU flags (e.g. started by the tray app),
those flags have no effect until the server is restarted.

This module handles killing any existing Ollama process and starting a fresh
one with the correct flags for Intel Arc GPU (Vulkan backend).
"""

from __future__ import annotations

import os
import subprocess
import time

import httpx

from local_model_lab.config import settings

_READY_URL = f"{settings.ollama_base_url}/api/version"
_STARTUP_TIMEOUT_S = 20
_STARTUP_POLL_S = 2


def _kill_existing() -> None:
    """Terminate any running Ollama processes (tray app and server)."""
    for exe in ("ollama app.exe", "ollama.exe"):
        subprocess.run(
            ["taskkill", "/F", "/IM", exe],
            capture_output=True,  # suppress errors if process isn't running
        )
    time.sleep(3)


def _wait_until_ready() -> None:
    """Poll until Ollama's HTTP server responds or timeout."""
    for _ in range(_STARTUP_TIMEOUT_S // _STARTUP_POLL_S):
        try:
            with httpx.Client(timeout=2.0) as client:
                r = client.get(_READY_URL)
                if r.status_code == 200:
                    return
        except httpx.HTTPError:
            pass
        time.sleep(_STARTUP_POLL_S)
    raise RuntimeError(
        f"Ollama server did not become ready within {_STARTUP_TIMEOUT_S}s"
    )


def start_with_gpu() -> subprocess.Popen:  # type: ignore[type-arg]
    """Kill any existing Ollama instance and start a fresh one with GPU flags.

    Environment variables set:
      OLLAMA_VULKAN=1        — enables the Vulkan compute backend (Intel Arc)
      OLLAMA_INTEL_GPU=1     — tells Ollama to target Intel GPU devices
      OLLAMA_KEEP_ALIVE=0    — models unload immediately after use (memory safety)

    Returns the Popen handle so the caller can terminate it when done.
    """
    _kill_existing()

    env = os.environ.copy()
    env["OLLAMA_VULKAN"] = "1"
    env["OLLAMA_INTEL_GPU"] = "1"
    env["OLLAMA_KEEP_ALIVE"] = "0"

    server = subprocess.Popen(
        ["ollama", "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    _wait_until_ready()
    return server


def stop_server(server: subprocess.Popen) -> None:  # type: ignore[type-arg]
    """Gracefully terminate the managed Ollama server process."""
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()
