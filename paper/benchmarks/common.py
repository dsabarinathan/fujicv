"""Shared utilities for the FujiCV paper benchmarks.

Handles environment capture (for reproducibility), timing, seeding, and writing
JSON results into ``paper/benchmarks/results/``.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict

import torch

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def capture_env() -> Dict[str, Any]:
    """Record the software/hardware environment for reproducibility."""
    try:
        import fujicv
        fujicv_version = fujicv.__version__
    except Exception:
        fujicv_version = "unknown"

    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "fujicv_version": fujicv_version,
        "git_hash": git_hash(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
        "gpus": gpus,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def set_seed(seed: int = 42) -> None:
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_result(name: str, payload: Dict[str, Any]) -> Path:
    """Write ``payload`` (plus env metadata) to ``results/<name>.json``."""
    payload = {"_env": capture_env(), **payload}
    path = RESULTS_DIR / f"{name}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[saved] {path}")
    return path


class Timer:
    """Context manager that records wall-clock seconds (CUDA-synchronized)."""

    def __enter__(self) -> "Timer":
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.t0
