"""Reproducibility and device utilities."""

from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger(__name__)


def get_device(preferred: str | None = None):
    """Auto-detect the best available compute device.

    Priority: CUDA → MPS (Apple Silicon) → CPU.
    Pass *preferred* to override (e.g. ``"cpu"`` to force CPU).

    Returns:
        A ``torch.device`` instance.
    """
    import torch

    if preferred is not None:
        device = torch.device(preferred)
        logger.info("Using user-specified device: %s", device)
        return device

    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        logger.info("CUDA available — using GPU: %s", name)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Apple MPS available — using MPS device")
    else:
        device = torch.device("cpu")
        logger.info("No GPU found — using CPU")

    return device


def set_seed(seed: int) -> None:
    """Seed every RNG that might affect training reproducibility.

    Sets:
    * Python built-in ``random``
    * ``numpy`` (if installed)
    * ``torch`` CPU & CUDA RNGs
    * ``torch.backends.cudnn`` deterministic / benchmark flags

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
