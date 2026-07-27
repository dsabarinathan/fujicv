"""TorchScript export and verification utilities."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def export_torchscript(
    model: nn.Module,
    output_path: Union[str, Path],
    example_inputs: torch.Tensor,
    method: str = "trace",
    strict: bool = True,
) -> torch.jit.ScriptModule:
    """Export a PyTorch model to TorchScript.

    Args:
        model: Trained ``nn.Module`` to export (set to eval mode automatically).
        output_path: Destination ``.pt`` file path.
        example_inputs: Example input tensor used for tracing (ignored for
            ``method='script'``).
        method: ``'trace'`` (default) — use ``torch.jit.trace``; or
            ``'script'`` — use ``torch.jit.script`` (requires scriptable model).
        strict: Passed to ``torch.jit.trace``.  Set ``False`` if the model has
            data-dependent control flow that tracing cannot capture exactly.

    Returns:
        The compiled :class:`torch.jit.ScriptModule`.

    Raises:
        ValueError: If *method* is not ``'trace'`` or ``'script'``.

    Example::

        from fujicv.export.torchscript import export_torchscript, verify_torchscript
        import torch

        scripted = export_torchscript(model, "model.pt", torch.randn(1, 3, 224, 224))
        verify_torchscript(scripted, torch.randn(1, 3, 224, 224), original_model=model)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval()
    with torch.no_grad():
        if method == "trace":
            scripted = torch.jit.trace(model, example_inputs, strict=strict)
        elif method == "script":
            scripted = torch.jit.script(model)
        else:
            raise ValueError(f"method must be 'trace' or 'script', got '{method!r}'")

    torch.jit.save(scripted, str(output_path))
    logger.info("Model exported to TorchScript: %s", output_path)
    return scripted


def verify_torchscript(
    scripted: torch.jit.ScriptModule,
    example_inputs: torch.Tensor,
    original_model: Optional[nn.Module] = None,
    atol: float = 1e-4,
) -> bool:
    """Verify that TorchScript outputs match the original model.

    Args:
        scripted: The exported :class:`torch.jit.ScriptModule`.
        example_inputs: Input tensor for comparison.
        original_model: Original PyTorch model.  If ``None``, skip comparison.
        atol: Absolute tolerance.

    Returns:
        ``True`` if outputs match within *atol* (or if *original_model* is
        ``None``).

    Raises:
        ValueError: If outputs differ by more than *atol*.
    """
    scripted.eval()
    with torch.no_grad():
        ts_out = scripted(example_inputs)

        if original_model is not None:
            original_model.eval()
            orig_out = original_model(example_inputs)
            if not torch.allclose(orig_out, ts_out, atol=atol):
                max_diff = float((orig_out - ts_out).abs().max())
                raise ValueError(
                    f"TorchScript output mismatch: max diff = {max_diff:.6f} (atol={atol})"
                )

    logger.info("TorchScript verification passed (atol=%.1e).", atol)
    return True


def load_torchscript(
    path: Union[str, Path],
    map_location: Optional[str] = None,
) -> torch.jit.ScriptModule:
    """Load a saved TorchScript model.

    Args:
        path: Path to the ``.pt`` file.
        map_location: Device mapping string, e.g. ``'cpu'`` or ``'cuda:0'``.

    Returns:
        The loaded :class:`torch.jit.ScriptModule`.
    """
    return torch.jit.load(str(path), map_location=map_location)
