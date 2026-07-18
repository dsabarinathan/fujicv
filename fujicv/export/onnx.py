"""ONNX export and verification utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def to_onnx(
    model: nn.Module,
    path: Union[str, Path],
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    opset_version: int = 17,
    dynamic_batch: bool = True,
) -> Path:
    """Export a PyTorch model to ONNX format.

    Args:
        model: Trained ``nn.Module`` to export.
        path: Destination ``.onnx`` file path.
        input_size: Dummy input shape ``(B, C, H, W)`` (default ``(1,3,224,224)``).
        opset_version: ONNX opset version (default 17).
        dynamic_batch: Export with a dynamic first dimension so the model
            accepts variable batch sizes at inference time.

    Returns:
        Path to the written ONNX file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy = torch.zeros(*input_size)

    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}} if dynamic_batch else {}

    torch.onnx.export(
        model,
        dummy,
        str(path),
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )
    logger.info("Model exported to ONNX: %s", path)
    return path


def verify_onnx(
    model: nn.Module,
    onnx_path: Union[str, Path],
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> bool:
    """Verify that an ONNX model produces outputs close to the PyTorch model.

    Requires ``onnxruntime`` and ``onnx`` to be installed.

    Args:
        model: Original PyTorch model.
        onnx_path: Path to the ONNX file.
        input_size: Input shape used for verification (default ``(1,3,224,224)``).
        atol: Absolute tolerance for ``numpy.allclose`` (default 1e-4).
        rtol: Relative tolerance for ``numpy.allclose`` (default 1e-3).

    Returns:
        ``True`` if outputs match within tolerance, ``False`` otherwise.

    Raises:
        ImportError: If ``onnxruntime`` or ``onnx`` is not installed.
    """
    try:
        import onnx  # type: ignore
        import onnxruntime as ort  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "onnx and onnxruntime are required for verification. "
            "Install with: pip install fujicv[onnx]"
        ) from exc

    onnx_path = Path(onnx_path)

    # Validate ONNX graph
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    dummy_np = np.random.rand(*input_size).astype(np.float32)
    dummy_torch = torch.tensor(dummy_np)

    # PyTorch output
    model.eval()
    with torch.no_grad():
        pt_out = model(dummy_torch).numpy()

    # ONNX Runtime output
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    ort_out = sess.run(None, {input_name: dummy_np})[0]

    match = bool(np.allclose(pt_out, ort_out, atol=atol, rtol=rtol))
    if match:
        logger.info("ONNX verification passed (atol=%.1e, rtol=%.1e).", atol, rtol)
    else:
        max_diff = float(np.abs(pt_out - ort_out).max())
        logger.warning(
            "ONNX verification FAILED. Max absolute difference: %.6f", max_diff
        )
    return match
