"""Post-Training Quantization (PTQ) for edge/CPU deployment.

Two complementary recipes:

* :func:`quantize_dynamic` — weights quantized to INT8 ahead of time, activations
  quantized on-the-fly at inference. No calibration data needed. Best for
  Linear/LSTM-heavy models (transformers, MLP heads).
* :func:`quantize_static` — both weights and activations quantized using
  calibration statistics gathered from a few representative batches (FX graph
  mode). Best for CNNs; typically the smallest, fastest result.

Both return CPU models (quantized inference runs on CPU). Use
:func:`measure_model_size` to report the on-disk size reduction.

Example::

    from fujicv.export.quantization import quantize_dynamic, measure_model_size

    fp32_mb = measure_model_size(model)
    qmodel  = quantize_dynamic(model)
    int8_mb = measure_model_size(qmodel)
    print(f"{fp32_mb:.1f} MB → {int8_mb:.1f} MB")
"""

from __future__ import annotations

import copy
import logging
import tempfile
from pathlib import Path
from typing import Iterable, Optional, Set, Type

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def measure_model_size(model: nn.Module) -> float:
    """Return the serialized size of *model*'s state dict in megabytes."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as fh:
        tmp = Path(fh.name)
    try:
        torch.save(model.state_dict(), tmp)
        size_mb = tmp.stat().st_size / (1024 * 1024)
    finally:
        tmp.unlink(missing_ok=True)
    return size_mb


def quantize_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
    layers: Optional[Set[Type[nn.Module]]] = None,
) -> nn.Module:
    """Apply dynamic post-training quantization (weights → INT8).

    Args:
        model: Trained FP32 model. It is deep-copied and moved to CPU first, so
            the original is left untouched.
        dtype: Quantized dtype (default ``torch.qint8``).
        layers: Module types to quantize (default ``{nn.Linear}``). Pass e.g.
            ``{nn.Linear, nn.LSTM}`` to include recurrent layers.

    Returns:
        A quantized CPU model ready for inference.
    """
    layers = layers or {nn.Linear}
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    quantized = torch.ao.quantization.quantize_dynamic(model_cpu, layers, dtype=dtype)
    logger.info("Dynamic quantization applied to %s.", {t.__name__ for t in layers})
    return quantized


def quantize_static(
    model: nn.Module,
    calibration_data: Iterable,
    backend: str = "x86",
    num_calibration_batches: int = 32,
) -> nn.Module:
    """Apply static post-training quantization via FX graph mode.

    Both weights and activations are quantized to INT8. Activation ranges are
    calibrated by running ``num_calibration_batches`` batches from
    *calibration_data* through the prepared model.

    Args:
        model: Trained FP32 model (deep-copied; original untouched).
        calibration_data: Iterable yielding either input tensors or
            ``(input, ...)`` tuples (only the first element is used).
        backend: Quantization backend — ``'x86'``/``'fbgemm'`` (server CPU) or
            ``'qnnpack'`` (ARM/mobile). Default ``'x86'``.
        num_calibration_batches: How many batches to use for calibration.

    Returns:
        A quantized CPU model.

    Raises:
        RuntimeError: If the model cannot be symbolically traced by FX (e.g. it
            contains data-dependent control flow). Fall back to
            :func:`quantize_dynamic` in that case.
    """
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    torch.backends.quantized.engine = "fbgemm" if backend == "x86" else backend
    model_cpu = copy.deepcopy(model).to("cpu").eval()

    # Grab one example input for tracing.
    example = None
    for batch in calibration_data:
        example = batch[0] if isinstance(batch, (tuple, list)) else batch
        break
    if example is None:
        raise ValueError("calibration_data yielded no batches.")
    example = example.to("cpu")

    qconfig_mapping = get_default_qconfig_mapping(backend)
    try:
        prepared = prepare_fx(model_cpu, qconfig_mapping, example_inputs=(example,))
    except Exception as exc:
        raise RuntimeError(
            "quantize_static: FX symbolic trace failed — the model likely has "
            "data-dependent control flow. Use quantize_dynamic instead. "
            f"Original error: {exc}"
        ) from exc

    # Calibrate.
    with torch.no_grad():
        for i, batch in enumerate(calibration_data):
            if i >= num_calibration_batches:
                break
            x = batch[0] if isinstance(batch, (tuple, list)) else batch
            prepared(x.to("cpu"))

    quantized = convert_fx(prepared)
    logger.info("Static (FX) quantization applied with backend '%s'.", backend)
    return quantized


__all__ = ["quantize_dynamic", "quantize_static", "measure_model_size"]
