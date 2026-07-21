"""Exponential Moving Average (EMA) of model weights."""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelEMA:
    """Maintains an Exponential Moving Average of a model's parameters.

    Shadow weights are updated after every training step:
    ``shadow = decay * shadow + (1 - decay) * param``

    Bias correction is applied for the first ``warmup_steps`` updates so the
    EMA is not biased toward zero at the start of training.

    Args:
        model: The model whose parameters to track.
        decay: EMA decay rate (default 0.9999). Typical range: 0.999 – 0.9999.
        warmup_steps: Number of updates before bias correction is disabled
            (default 2000).  Set to 0 to skip bias correction entirely.

    Example::

        ema = ModelEMA(model, decay=0.9999)

        for images, targets in train_loader:
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            ema.update(model)       # call after every optimizer.step()

        # Evaluate with EMA weights
        with ema.average_parameters(model):
            val_loss = evaluate(model, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 2000,
    ) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        self.decay = decay
        self.warmup_steps = warmup_steps
        self._num_updates = 0

        # Shadow model — detached copy, no grad
        self.shadow: nn.Module = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------

    def _effective_decay(self) -> float:
        """Bias-corrected decay for the first ``warmup_steps`` updates."""
        if self.warmup_steps == 0:
            return self.decay
        # Rampup: min(decay, (1 + n) / (10 + n))  — common in timm/EfficientNet
        return min(self.decay, (1.0 + self._num_updates) / (self.warmup_steps + self._num_updates))

    def update(self, model: nn.Module) -> None:
        """Update shadow weights from the current model parameters.

        Call this **after** ``optimizer.step()`` on every training step.
        """
        self._num_updates += 1
        d = self._effective_decay()
        with torch.no_grad():
            for s_param, m_param in zip(
                self.shadow.parameters(), model.parameters()
            ):
                s_param.data.mul_(d).add_(m_param.data, alpha=1.0 - d)
            # Also track buffers (e.g. BatchNorm running mean/var)
            for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
                s_buf.copy_(m_buf)

    def apply_to(self, model: nn.Module) -> None:
        """Copy EMA weights into *model* in-place."""
        model.load_state_dict(self.shadow.state_dict())

    def restore(self, original_state: dict) -> None:
        """Restore a model to a previously saved state dict."""
        # Convenience: pair with apply_to for temporary evaluation
        pass

    def state_dict(self) -> dict:
        return {
            "shadow": self.shadow.state_dict(),
            "num_updates": self._num_updates,
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
        }

    def load_state_dict(self, state: dict) -> None:
        self.shadow.load_state_dict(state["shadow"])
        self._num_updates = state["num_updates"]
        self.decay = state["decay"]
        self.warmup_steps = state["warmup_steps"]

    # Context manager: temporarily swap EMA weights into model for evaluation
    def average_parameters(self, model: nn.Module):
        """Context manager: swap EMA weights in, restore originals on exit.

        Example::

            with ema.average_parameters(model):
                acc = evaluate(model, val_loader)
        """
        return _EMAContext(self, model)


class _EMAContext:
    def __init__(self, ema: ModelEMA, model: nn.Module) -> None:
        self.ema   = ema
        self.model = model
        self._saved: Optional[dict] = None

    def __enter__(self):
        self._saved = copy.deepcopy(self.model.state_dict())
        self.ema.apply_to(self.model)
        return self.model

    def __exit__(self, *args):
        self.model.load_state_dict(self._saved)
