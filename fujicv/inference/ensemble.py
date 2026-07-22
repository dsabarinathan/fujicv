"""Multi-model ensemble prediction utilities."""

from __future__ import annotations

import logging
from typing import Callable, List, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from fujicv.utils.seed import get_device

logger = logging.getLogger(__name__)

MergeStrategy = Literal["mean", "vote", "max", "weighted_mean"]


class EnsemblePredictor:
    """Combine predictions from multiple models.

    Supports three tasks (classification, regression, multilabel) and four
    merge strategies.  All models are moved to the target device and set to
    eval mode automatically.

    Args:
        models: List of trained ``nn.Module`` instances.
        merge: How to combine per-model outputs:

            * ``'mean'`` — average raw logits / regression outputs (default).
            * ``'vote'`` — hard majority vote (classification only).
            * ``'max'`` — take element-wise maximum (multilabel).
            * ``'weighted_mean'`` — weighted average; requires *weights*.
        task: ``'classification'``, ``'regression'``, or ``'multilabel'``.
        weights: Per-model weights for ``'weighted_mean'``.  Must sum to 1.
        device: Compute device (default: auto).

    Example::

        from fujicv.inference.ensemble import EnsemblePredictor

        ensemble = EnsemblePredictor([model_a, model_b, model_c],
                                      merge='mean', task='classification')
        probs  = ensemble.predict_proba(image_tensor)   # (num_classes,)
        label  = ensemble.predict(image_tensor)          # int
        result = ensemble.predict_batch(val_loader)
    """

    def __init__(
        self,
        models: List[nn.Module],
        merge: MergeStrategy = "mean",
        task: str = "classification",
        weights: Optional[List[float]] = None,
        device: Optional[str] = None,
    ) -> None:
        if not models:
            raise ValueError("models list must not be empty")

        valid_merges = {"mean", "vote", "max", "weighted_mean"}
        if merge not in valid_merges:
            raise ValueError(f"merge must be one of {valid_merges}, got '{merge}'")

        if merge == "weighted_mean":
            if weights is None:
                raise ValueError("weights are required for merge='weighted_mean'")
            if len(weights) != len(models):
                raise ValueError("len(weights) must equal len(models)")
            total = sum(weights)
            self._weights = [w / total for w in weights]
        else:
            self._weights = None

        self.device = get_device(device)
        self.task   = task
        self.merge  = merge
        self.models = [m.to(self.device).eval() for m in models]

    # ------------------------------------------------------------------

    def _forward_all(self, image: torch.Tensor) -> torch.Tensor:
        """Run all models and return stacked logits ``(M, B, C)``."""
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device)

        outputs = []
        with torch.no_grad():
            for model in self.models:
                out = model(image)
                outputs.append(out)

        return torch.stack(outputs, dim=0)   # (M, B, ...)

    def _merge(self, stacked: torch.Tensor) -> torch.Tensor:
        """Merge M model outputs along dim 0."""
        if self.merge == "mean":
            return stacked.mean(dim=0)

        if self.merge == "weighted_mean":
            w = torch.tensor(self._weights, dtype=stacked.dtype, device=stacked.device)
            while w.ndim < stacked.ndim:
                w = w.unsqueeze(-1)
            return (stacked * w).sum(dim=0)

        if self.merge == "vote":
            preds = stacked.argmax(dim=-1)   # (M, B)
            # Majority vote per sample
            B = preds.shape[1]
            result = []
            for b in range(B):
                votes = preds[:, b]
                mode  = int(torch.mode(votes).values.item())
                result.append(mode)
            # Return one-hot style — for consistency, return mean logits
            return stacked.mean(dim=0)   # label extracted in predict()

        if self.merge == "max":
            return stacked.max(dim=0).values

        raise ValueError(f"Unknown merge strategy: {self.merge}")

    # ------------------------------------------------------------------

    def predict(self, image: torch.Tensor) -> Union[int, float, np.ndarray]:
        """Predict a single image.

        Returns:
            int class index (classification), float value (regression),
            or 1-D numpy array of binary labels (multilabel).
        """
        stacked = self._forward_all(image)        # run once, reuse below
        merged  = self._merge(stacked)            # (1, C) or (1,)

        if self.task == "classification":
            if self.merge == "vote":
                preds = stacked.argmax(dim=-1).squeeze(1)   # (M,)
                return int(torch.mode(preds).values.item())
            return int(merged.argmax(dim=-1).item())

        if self.task == "regression":
            return float(merged.squeeze().item())

        if self.task == "multilabel":
            return (torch.sigmoid(merged) > 0.5).squeeze().cpu().numpy().astype(int)

        raise ValueError(f"Unknown task: {self.task}")

    def predict_proba(self, image: torch.Tensor) -> np.ndarray:
        """Return averaged softmax probabilities ``(num_classes,)``."""
        merged = self._merge(self._forward_all(image))   # (1, C)
        if self.task == "multilabel":
            return torch.sigmoid(merged).squeeze().cpu().numpy()
        return F.softmax(merged, dim=-1).squeeze().cpu().numpy()

    def predict_batch(
        self,
        loader: DataLoader,
        return_targets: bool = False,
    ) -> Union[np.ndarray, tuple]:
        """Run ensemble over a full DataLoader.

        Args:
            loader: DataLoader yielding ``(images, targets)`` or ``(images,)``.
            return_targets: If True, also return ground-truth targets.

        Returns:
            ``predictions`` array, or ``(predictions, targets)`` if
            *return_targets* is True.
        """
        all_preds:   List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images, targets = batch
                if return_targets:
                    all_targets.append(targets.numpy())
            else:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch

            stacked = self._forward_all(images)
            merged  = self._merge(stacked)   # (B, C)

            if self.task == "classification":
                if self.merge == "vote":
                    hard_preds = stacked.argmax(dim=-1)   # (M, B)
                    batch_preds = torch.mode(hard_preds, dim=0).values.cpu().numpy()
                else:
                    batch_preds = merged.argmax(dim=-1).cpu().numpy()
            elif self.task == "regression":
                batch_preds = merged.squeeze(-1).cpu().numpy()
            else:
                batch_preds = (torch.sigmoid(merged) > 0.5).cpu().numpy().astype(int)

            all_preds.append(batch_preds)

        predictions = np.concatenate(all_preds)
        if return_targets and all_targets:
            return predictions, np.concatenate(all_targets)
        return predictions
