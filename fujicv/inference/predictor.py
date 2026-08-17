"""Inference predictor for loading checkpoints and running predictions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

from fujicv.data.transforms import get_val_transforms

logger = logging.getLogger(__name__)


class Predictor:
    """High-level inference wrapper.

    Instantiate via :meth:`from_checkpoint` rather than the constructor
    directly.

    Args:
        model: Trained ``nn.Module``.
        class_to_idx: Optional class-name → index mapping (classification only).
        task: Task type — ``'classification'``, ``'regression'``, or ``'multilabel'``.
        image_size: Expected input image size (default 224).
        device: Inference device (default auto).
    """

    def __init__(
        self,
        model: nn.Module,
        class_to_idx: Optional[Dict[str, int]] = None,
        task: str = "classification",
        image_size: int = 224,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.class_to_idx = class_to_idx or {}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.task = task
        self.image_size = image_size
        self._transform = get_val_transforms(image_size)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device).eval()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: Union[str, Path],
        model: Optional[nn.Module] = None,
        device: Optional[str] = None,
        image_size: int = 224,
    ) -> "Predictor":
        """Load a checkpoint and return a ready-to-use :class:`Predictor`.

        The checkpoint must contain at minimum ``model_state_dict``.  If it
        also contains ``class_to_idx`` and/or ``task`` those values are used
        automatically.

        Args:
            path: Path to a ``.pt`` checkpoint file.
            model: Optional pre-built model skeleton.  Must be provided if the
                checkpoint does not embed the full model (which is the case for
                FujiCV checkpoints — they only store ``state_dict``).
            device: Target device.
            image_size: Expected input size.

        Returns:
            A configured :class:`Predictor` instance.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt = torch.load(path, map_location=device)

        class_to_idx = ckpt.get("class_to_idx", {})
        task = ckpt.get("task", "classification")

        if model is None:
            raise ValueError(
                "A model skeleton must be supplied via the `model` argument. "
                "FujiCV checkpoints store only state_dict, not the full model."
            )

        model.load_state_dict(ckpt["model_state_dict"])
        return cls(
            model=model,
            class_to_idx=class_to_idx,
            task=task,
            image_size=image_size,
            device=device,
        )

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _load_image(self, image_or_path: Union[str, Path, np.ndarray]) -> torch.Tensor:
        if isinstance(image_or_path, (str, Path)):
            img = np.array(Image.open(image_or_path).convert("RGB"))
        else:
            img = np.asarray(image_or_path)
        result = self._transform(image=img)
        return result["image"].unsqueeze(0)  # (1, C, H, W)

    def predict(
        self,
        image_or_path: Union[str, Path, np.ndarray],
        use_tta: bool = False,
    ) -> Tuple[Any, float]:
        """Predict a single image.

        Args:
            image_or_path: Image file path or numpy RGB array.
            use_tta: If ``True``, average predictions over the original image
                and its horizontal flip (test-time augmentation). Averaging is
                done in probability space for classification/multilabel and in
                output space for regression.

        Returns:
            For classification: ``(label_string, confidence_float)``.
            For regression: ``(scalar_float_or_list, 1.0)``.
            For multilabel: ``(list_of_predicted_labels, mean_confidence)``.
        """
        tensor = self._load_image(image_or_path).to(self.device)
        with torch.no_grad():
            if use_tta:
                views = [tensor, torch.flip(tensor, dims=[-1])]  # original + hflip
                scores = torch.stack([self._to_scores(self.model(v)) for v in views]).mean(0)
            else:
                scores = self._to_scores(self.model(tensor))

        labels, confs = self._decode_scores(scores)
        return labels[0], confs[0]

    # ------------------------------------------------------------------
    # Vectorized decoding
    # ------------------------------------------------------------------

    def _to_scores(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert raw logits to the score space used for a task.

        Classification/multiclass → softmax probs; multilabel → sigmoid probs;
        regression → raw outputs. Shape is always ``(B, ...)``.
        """
        if self.task in ("classification", "multiclass"):
            return torch.softmax(logits, dim=-1)
        if self.task == "multilabel":
            return torch.sigmoid(logits)
        return logits  # regression

    def _decode_scores(self, scores: torch.Tensor) -> Tuple[List[Any], List[float]]:
        """Vectorized decode of a ``(B, ...)`` score tensor into labels + confidences."""
        if self.task in ("classification", "multiclass"):
            conf, idx = scores.max(dim=-1)
            labels = [self.idx_to_class.get(int(i), str(int(i))) for i in idx.tolist()]
            return labels, [float(c) for c in conf.tolist()]

        if self.task == "regression":
            vals = scores
            if vals.ndim == 1:  # (B,)
                return [float(v) for v in vals.tolist()], [1.0] * vals.shape[0]
            return [row for row in vals.tolist()], [1.0] * vals.shape[0]  # (B, n)

        if self.task == "multilabel":
            labels_out: List[Any] = []
            confs_out: List[float] = []
            mask = scores >= 0.5
            for row_scores, row_mask in zip(scores, mask):
                present = [self.idx_to_class.get(i, str(i)) for i, m in enumerate(row_mask) if m]
                labels_out.append(present)
                confs_out.append(float(row_scores[row_mask].mean().item()) if row_mask.any() else 0.0)
            return labels_out, confs_out

        raise ValueError(f"Unknown task: {self.task}")

    def _decode(self, logits: torch.Tensor) -> Tuple[Any, float]:
        """Decode a single ``(1, ...)`` logit tensor (kept for backward compat)."""
        labels, confs = self._decode_scores(self._to_scores(logits))
        return labels[0], confs[0]

    def predict_batch(
        self,
        dataloader: DataLoader,
        image_col: str = "image",
        ids: Optional[List[Any]] = None,
        yields_ids: bool = False,
        use_tta: bool = False,
    ) -> pd.DataFrame:
        """Run predictions over a DataLoader and return a results DataFrame.

        Identifiers (for a Kaggle ``submission.csv`` etc.) are resolved in this
        priority order:

        1. ``yields_ids=True`` — the DataLoader yields ``(images, ids)`` batches
           and the second element is used directly as the identifier.
        2. ``ids`` — an explicit list aligned with the dataset order (requires
           ``shuffle=False``); it is sliced per batch.
        3. Fallback — a running ``sample_<idx>`` index.

        Args:
            dataloader: DataLoader yielding ``(images, ...)`` batches.
            image_col: Column name for the identifier in the output.
            ids: Optional list of identifiers aligned with dataset order.
            yields_ids: Treat each batch's second element as identifiers.
            use_tta: Average over the original image and its horizontal flip.

        Returns:
            DataFrame with columns: ``<image_col>``, ``prediction``, ``confidence``.
        """
        all_ids: List[Any] = []
        all_labels: List[Any] = []
        all_confs: List[float] = []
        self.model.eval()

        running = 0
        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(self.device)
                if use_tta:
                    views = [images, torch.flip(images, dims=[-1])]
                    scores = torch.stack([self._to_scores(self.model(v)) for v in views]).mean(0)
                else:
                    scores = self._to_scores(self.model(images))

                labels, confs = self._decode_scores(scores)
                bs = images.size(0)

                if yields_ids and len(batch) > 1:
                    batch_ids = batch[1]
                    batch_ids = batch_ids.tolist() if torch.is_tensor(batch_ids) else list(batch_ids)
                elif ids is not None:
                    batch_ids = ids[running : running + bs]
                else:
                    batch_ids = [f"sample_{running + i}" for i in range(bs)]

                all_ids.extend(batch_ids)
                all_labels.extend(labels)
                all_confs.extend(confs)
                running += bs

        return pd.DataFrame(
            {image_col: all_ids, "prediction": all_labels, "confidence": all_confs}
        )
