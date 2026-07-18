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
    ) -> Tuple[Any, float]:
        """Predict a single image.

        Args:
            image_or_path: Image file path or numpy RGB array.

        Returns:
            For classification: ``(label_string, confidence_float)``.
            For regression: ``(scalar_float, 1.0)``.
            For multilabel: ``(list_of_predicted_labels, mean_confidence)``.
        """
        tensor = self._load_image(image_or_path).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)

        return self._decode(logits)

    def _decode(self, logits: torch.Tensor) -> Tuple[Any, float]:
        if self.task in ("classification", "multiclass"):
            probs = torch.softmax(logits, dim=-1)[0]
            idx = probs.argmax().item()
            label = self.idx_to_class.get(idx, str(idx))
            return label, float(probs[idx].item())
        elif self.task == "regression":
            return float(logits[0].item()), 1.0
        elif self.task == "multilabel":
            probs = torch.sigmoid(logits)[0]
            mask = probs >= 0.5
            labels = [self.idx_to_class.get(i, str(i)) for i, m in enumerate(mask) if m]
            mean_conf = float(probs[mask].mean().item()) if mask.any() else 0.0
            return labels, mean_conf
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def predict_batch(
        self,
        dataloader: DataLoader,
        image_col: str = "image",
    ) -> pd.DataFrame:
        """Run predictions over a DataLoader and return a results DataFrame.

        Args:
            dataloader: DataLoader yielding ``(image_tensor, label)`` batches.
            image_col: Column name for the image identifier in the output.

        Returns:
            DataFrame with columns: ``image``, ``prediction``, ``confidence``.
        """
        rows: List[Dict[str, Any]] = []
        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images = batch[0].to(self.device)
                logits = self.model(images)
                batch_size = images.size(0)
                for i in range(batch_size):
                    label, conf = self._decode(logits[i : i + 1])
                    rows.append(
                        {
                            image_col: f"batch{batch_idx}_sample{i}",
                            "prediction": label,
                            "confidence": conf,
                        }
                    )

        return pd.DataFrame(rows)
