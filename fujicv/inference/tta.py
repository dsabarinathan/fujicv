"""Test-Time Augmentation (TTA) for FujiCV inference."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in TTA transform sets
# ---------------------------------------------------------------------------

def _hflip(img: np.ndarray) -> np.ndarray:
    return img[:, ::-1, :].copy()

def _vflip(img: np.ndarray) -> np.ndarray:
    return img[::-1, :, :].copy()

def _rot90(img: np.ndarray) -> np.ndarray:
    return np.rot90(img, k=1).copy()

def _rot180(img: np.ndarray) -> np.ndarray:
    return np.rot90(img, k=2).copy()

def _rot270(img: np.ndarray) -> np.ndarray:
    return np.rot90(img, k=3).copy()

def _identity(img: np.ndarray) -> np.ndarray:
    return img

def _brightness_up(img: np.ndarray) -> np.ndarray:
    return np.clip(img.astype(np.int32) + 20, 0, 255).astype(np.uint8)

def _brightness_down(img: np.ndarray) -> np.ndarray:
    return np.clip(img.astype(np.int32) - 20, 0, 255).astype(np.uint8)


_PRESET_AUGMENTS: Dict[str, List] = {
    "hflip": [_identity, _hflip],
    "hflip_vflip": [_identity, _hflip, _vflip],
    "rotate": [_identity, _rot90, _rot180, _rot270],
    "hflip_rotate": [_identity, _hflip, _rot90, _rot180, _rot270],
    "brightness": [_identity, _brightness_up, _brightness_down],
    "standard": [_identity, _hflip, _vflip, _rot90, _rot180, _rot270],
    "full": [
        _identity, _hflip, _vflip,
        _rot90, _rot180, _rot270,
        _brightness_up, _brightness_down,
    ],
}


class TTAPredictor:
    """Test-Time Augmentation wrapper for FujiCV models.

    Runs each image through multiple augmented views, then averages (or
    takes the max of) the resulting probability distributions.

    Args:
        model: Trained ``nn.Module`` in eval mode.
        transform: Albumentations ``Compose`` pipeline (the standard val
            transform — normalisation + resize only; TTA augments are applied
            *before* this).
        task: ``'classification'``, ``'regression'``, or ``'multilabel'``.
        augments: Either a preset name string (``'hflip'``, ``'rotate'``,
            ``'hflip_rotate'``, ``'brightness'``, ``'standard'``, ``'full'``)
            or a list of callables ``fn(np.ndarray) -> np.ndarray``.
            Default ``'hflip'`` (original + horizontal flip).
        merge: How to combine predictions across augmented views.
            ``'mean'`` (default) averages probabilities/logits.
            ``'max'`` takes the element-wise maximum.
        class_to_idx: Label-name → int mapping (classification).
        device: Inference device.

    Example::

        from fujicv.inference.tta import TTAPredictor
        from fujicv.data.transforms import get_val_transforms

        tta = TTAPredictor(
            model=model,
            transform=get_val_transforms(224),
            task='classification',
            augments='hflip_rotate',
            class_to_idx=class_to_idx,
        )
        label, confidence = tta.predict('cat.jpg')
        df = tta.predict_batch(val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        transform: Any,
        task: str = "classification",
        augments: Union[str, List] = "hflip",
        merge: str = "mean",
        class_to_idx: Optional[Dict[str, int]] = None,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.transform = transform
        self.task = task
        self.merge = merge
        self.class_to_idx = class_to_idx or {}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        if isinstance(augments, str):
            if augments not in _PRESET_AUGMENTS:
                raise ValueError(
                    f"Unknown augments preset {augments!r}. "
                    f"Choose from: {list(_PRESET_AUGMENTS)}"
                )
            self.augments = _PRESET_AUGMENTS[augments]
        else:
            self.augments = list(augments)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device).eval()

        logger.info(
            "TTAPredictor: task=%s  augments=%d views  merge=%s  device=%s",
            task, len(self.augments), merge, self.device,
        )

    # ------------------------------------------------------------------
    # Image → augmented tensors
    # ------------------------------------------------------------------

    def _load_np(self, image_or_path) -> np.ndarray:
        if isinstance(image_or_path, (str,)):
            return np.array(Image.open(image_or_path).convert("RGB"))
        elif hasattr(image_or_path, "__fspath__"):
            return np.array(Image.open(image_or_path).convert("RGB"))
        elif isinstance(image_or_path, Image.Image):
            return np.array(image_or_path.convert("RGB"))
        else:
            return np.asarray(image_or_path)

    def _augmented_batch(self, img_np: np.ndarray) -> torch.Tensor:
        """Return a (N_aug, C, H, W) tensor of augmented views."""
        tensors = []
        for aug_fn in self.augments:
            aug_img = aug_fn(img_np)
            t = self.transform(image=aug_img)["image"]  # (C, H, W)
            tensors.append(t)
        return torch.stack(tensors)  # (N_aug, C, H, W)

    # ------------------------------------------------------------------
    # Core TTA forward
    # ------------------------------------------------------------------

    def _tta_forward(self, aug_batch: torch.Tensor) -> torch.Tensor:
        """Run all augmented views and merge logits/probs.

        Args:
            aug_batch: ``(N_aug, C, H, W)`` tensor on CPU.

        Returns:
            Merged output tensor of shape ``(1, num_classes)`` or ``(1,)``.
        """
        aug_batch = aug_batch.to(self.device)
        with torch.no_grad():
            logits = self.model(aug_batch)  # (N_aug, num_classes) or (N_aug,)

        if self.task in ("classification", "multiclass"):
            probs = torch.softmax(logits, dim=-1)
            merged = probs.mean(dim=0, keepdim=True) if self.merge == "mean" else probs.max(dim=0).values.unsqueeze(0)
        elif self.task == "multilabel":
            probs = torch.sigmoid(logits)
            merged = probs.mean(dim=0, keepdim=True) if self.merge == "mean" else probs.max(dim=0).values.unsqueeze(0)
        else:
            # regression — average raw outputs
            merged = logits.mean(dim=0, keepdim=True)

        return merged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_or_path) -> Tuple[Any, float]:
        """TTA prediction for a single image.

        Args:
            image_or_path: File path string, ``pathlib.Path``, PIL Image,
                or ``np.ndarray`` (H, W, 3) uint8.

        Returns:
            Same contract as ``Predictor.predict``:
            - Classification: ``(label_str, confidence)``
            - Regression: ``(scalar_float, 1.0)``
            - Multilabel: ``(list_of_labels, mean_confidence)``
        """
        img_np = self._load_np(image_or_path)
        aug_batch = self._augmented_batch(img_np)
        merged = self._tta_forward(aug_batch)
        return self._decode(merged)

    def predict_proba(self, image_or_path) -> np.ndarray:
        """Return merged probability array for a single image.

        For classification returns shape ``(num_classes,)``.
        For multilabel returns shape ``(num_labels,)``.
        For regression returns shape ``(1,)``.
        """
        img_np = self._load_np(image_or_path)
        aug_batch = self._augmented_batch(img_np)
        merged = self._tta_forward(aug_batch)
        return merged.squeeze(0).cpu().numpy()

    def predict_batch(
        self,
        dataloader: DataLoader,
        return_proba: bool = False,
    ) -> "pd.DataFrame":
        """Run TTA over a DataLoader and return a results DataFrame.

        The DataLoader must yield ``(image_tensor, label)`` batches where
        ``image_tensor`` is already normalised by the standard val transform.
        TTA augments **cannot** be applied here since we receive tensors,
        not raw images.  For raw-image TTA over many files use
        :meth:`predict` in a loop or :meth:`predict_dataset`.

        When ``return_proba=True``, a ``proba`` column with the probability
        array is included.

        Args:
            dataloader: DataLoader of (image_tensor, *) batches.
            return_proba: Include softmax probability arrays in output.

        Returns:
            DataFrame with columns ``prediction``, ``confidence``
            (and optionally ``proba``).
        """
        import pandas as pd  # local to keep import optional at module level

        rows: List[Dict[str, Any]] = []
        self.model.eval()

        with torch.no_grad():
            for batch in dataloader:
                images = batch[0].to(self.device)
                logits = self.model(images)

                if self.task in ("classification", "multiclass"):
                    probs = torch.softmax(logits, dim=-1)
                elif self.task == "multilabel":
                    probs = torch.sigmoid(logits)
                else:
                    probs = logits

                for i in range(images.size(0)):
                    p = probs[i : i + 1]
                    label, conf = self._decode(p)
                    row: Dict[str, Any] = {"prediction": label, "confidence": conf}
                    if return_proba:
                        row["proba"] = p.squeeze(0).cpu().numpy()
                    rows.append(row)

        return pd.DataFrame(rows)

    def predict_dataset(
        self,
        paths: List[Union[str, "Path"]],
        return_proba: bool = False,
    ) -> "pd.DataFrame":
        """Run full TTA (with image augmentation) over a list of file paths.

        This is the correct method to use when you want TTA augmentation
        applied at inference time to disk images.

        Args:
            paths: List of image file paths.
            return_proba: Include probability array in output.

        Returns:
            DataFrame with columns ``path``, ``prediction``, ``confidence``
            (and optionally ``proba``).
        """
        import pandas as pd

        rows = []
        for path in paths:
            label, conf = self.predict(path)
            row: Dict[str, Any] = {"path": str(path), "prediction": label, "confidence": conf}
            if return_proba:
                row["proba"] = self.predict_proba(path)
            rows.append(row)
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def _decode(self, probs: torch.Tensor) -> Tuple[Any, float]:
        if self.task in ("classification", "multiclass"):
            p = probs[0]
            idx = p.argmax().item()
            label = self.idx_to_class.get(idx, str(idx))
            return label, float(p[idx].item())
        elif self.task == "regression":
            return float(probs[0].item()), 1.0
        elif self.task == "multilabel":
            p = probs[0]
            mask = p >= 0.5
            labels = [self.idx_to_class.get(i, str(i)) for i, m in enumerate(mask) if m]
            conf = float(p[mask].mean().item()) if mask.any() else 0.0
            return labels, conf
        else:
            raise ValueError(f"Unknown task: {self.task}")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def tta_predict(
    model: nn.Module,
    image_or_path,
    transform: Any,
    task: str = "classification",
    augments: Union[str, List] = "hflip",
    class_to_idx: Optional[Dict[str, int]] = None,
    device: Optional[str] = None,
) -> Tuple[Any, float]:
    """One-shot TTA prediction without manually instantiating TTAPredictor.

    Args:
        model: Trained model.
        image_or_path: Image path or array.
        transform: Val transform (albumentations Compose).
        task: ``'classification'``, ``'regression'``, or ``'multilabel'``.
        augments: Preset name or list of augmentation callables.
        class_to_idx: Label mapping.
        device: Inference device.

    Returns:
        ``(prediction, confidence)`` — same as ``TTAPredictor.predict``.

    Example::

        from fujicv.inference.tta import tta_predict
        from fujicv.data.transforms import get_val_transforms

        label, conf = tta_predict(
            model, "cat.jpg",
            transform=get_val_transforms(224),
            task="classification",
            augments="hflip_rotate",
            class_to_idx={"cat": 0, "dog": 1},
        )
    """
    predictor = TTAPredictor(
        model=model,
        transform=transform,
        task=task,
        augments=augments,
        class_to_idx=class_to_idx,
        device=device,
    )
    return predictor.predict(image_or_path)
