"""HuggingFace Datasets integration for FujiCV."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_VALID_TASKS = {"classification", "regression", "multilabel"}


class HFImageDataset(Dataset):
    """Wraps a HuggingFace ``datasets.Dataset`` for use with FujiCV Trainer.

    Applies an albumentations (or any ``image= -> dict``) transform pipeline
    and converts labels to the correct tensor type for each task.

    Args:
        hf_dataset: A ``datasets.Dataset`` object (already split/filtered).
        image_col: Name of the column holding PIL Images or file paths.
        label_col: Name of the column holding labels.
        task: ``'classification'``, ``'regression'``, or ``'multilabel'``.
        transform: Albumentations ``Compose`` pipeline or any callable that
            accepts ``image=np.ndarray`` and returns ``{"image": Tensor}``.
        class_to_idx: Mapping from label string → int for classification.
            Built automatically from unique labels if not supplied.
        label_names: For multilabel tasks, list of column names that represent
            each binary label.  When set, *label_col* is ignored.

    Example::

        from datasets import load_dataset
        from fujicv.data.hf_dataset import HFImageDataset, load_hf_dataset
        from fujicv.data.transforms import get_train_transforms

        train_ds, val_ds, class_to_idx = load_hf_dataset(
            "beans", image_col="image", label_col="labels",
            task="classification",
            train_transform=get_train_transforms(224),
            val_transform=get_val_transforms(224),
        )
    """

    def __init__(
        self,
        hf_dataset: Any,
        image_col: str = "image",
        label_col: str = "label",
        task: str = "classification",
        transform: Optional[Any] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        label_names: Optional[List[str]] = None,
    ) -> None:
        if task not in _VALID_TASKS:
            raise ValueError(f"task must be one of {_VALID_TASKS}, got {task!r}")

        self.hf_dataset = hf_dataset
        self.image_col = image_col
        self.label_col = label_col
        self.task = task
        self.transform = transform
        self.label_names = label_names

        # Build class_to_idx for classification
        if task == "classification":
            if class_to_idx is not None:
                self.class_to_idx = class_to_idx
            else:
                # Try to get from HF ClassLabel feature first
                feature = hf_dataset.features.get(label_col)
                if hasattr(feature, "names"):
                    self.class_to_idx = {n: i for i, n in enumerate(feature.names)}
                    logger.info(
                        "Loaded %d classes from HF ClassLabel: %s",
                        len(self.class_to_idx),
                        list(self.class_to_idx.keys())[:10],
                    )
                else:
                    unique = sorted(set(hf_dataset[label_col]))
                    self.class_to_idx = {str(v): i for i, v in enumerate(unique)}
                    logger.info("Built class_to_idx from unique values: %d classes", len(self.class_to_idx))
        else:
            self.class_to_idx = {}

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int):
        sample = self.hf_dataset[idx]

        # --- Image ---
        raw = sample[self.image_col]
        if isinstance(raw, Image.Image):
            img = raw.convert("RGB")
        elif isinstance(raw, (str, bytes)):
            img = Image.open(raw).convert("RGB")
        else:
            img = Image.fromarray(np.array(raw)).convert("RGB")

        img_np = np.array(img)

        if self.transform is not None:
            result = self.transform(image=img_np)
            img_tensor = result["image"]
        else:
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0

        # --- Label ---
        if self.task == "classification":
            raw_label = sample[self.label_col]
            # HF ClassLabel stores as int already; str labels need mapping
            if isinstance(raw_label, int):
                label = torch.tensor(raw_label, dtype=torch.long)
            else:
                label = torch.tensor(self.class_to_idx[str(raw_label)], dtype=torch.long)

        elif self.task == "regression":
            label = torch.tensor(float(sample[self.label_col]), dtype=torch.float32)

        elif self.task == "multilabel":
            if self.label_names:
                vals = [float(sample[c]) for c in self.label_names]
            else:
                vals = [float(v) for v in sample[self.label_col]]
            label = torch.tensor(vals, dtype=torch.float32)

        return img_tensor, label


def load_hf_dataset(
    repo_id: str,
    image_col: str = "image",
    label_col: str = "label",
    task: str = "classification",
    train_split: str = "train",
    val_split: Optional[str] = "validation",
    test_split: Optional[str] = "test",
    train_transform: Optional[Any] = None,
    val_transform: Optional[Any] = None,
    val_fraction: float = 0.1,
    seed: int = 42,
    label_names: Optional[List[str]] = None,
    **load_kwargs: Any,
) -> tuple:
    """Download a HuggingFace dataset and return FujiCV-compatible Dataset objects.

    Args:
        repo_id: HuggingFace Hub dataset ID, e.g. ``'beans'``, ``'food101'``,
            ``'Maysee/tiny-imagenet'``.
        image_col: Column name for images (default ``'image'``).
        label_col: Column name for labels (default ``'label'``).
        task: ``'classification'`` (default), ``'regression'``, or ``'multilabel'``.
        train_split: Name of the training split (default ``'train'``).
        val_split: Name of the validation split.  If the dataset has no
            validation split, pass ``None`` and a fraction of *train_split*
            will be held out automatically (controlled by *val_fraction*).
        test_split: Name of the test split (default ``'test'``).  Pass
            ``None`` if the dataset has no test split.
        train_transform: Albumentations transform for training images.
        val_transform: Albumentations transform for validation/test images.
        val_fraction: Fraction of training data to use as validation when the
            dataset has no dedicated validation split.
        seed: Random seed for the automatic train/val split.
        label_names: For multilabel tasks, list of column names per label.
        **load_kwargs: Extra keyword arguments forwarded to
            ``datasets.load_dataset`` (e.g. ``trust_remote_code=True``).

    Returns:
        ``(train_dataset, val_dataset, class_to_idx)`` where *test_dataset* is
        included as a 4th element when *test_split* is not None:
        ``(train_dataset, val_dataset, test_dataset, class_to_idx)``.

    Raises:
        ImportError: When the ``datasets`` package is not installed.

    Example::

        from fujicv.data.hf_dataset import load_hf_dataset
        from fujicv.data.transforms import get_train_transforms, get_val_transforms

        train_ds, val_ds, class_to_idx = load_hf_dataset(
            "beans",
            train_transform=get_train_transforms(224),
            val_transform=get_val_transforms(224),
        )
        print(f"{len(train_ds)} train | {len(val_ds)} val | classes: {list(class_to_idx)}")
    """
    try:
        from datasets import load_dataset as hf_load
    except ImportError as exc:
        raise ImportError(
            "HuggingFace Datasets is required. "
            "Install with: pip install 'fujicv[hf]' or pip install datasets"
        ) from exc

    logger.info("Loading HuggingFace dataset '%s' …", repo_id)
    raw = hf_load(repo_id, **load_kwargs)

    # --- Training split ---
    train_hf = raw[train_split]

    # --- Validation split ---
    has_val_split = val_split is not None and val_split in raw
    if has_val_split:
        val_hf = raw[val_split]
    else:
        logger.info(
            "No '%s' split found; holding out %.0f%% of train as validation.",
            val_split,
            val_fraction * 100,
        )
        split = train_hf.train_test_split(test_size=val_fraction, seed=seed)
        train_hf = split["train"]
        val_hf = split["test"]

    # --- Test split ---
    test_hf = raw.get(test_split) if test_split else None

    # Build class_to_idx from training split only
    if task == "classification":
        feature = train_hf.features.get(label_col)
        if hasattr(feature, "names"):
            class_to_idx: Dict[str, int] = {n: i for i, n in enumerate(feature.names)}
        else:
            unique = sorted(set(train_hf[label_col]))
            class_to_idx = {str(v): i for i, v in enumerate(unique)}
        logger.info("Classes (%d): %s", len(class_to_idx), list(class_to_idx.keys()))
    else:
        class_to_idx = {}

    def _wrap(hf_ds, transform):
        return HFImageDataset(
            hf_ds,
            image_col=image_col,
            label_col=label_col,
            task=task,
            transform=transform,
            class_to_idx=class_to_idx or None,
            label_names=label_names,
        )

    train_ds = _wrap(train_hf, train_transform)
    val_ds = _wrap(val_hf, val_transform)

    logger.info("Dataset ready — train: %d | val: %d", len(train_ds), len(val_ds))

    if test_hf is not None:
        test_ds = _wrap(test_hf, val_transform)
        logger.info("test: %d", len(test_ds))
        return train_ds, val_ds, test_ds, class_to_idx

    return train_ds, val_ds, class_to_idx
