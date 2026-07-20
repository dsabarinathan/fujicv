"""DataLoader factory."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import pandas as pd
from torch.utils.data import DataLoader

from fujicv.data.datasets import CSVImageDataset
from fujicv.data.transforms import get_train_transforms, get_val_transforms


def build_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame],
    dataset_cfg: Dict[str, Any],
    aug_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """Build train, val, and (optionally) test DataLoaders.

    Args:
        train_df: Training split DataFrame.
        val_df: Validation split DataFrame.
        test_df: Test split DataFrame (may be ``None`` or empty).
        dataset_cfg: Dataset configuration dict.  Expected keys:

            * ``image_dir`` — image root directory.
            * ``image_col`` — image path column.
            * ``label_col`` — label column.
            * ``task`` — task type.
            * ``class_to_idx`` — (optional) pre-built class mapping.
            * ``batch_size`` — (optional, default 32).
            * ``num_workers`` — (optional, default 4).
            * ``pin_memory`` — (optional, default True).

        aug_cfg: Augmentation configuration dict.  Expected keys:

            * ``image_size`` — (optional, default 224).
            * ``level`` — augmentation level (default ``'medium'``).

    Returns:
        Tuple ``(train_loader, val_loader, test_loader)``.
        *test_loader* is ``None`` when *test_df* is empty or ``None``.
    """
    aug_cfg = aug_cfg or {}
    image_size = int(aug_cfg.get("image_size", 224))
    level = str(aug_cfg.get("level", "medium"))

    image_dir = dataset_cfg["image_dir"]
    image_col = dataset_cfg["image_col"]
    label_col = dataset_cfg["label_col"]
    task = dataset_cfg["task"]
    class_to_idx = dataset_cfg.get("class_to_idx")

    batch_size = int(dataset_cfg.get("batch_size", 32))
    num_workers = int(dataset_cfg.get("num_workers", 4))
    pin_memory = bool(dataset_cfg.get("pin_memory", True))

    train_transform = get_train_transforms(image_size, level=level)
    val_transform = get_val_transforms(image_size)

    train_ds = CSVImageDataset(
        train_df, image_dir, image_col, label_col, task, train_transform, class_to_idx
    )
    val_ds = CSVImageDataset(
        val_df, image_dir, image_col, label_col, task, val_transform,
        class_to_idx or train_ds.class_to_idx
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(train_ds) % batch_size == 1,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader: Optional[DataLoader] = None
    if test_df is not None and len(test_df) > 0:
        test_ds = CSVImageDataset(
            test_df, image_dir, image_col, label_col, task, val_transform,
            class_to_idx or train_ds.class_to_idx
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader, test_loader
