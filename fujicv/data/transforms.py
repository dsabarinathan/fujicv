"""Albumentations-based transform pipelines."""

from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(
    image_size: int = 224,
    level: str = "medium",
) -> A.Compose:
    """Return an albumentations transform pipeline for training.

    Args:
        image_size: Target square image size (default 224).
        level: Augmentation level — ``'light'``, ``'medium'`` (default), or
            ``'heavy'``.

    Returns:
        An ``albumentations.Compose`` pipeline ending with Normalize + ToTensorV2.

    Raises:
        ValueError: If *level* is not recognised.
    """
    normalize = A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    if level == "light":
        transforms = [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            normalize,
            ToTensorV2(),
        ]
    elif level == "medium":
        transforms = [
            A.Resize(int(image_size * 1.1), int(image_size * 1.1)),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.GaussNoise(std_range=(0.04, 0.2), p=0.2),
            normalize,
            ToTensorV2(),
        ]
    elif level == "heavy":
        transforms = [
            A.Resize(int(image_size * 1.15), int(image_size * 1.15)),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            A.Rotate(limit=30, p=0.5),
            A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, scale=(0.8, 1.2), rotate=(-20, 20), p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4),
            A.GaussNoise(std_range=(0.04, 0.3), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.2),
            A.CoarseDropout(num_holes_range=(4, 8), hole_height_range=(0.1, 0.15), hole_width_range=(0.1, 0.15), p=0.3),
            A.GridDistortion(p=0.2),
            normalize,
            ToTensorV2(),
        ]
    else:
        raise ValueError(f"level must be 'light', 'medium', or 'heavy', got {level!r}")

    return A.Compose(transforms)


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """Return a deterministic transform pipeline for validation/test.

    Applies resize → centre crop → normalize → to tensor.

    Args:
        image_size: Target square image size (default 224).

    Returns:
        An ``albumentations.Compose`` pipeline.
    """
    return A.Compose(
        [
            A.Resize(int(image_size * 1.143), int(image_size * 1.143)),  # 256 for size=224
            A.CenterCrop(image_size, image_size),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )
