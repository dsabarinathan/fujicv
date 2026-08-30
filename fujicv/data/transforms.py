"""Albumentations-based transform pipelines."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ImageNet statistics
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(
    image_size: int = 224,
    level: str = "medium",
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> A.Compose:
    """Return an albumentations transform pipeline for training.

    Args:
        image_size: Target square image size (default 224).
        level: Augmentation level — ``'light'``, ``'medium'`` (default), or
            ``'heavy'``.
        mean: Per-channel normalization mean (default ImageNet). Override to
            match a specific pretrained encoder (e.g. CLIP / SigLIP).
        std: Per-channel normalization std (default ImageNet).

    Returns:
        An ``albumentations.Compose`` pipeline ending with Normalize + ToTensorV2.

    Raises:
        ValueError: If *level* is not recognised.
    """
    normalize = A.Normalize(mean=mean or _IMAGENET_MEAN, std=std or _IMAGENET_STD)

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


def get_val_transforms(
    image_size: int = 224,
    mean: Optional[Sequence[float]] = None,
    std: Optional[Sequence[float]] = None,
) -> A.Compose:
    """Return a deterministic transform pipeline for validation/test.

    Applies resize → centre crop → normalize → to tensor.

    Args:
        image_size: Target square image size (default 224).
        mean: Per-channel normalization mean (default ImageNet).
        std: Per-channel normalization std (default ImageNet).

    Returns:
        An ``albumentations.Compose`` pipeline.
    """
    return A.Compose(
        [
            A.Resize(int(image_size * 1.143), int(image_size * 1.143)),  # 256 for size=224
            A.CenterCrop(image_size, image_size),
            A.Normalize(mean=mean or _IMAGENET_MEAN, std=std or _IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def _resolve_processor_size(size: Any, default: int = 224) -> int:
    """Extract a single square image size from a HF processor's ``size`` field.

    Handles the several shapes HF uses: ``{"height": H, "width": W}``,
    ``{"shortest_edge": S}``, a bare int, or ``None``.
    """
    if isinstance(size, int):
        return size
    if isinstance(size, dict):
        if "height" in size:
            return int(size["height"])
        if "shortest_edge" in size:
            return int(size["shortest_edge"])
    return default


def get_hf_transforms(
    model_name: str,
    train: bool = False,
    level: str = "medium",
    image_size: Optional[int] = None,
) -> A.Compose:
    """Build transforms that match a Hugging Face encoder's own image processor.

    Reads ``AutoImageProcessor`` to recover the exact normalization statistics
    and input size the pretrained encoder expects. Using the wrong mean/std with
    a model like CLIP or SigLIP badly degrades accuracy, so pair this with
    ``ModelBuilder(backbone_source="hf", ...)``.

    Args:
        model_name: A Hugging Face repo id (e.g. ``"google/siglip-base-patch16-224"``).
        train: Return the augmented training pipeline (default: deterministic val).
        level: Augmentation level for the training pipeline.
        image_size: Override the processor's inferred input size.

    Returns:
        An ``albumentations.Compose`` pipeline.

    Example::

        from fujicv.data.transforms import get_hf_transforms
        from fujicv.models.builder import ModelBuilder

        name = "facebook/dinov2-base"
        train_tf = get_hf_transforms(name, train=True)
        val_tf   = get_hf_transforms(name, train=False)
        model = ModelBuilder(name, backbone_source="hf",
                             task="classification", num_outputs=10).build()
    """
    try:
        from transformers import AutoImageProcessor
    except ImportError as exc:
        raise ImportError(
            "transformers is required for get_hf_transforms. "
            'Install with: pip install "fujicv[hf-models]"'
        ) from exc

    proc = AutoImageProcessor.from_pretrained(model_name)
    mean: Tuple[float, ...] = tuple(getattr(proc, "image_mean", None) or _IMAGENET_MEAN)
    std: Tuple[float, ...] = tuple(getattr(proc, "image_std", None) or _IMAGENET_STD)
    size = image_size or _resolve_processor_size(getattr(proc, "size", None))

    if train:
        return get_train_transforms(size, level=level, mean=mean, std=std)
    return get_val_transforms(size, mean=mean, std=std)
