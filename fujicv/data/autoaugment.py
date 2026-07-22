"""RandAugment and augmentation policy utilities."""

from __future__ import annotations

import random
from typing import Callable, List, Optional, Tuple

import numpy as np


# ── Operation bank ─────────────────────────────────────────────────────────────

def _identity(img: np.ndarray, magnitude: float) -> np.ndarray:
    return img


def _auto_contrast(img: np.ndarray, magnitude: float) -> np.ndarray:
    try:
        from PIL import ImageOps
        from PIL import Image as _PIL
        pil = _PIL.fromarray(img)
        return np.array(ImageOps.autocontrast(pil))
    except ImportError:
        return img


def _equalize(img: np.ndarray, magnitude: float) -> np.ndarray:
    try:
        from PIL import ImageOps
        from PIL import Image as _PIL
        pil = _PIL.fromarray(img)
        return np.array(ImageOps.equalize(pil))
    except ImportError:
        return img


def _rotate(img: np.ndarray, magnitude: float) -> np.ndarray:
    try:
        from PIL import Image as _PIL
        angle = magnitude * 30.0
        if random.random() < 0.5:
            angle = -angle
        pil = _PIL.fromarray(img)
        return np.array(pil.rotate(angle, fillcolor=(128, 128, 128)))
    except ImportError:
        return img


def _solarize(img: np.ndarray, magnitude: float) -> np.ndarray:
    threshold = int(256 - magnitude * 256)
    return np.where(img < threshold, img, 255 - img).astype(np.uint8)


def _posterize(img: np.ndarray, magnitude: float) -> np.ndarray:
    bits = int(4 - magnitude * 4) + 1   # 1–4 bits
    shift = 8 - bits
    return ((img >> shift) << shift).astype(np.uint8)


def _sharpness(img: np.ndarray, magnitude: float) -> np.ndarray:
    try:
        from PIL import ImageEnhance
        from PIL import Image as _PIL
        factor = 1.0 + magnitude * 1.8 * (1 if random.random() < 0.5 else -1)
        pil = _PIL.fromarray(img)
        return np.array(ImageEnhance.Sharpness(pil).enhance(max(0.1, factor)))
    except ImportError:
        return img


def _color(img: np.ndarray, magnitude: float) -> np.ndarray:
    try:
        from PIL import ImageEnhance
        from PIL import Image as _PIL
        factor = 1.0 + magnitude * 1.8 * (1 if random.random() < 0.5 else -1)
        pil = _PIL.fromarray(img)
        return np.array(ImageEnhance.Color(pil).enhance(max(0.1, factor)))
    except ImportError:
        return img


def _brightness(img: np.ndarray, magnitude: float) -> np.ndarray:
    try:
        from PIL import ImageEnhance
        from PIL import Image as _PIL
        factor = 1.0 + magnitude * 1.8 * (1 if random.random() < 0.5 else -1)
        pil = _PIL.fromarray(img)
        return np.array(ImageEnhance.Brightness(pil).enhance(max(0.1, factor)))
    except ImportError:
        return img


def _contrast(img: np.ndarray, magnitude: float) -> np.ndarray:
    try:
        from PIL import ImageEnhance
        from PIL import Image as _PIL
        factor = 1.0 + magnitude * 1.8 * (1 if random.random() < 0.5 else -1)
        pil = _PIL.fromarray(img)
        return np.array(ImageEnhance.Contrast(pil).enhance(max(0.1, factor)))
    except ImportError:
        return img


def _shear_x(img: np.ndarray, magnitude: float) -> np.ndarray:
    try:
        from PIL import Image as _PIL
        shear = magnitude * 0.3 * (1 if random.random() < 0.5 else -1)
        pil = _PIL.fromarray(img)
        return np.array(pil.transform(
            pil.size, _PIL.AFFINE, (1, shear, 0, 0, 1, 0), fillcolor=(128, 128, 128)
        ))
    except ImportError:
        return img


def _shear_y(img: np.ndarray, magnitude: float) -> np.ndarray:
    try:
        from PIL import Image as _PIL
        shear = magnitude * 0.3 * (1 if random.random() < 0.5 else -1)
        pil = _PIL.fromarray(img)
        return np.array(pil.transform(
            pil.size, _PIL.AFFINE, (1, 0, 0, shear, 1, 0), fillcolor=(128, 128, 128)
        ))
    except ImportError:
        return img


def _translate_x(img: np.ndarray, magnitude: float) -> np.ndarray:
    try:
        from PIL import Image as _PIL
        pixels = int(magnitude * img.shape[1] * 0.33)
        if random.random() < 0.5:
            pixels = -pixels
        pil = _PIL.fromarray(img)
        return np.array(pil.transform(
            pil.size, _PIL.AFFINE, (1, 0, pixels, 0, 1, 0), fillcolor=(128, 128, 128)
        ))
    except ImportError:
        return img


def _translate_y(img: np.ndarray, magnitude: float) -> np.ndarray:
    try:
        from PIL import Image as _PIL
        pixels = int(magnitude * img.shape[0] * 0.33)
        if random.random() < 0.5:
            pixels = -pixels
        pil = _PIL.fromarray(img)
        return np.array(pil.transform(
            pil.size, _PIL.AFFINE, (1, 0, 0, 0, 1, pixels), fillcolor=(128, 128, 128)
        ))
    except ImportError:
        return img


_OPERATIONS: List[Tuple[str, Callable]] = [
    ("Identity",      _identity),
    ("AutoContrast",  _auto_contrast),
    ("Equalize",      _equalize),
    ("Rotate",        _rotate),
    ("Solarize",      _solarize),
    ("Posterize",     _posterize),
    ("Sharpness",     _sharpness),
    ("Color",         _color),
    ("Brightness",    _brightness),
    ("Contrast",      _contrast),
    ("ShearX",        _shear_x),
    ("ShearY",        _shear_y),
    ("TranslateX",    _translate_x),
    ("TranslateY",    _translate_y),
]


# ── RandAugment ────────────────────────────────────────────────────────────────

class RandAugment:
    """RandAugment (Cubuk et al., 2019) augmentation policy.

    Randomly selects *N* operations from a fixed bank and applies each
    with magnitude *M* (on a scale of 0–10).  Compatible with
    albumentations pipelines as an ``additional_targets`` transform or
    used standalone on numpy images.

    Args:
        n: Number of augmentations to apply (default 2).
        magnitude: Strength of each augmentation, 0–10 (default 9).
        magnitude_std: If > 0, sample magnitude from ``N(magnitude, std)``
            each call, clamped to [0, 10] (default 0.5).
        prob: Probability of applying RandAugment at all (default 1.0).

    Example (standalone)::

        from fujicv.data.autoaugment import RandAugment
        import numpy as np
        rand_aug = RandAugment(n=2, magnitude=9)
        aug_image = rand_aug(image_np)   # np.ndarray (H, W, 3) uint8

    Example (albumentations integration)::

        import albumentations as A
        from fujicv.data.autoaugment import RandAugmentTransform
        transform = A.Compose([
            A.Resize(224, 224),
            RandAugmentTransform(n=2, magnitude=9),
            A.Normalize(...),
        ])
    """

    def __init__(
        self,
        n: int = 2,
        magnitude: float = 9.0,
        magnitude_std: float = 0.5,
        prob: float = 1.0,
        ops: Optional[List[Tuple[str, Callable]]] = None,
    ) -> None:
        if not 0 <= magnitude <= 10:
            raise ValueError(f"magnitude must be in [0, 10], got {magnitude}")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        self.n             = n
        self.magnitude     = magnitude
        self.magnitude_std = magnitude_std
        self.prob          = prob
        self.ops           = ops or _OPERATIONS

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply RandAugment to a numpy image (H, W, 3) uint8."""
        if random.random() > self.prob:
            return image

        chosen = random.sample(self.ops, min(self.n, len(self.ops)))
        for name, op in chosen:
            if self.magnitude_std > 0:
                mag = float(np.clip(
                    np.random.normal(self.magnitude, self.magnitude_std), 0, 10
                ))
            else:
                mag = self.magnitude
            image = op(image, mag / 10.0)

        return image

    def __repr__(self) -> str:
        return f"RandAugment(n={self.n}, magnitude={self.magnitude})"


class RandAugmentTransform:
    """Albumentations-compatible wrapper for :class:`RandAugment`.

    Implements the albumentations ``__call__(image=...) -> dict`` interface
    so it can be dropped into any ``A.Compose`` pipeline.

    Example::

        import albumentations as A
        from fujicv.data.autoaugment import RandAugmentTransform

        transform = A.Compose([
            A.Resize(224, 224),
            RandAugmentTransform(n=2, magnitude=9),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            A.pytorch.ToTensorV2(),
        ])
    """

    def __init__(self, n: int = 2, magnitude: float = 9.0, **kwargs) -> None:
        self._rand_aug = RandAugment(n=n, magnitude=magnitude, **kwargs)

    def __call__(self, image: np.ndarray, **kwargs) -> dict:
        result = dict(kwargs)
        result["image"] = self._rand_aug(image)
        return result

    def __repr__(self) -> str:
        return repr(self._rand_aug)
