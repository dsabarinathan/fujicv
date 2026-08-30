"""Qualitative inspection plots: image grids, top-losses, class balance, confidence.

Complements the quantitative curves (loss/metric/ROC/PR) with the "look at your
data and your mistakes" visualisations that catch label noise, bad
augmentations, and systematic errors early.

All functions return a ``matplotlib.figure.Figure`` and never call ``show()``,
so they work headless (CI, notebooks, scripts).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

ArrayLike = Any  # torch.Tensor or np.ndarray


def _to_numpy(img: ArrayLike) -> np.ndarray:
    if hasattr(img, "detach"):  # torch.Tensor
        img = img.detach().cpu().numpy()
    return np.asarray(img)


def _to_displayable(
    img: ArrayLike,
    normalize: bool = True,
    mean: Sequence[float] = _IMAGENET_MEAN,
    std: Sequence[float] = _IMAGENET_STD,
) -> np.ndarray:
    """Convert a CHW/HWC/HW image (tensor or array) to HWC float in [0, 1]."""
    arr = _to_numpy(img).astype(np.float32)

    # CHW -> HWC when the first dim looks like channels.
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[2] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]  # single-channel -> HW grayscale

    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)

    looks_normalized = arr.min() < 0.0
    if normalize and arr.ndim == 3 and arr.shape[2] == 3 and looks_normalized:
        arr = arr * np.asarray(std) + np.asarray(mean)
    elif arr.max() > 1.5:  # 0-255 range
        arr = arr / 255.0

    return np.clip(arr, 0.0, 1.0)


def _grid_dims(n: int, ncols: int) -> tuple:
    ncols = max(1, min(ncols, n))
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def plot_image_grid(
    images: Union[Sequence[ArrayLike], ArrayLike],
    titles: Optional[Sequence[str]] = None,
    title_colors: Optional[Sequence[str]] = None,
    ncols: int = 4,
    figsize: Optional[tuple] = None,
    normalize: bool = True,
    suptitle: Optional[str] = None,
) -> plt.Figure:
    """Render a grid of images with optional per-image titles and title colours.

    Args:
        images: A batch tensor ``(N, C, H, W)``, an ``(N, H, W, C)`` array, or a
            sequence of individual images (tensors or arrays).
        titles: Optional per-image caption strings.
        title_colors: Optional per-image title colours (e.g. ``"green"`` /
            ``"red"`` for correct / incorrect).
        ncols: Number of columns.
        figsize: Figure size; auto-scaled from the grid when ``None``.
        normalize: Undo ImageNet normalisation for display (default ``True``).
        suptitle: Optional overall title.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    imgs = list(images)
    n = len(imgs)
    if n == 0:
        raise ValueError("plot_image_grid received no images.")
    nrows, ncols = _grid_dims(n, ncols)
    if figsize is None:
        figsize = (ncols * 2.6, nrows * 2.8)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < n:
            disp = _to_displayable(imgs[i], normalize=normalize)
            ax.imshow(disp, cmap="gray" if disp.ndim == 2 else None)
            if titles is not None and i < len(titles):
                color = title_colors[i] if title_colors is not None and i < len(title_colors) else "black"
                ax.set_title(titles[i], fontsize=9, color=color)
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    return fig


def plot_predictions(
    images: Union[Sequence[ArrayLike], ArrayLike],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[Sequence[str]] = None,
    confidences: Optional[Sequence[float]] = None,
    ncols: int = 4,
    max_images: int = 16,
    normalize: bool = True,
) -> plt.Figure:
    """Grid of predictions with green (correct) / red (incorrect) captions.

    Each caption reads ``pred (conf)`` over ``true: <label>``.
    """
    imgs = list(images)[:max_images]
    y_true = list(y_true)[:max_images]
    y_pred = list(y_pred)[:max_images]

    def name(i: int) -> str:
        return class_names[i] if class_names is not None else str(i)

    titles, colors = [], []
    for j in range(len(imgs)):
        t, p = int(y_true[j]), int(y_pred[j])
        conf = f"  {confidences[j]:.0%}" if confidences is not None and j < len(confidences) else ""
        titles.append(f"pred: {name(p)}{conf}\ntrue: {name(t)}")
        colors.append("green" if t == p else "red")

    return plot_image_grid(
        imgs, titles=titles, title_colors=colors, ncols=ncols,
        normalize=normalize, suptitle="Predictions (green=correct, red=wrong)",
    )


def plot_top_losses(
    images: Union[Sequence[ArrayLike], ArrayLike],
    losses: Sequence[float],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Optional[Sequence[str]] = None,
    confidences: Optional[Sequence[float]] = None,
    k: int = 9,
    ncols: int = 3,
    normalize: bool = True,
) -> plt.Figure:
    """Show the ``k`` highest-loss samples — the fastest way to find label noise
    and systematic errors (fast.ai's classic diagnostic).

    Args:
        images: Per-sample images (batch tensor / array / sequence).
        losses: Per-sample loss values (same order as *images*).
        y_true, y_pred: Per-sample true / predicted class indices.
        class_names: Optional class names.
        confidences: Optional per-sample predicted-class confidence.
        k: Number of worst samples to show.
        ncols: Grid columns.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    imgs = list(images)
    losses = np.asarray(losses, dtype=np.float32)
    if len(losses) != len(imgs):
        raise ValueError(f"losses ({len(losses)}) and images ({len(imgs)}) length mismatch.")

    k = min(k, len(imgs))
    order = np.argsort(-losses)[:k]  # highest loss first

    def name(i: int) -> str:
        return class_names[i] if class_names is not None else str(i)

    sel_imgs, titles, colors = [], [], []
    for idx in order:
        idx = int(idx)
        t, p = int(y_true[idx]), int(y_pred[idx])
        conf = f"  {confidences[idx]:.0%}" if confidences is not None else ""
        sel_imgs.append(imgs[idx])
        titles.append(f"loss={losses[idx]:.2f}\npred: {name(p)}{conf}\ntrue: {name(t)}")
        colors.append("green" if t == p else "red")

    return plot_image_grid(
        sel_imgs, titles=titles, title_colors=colors, ncols=ncols,
        normalize=normalize, suptitle=f"Top {k} losses",
    )


def plot_class_distribution(
    labels: Sequence[int],
    class_names: Optional[Sequence[str]] = None,
    splits: Optional[Dict[str, Sequence[int]]] = None,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """Bar chart of class frequencies — spot imbalance before training.

    Args:
        labels: Integer labels for a single split (ignored if *splits* given).
        class_names: Optional class names for the x-axis.
        splits: Optional ``{split_name: labels}`` to draw grouped bars for
            multiple splits (e.g. train vs val) side by side.
        figsize: Figure size.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    data = splits if splits is not None else {"data": labels}
    all_labels = np.concatenate([np.asarray(v).reshape(-1) for v in data.values()])
    if all_labels.size == 0:
        raise ValueError("plot_class_distribution received no labels.")
    n_classes = int(all_labels.max()) + 1
    xs = np.arange(n_classes)
    names = list(class_names) if class_names is not None else [str(i) for i in xs]

    fig, ax = plt.subplots(figsize=figsize)
    width = 0.8 / len(data)
    for j, (split_name, lab) in enumerate(data.items()):
        counts = np.bincount(np.asarray(lab).reshape(-1), minlength=n_classes)
        ax.bar(xs + j * width - 0.4 + width / 2, counts, width=width, label=split_name)

    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    if len(data) > 1:
        ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_confidence_histogram(
    confidences: Sequence[float],
    correct: Sequence[bool],
    bins: int = 20,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Overlaid confidence histograms for correct vs. incorrect predictions.

    A well-calibrated, well-separated model puts correct predictions at high
    confidence and errors at low confidence. Overlap in the middle is where a
    rejection threshold buys you the most.

    Args:
        confidences: Per-sample predicted-class probability.
        correct: Per-sample boolean (prediction was correct).
        bins: Histogram bins.
        figsize: Figure size.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    conf = np.asarray(confidences, dtype=np.float32).reshape(-1)
    ok = np.asarray(correct).reshape(-1).astype(bool)
    if conf.shape[0] != ok.shape[0]:
        raise ValueError("confidences and correct must be the same length.")

    edges = np.linspace(0.0, 1.0, bins + 1)
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(conf[ok], bins=edges, alpha=0.6, label=f"correct (n={int(ok.sum())})", color="green")
    ax.hist(conf[~ok], bins=edges, alpha=0.6, label=f"incorrect (n={int((~ok).sum())})", color="red")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _apply_transform(transform: Callable, img_np: np.ndarray) -> ArrayLike:
    """Apply an albumentations Compose or a plain callable to an HWC uint8 image."""
    try:
        out = transform(image=img_np)  # albumentations signature
        return out["image"] if isinstance(out, dict) else out
    except TypeError:
        return transform(img_np)       # plain callable (e.g. torchvision on array)


def plot_augmentations(
    image: ArrayLike,
    transform: Callable,
    n: int = 8,
    ncols: int = 4,
    normalize: bool = True,
) -> plt.Figure:
    """Visualise an augmentation pipeline by applying it ``n`` times to one image.

    Catches over-aggressive or label-breaking augmentations at a glance.

    Args:
        image: A source image (HWC uint8 array, or a CHW/HWC tensor).
        transform: An albumentations ``Compose`` (called as ``t(image=arr)``) or
            any callable ``t(img) -> image``.
        n: Number of augmented samples to draw.
        ncols: Grid columns.
        normalize: Undo ImageNet normalisation on the outputs for display.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    base = _to_numpy(image)
    if base.ndim == 3 and base.shape[0] in (1, 3) and base.shape[2] not in (1, 3):
        base = np.transpose(base, (1, 2, 0))  # CHW -> HWC for the transform
    if base.dtype != np.uint8 and base.max() <= 1.0:
        base = (np.clip(base, 0, 1) * 255).astype(np.uint8)

    augmented = [_apply_transform(transform, base) for _ in range(n)]
    return plot_image_grid(
        augmented, ncols=ncols, normalize=normalize, suptitle="Augmentation samples",
    )


__all__ = [
    "plot_image_grid",
    "plot_predictions",
    "plot_top_losses",
    "plot_class_distribution",
    "plot_confidence_histogram",
    "plot_augmentations",
]
