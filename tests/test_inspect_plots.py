"""Tests for qualitative inspection plots (headless / Agg backend)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless — must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from fujicv.eval.inspect import (  # noqa: E402
    plot_augmentations,
    plot_class_distribution,
    plot_confidence_histogram,
    plot_image_grid,
    plot_predictions,
    plot_top_losses,
)


def _batch(n=6, c=3, hw=16):
    return torch.randn(n, c, hw, hw)


def test_image_grid_from_batch_tensor():
    fig = plot_image_grid(_batch(6), ncols=3)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 6
    plt.close(fig)


def test_image_grid_grayscale_and_hwc():
    gray = [np.random.rand(16, 16) for _ in range(3)]          # HW
    hwc = [np.random.rand(16, 16, 3) for _ in range(2)]        # HWC
    fig = plot_image_grid(gray + hwc, ncols=2)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_image_grid_empty_raises():
    with pytest.raises(ValueError, match="no images"):
        plot_image_grid([])


def test_predictions_colors_correct_vs_wrong():
    imgs = _batch(4)
    fig = plot_predictions(
        imgs, y_true=[0, 1, 2, 0], y_pred=[0, 1, 0, 0],
        class_names=["a", "b", "c"], confidences=[0.9, 0.8, 0.6, 0.7], ncols=2,
    )
    # Collect title colors from the image axes.
    colors = [ax.title.get_color() for ax in fig.axes if ax.title.get_text()]
    assert "green" in colors and "red" in colors
    plt.close(fig)


def test_top_losses_selects_highest():
    imgs = _batch(10)
    losses = list(range(10))  # index 9 has the highest loss
    fig = plot_top_losses(
        imgs, losses=losses, y_true=[0] * 10, y_pred=[1] * 10, k=3, ncols=3,
    )
    assert isinstance(fig, plt.Figure)
    # First selected title should reference the largest loss (9.00).
    titles = [ax.title.get_text() for ax in fig.axes if ax.title.get_text()]
    assert any("loss=9.00" in t for t in titles)
    plt.close(fig)


def test_top_losses_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        plot_top_losses(_batch(4), losses=[1, 2], y_true=[0] * 4, y_pred=[0] * 4)


def test_class_distribution_single():
    labels = [0, 0, 1, 2, 2, 2]
    fig = plot_class_distribution(labels, class_names=["a", "b", "c"])
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_class_distribution_multiple_splits():
    fig = plot_class_distribution(
        labels=[], splits={"train": [0, 1, 1, 2], "val": [0, 0, 2]},
        class_names=["a", "b", "c"],
    )
    # Grouped bars → legend present.
    assert fig.axes[0].get_legend() is not None
    plt.close(fig)


def test_confidence_histogram():
    conf = np.random.rand(50)
    correct = np.random.rand(50) > 0.5
    fig = plot_confidence_histogram(conf, correct)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_confidence_histogram_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        plot_confidence_histogram([0.1, 0.2], [True])


def test_augmentations_with_albumentations_style_callable():
    # Mimic an albumentations Compose: called as t(image=arr) -> {"image": ...}
    def fake_aug(image):
        return {"image": image + np.random.randint(-5, 5, image.shape).astype(image.dtype)}

    img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    fig = plot_augmentations(img, fake_aug, n=4, ncols=2)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_augmentations_with_plain_callable():
    def plain(img):
        return np.clip(img.astype(np.int16) + 10, 0, 255).astype(np.uint8)

    img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    fig = plot_augmentations(img, plain, n=3, ncols=3)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)
