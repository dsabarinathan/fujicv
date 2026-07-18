"""Attention map and Grad-CAM visualisation."""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grad-CAM (CNN)
# ---------------------------------------------------------------------------

class _GradCAMHook:
    """Register forward + backward hooks for Grad-CAM."""

    def __init__(self, layer: nn.Module) -> None:
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._fwd_hook = layer.register_forward_hook(self._save_activations)
        self._bwd_hook = layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, _module, _inp, output):
        self.activations = output.detach()

    def _save_gradients(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def _find_last_conv(model: nn.Module) -> Optional[nn.Module]:
    """Return the last Conv2d layer in a model (DFS)."""
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    return last_conv


def _grad_cam(
    model: nn.Module,
    image: torch.Tensor,
    target_class: int,
    device: torch.device,
) -> np.ndarray:
    """Compute a Grad-CAM saliency map.

    Args:
        model: CNN model with head.
        image: Single image tensor of shape ``(1, C, H, W)``.
        target_class: Class index to compute gradients for.
        device: Device.

    Returns:
        Normalised saliency map as numpy array of shape ``(H, W)`` in [0, 1].
    """
    backbone = getattr(model, "backbone", model)
    last_conv = _find_last_conv(backbone)
    if last_conv is None:
        logger.warning("No Conv2d found in backbone; returning uniform map.")
        return np.ones((image.shape[2], image.shape[3]))

    hook = _GradCAMHook(last_conv)
    model.eval()
    image = image.to(device).requires_grad_(False)
    image.requires_grad_(False)

    logits = model(image)
    model.zero_grad()
    score = logits[0, target_class]
    score.backward()

    hook.remove()

    if hook.activations is None or hook.gradients is None:
        return np.ones((image.shape[2], image.shape[3]))

    weights = hook.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = (weights * hook.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam


# ---------------------------------------------------------------------------
# ViT attention rollout
# ---------------------------------------------------------------------------

def _attention_rollout(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Compute attention rollout from a ViT backbone.

    Handles timm ViT models that expose ``.blocks`` with ``.attn.attn_drop``.
    Falls back to a uniform map if the architecture is not supported.

    Returns:
        Saliency map as numpy array of shape ``(H, W)`` in [0, 1].
    """
    backbone = getattr(model, "backbone", model)
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        logger.warning("ViT backbone has no .blocks; returning uniform map.")
        h = w = int(math.sqrt(image.shape[2] * image.shape[3]))
        return np.ones((h, w))

    attention_matrices: List[torch.Tensor] = []
    hooks = []

    def _hook_fn(idx):
        def _fn(_m, _inp, output):
            attention_matrices.append(output.detach().cpu())
        return _fn

    # timm ViT: block.attn outputs attention weights when attn_drop forward is called
    # We hook the softmax output inside attn via the attn.attn_drop module
    for blk in blocks:
        attn_module = getattr(blk, "attn", None)
        if attn_module is None:
            continue
        # Try hooking attn_drop (gets called after softmax)
        attn_drop = getattr(attn_module, "attn_drop", None)
        if attn_drop is not None:
            hooks.append(attn_drop.register_forward_hook(_hook_fn(len(hooks))))

    model.eval()
    with torch.no_grad():
        model(image.to(device))

    for h in hooks:
        h.remove()

    if not attention_matrices:
        logger.warning("No attention weights captured; returning uniform map.")
        h = w = int(math.sqrt(image.shape[2] * image.shape[3]))
        return np.ones((h, w))

    # Rollout: multiply attention matrices through layers
    # attention_matrices: list of (B, heads, N, N)
    rollout = torch.eye(attention_matrices[0].shape[-1])
    for attn in attention_matrices:
        attn_mean = attn[0].mean(dim=0)  # (N, N)
        attn_mean = attn_mean + torch.eye(attn_mean.shape[0])
        attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(attn_mean, rollout)

    # CLS token row → patch attentions
    mask = rollout[0, 1:]  # exclude CLS itself → (N_patches,)
    n_patches = mask.shape[0]
    side = int(math.sqrt(n_patches))
    if side * side != n_patches:
        logger.warning("Non-square patch grid (%d patches); truncating.", n_patches)
        side = int(math.sqrt(n_patches))
        mask = mask[:side * side]
    mask = mask.reshape(side, side).numpy()
    mask -= mask.min()
    if mask.max() > 0:
        mask /= mask.max()
    return mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_attention_grid(
    model: nn.Module,
    dataloader: DataLoader,
    arch_family: str,
    n_correct: int = 8,
    n_wrong: int = 8,
    class_names: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> plt.Figure:
    """Generate a grid of images overlaid with attention / saliency maps.

    Selects *n_correct* correctly classified and *n_wrong* misclassified
    samples and overlays Grad-CAM (CNN) or attention rollout (ViT) maps.

    Args:
        model: Trained model (``_AssembledModel`` or compatible).
        dataloader: DataLoader yielding ``(image, label)`` batches.
        arch_family: ``'cnn'`` or ``'vit'``.
        n_correct: Number of correct examples to display.
        n_wrong: Number of wrong examples to display.
        class_names: Class name strings for title annotations.
        device: Target device string (defaults to CUDA if available).

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    model.eval()
    model.to(dev)

    correct_items: List[Tuple] = []
    wrong_items: List[Tuple] = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(dev)
            logits = model(images)
            preds = logits.argmax(dim=1)
            for i in range(len(images)):
                pred = preds[i].item()
                true = int(labels[i].item()) if hasattr(labels[i], "item") else int(labels[i])
                img_np = images[i].cpu().numpy().transpose(1, 2, 0)
                # Unnormalise roughly
                img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img_np = img_np.clip(0, 1)
                item = (images[i : i + 1], img_np, true, pred)
                if pred == true and len(correct_items) < n_correct:
                    correct_items.append(item)
                elif pred != true and len(wrong_items) < n_wrong:
                    wrong_items.append(item)
            if len(correct_items) >= n_correct and len(wrong_items) >= n_wrong:
                break

    all_items = correct_items + wrong_items
    n_total = len(all_items)
    if n_total == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No samples collected", ha="center", va="center")
        return fig

    ncols = min(8, n_total)
    nrows = math.ceil(n_total / ncols) * 2  # one row image, one row map
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.5))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    if ncols == 1:
        axes = axes[:, np.newaxis]

    for idx, (tensor, img_np, true, pred) in enumerate(all_items):
        row_base = (idx // ncols) * 2
        col = idx % ncols

        # Compute saliency
        if arch_family == "vit":
            saliency = _attention_rollout(model, tensor, dev)
        else:
            saliency = _grad_cam(model, tensor, true, dev)

        # Resize saliency to image size
        saliency_t = torch.tensor(saliency).unsqueeze(0).unsqueeze(0)
        h, w = img_np.shape[:2]
        saliency_resized = F.interpolate(
            saliency_t, size=(h, w), mode="bilinear", align_corners=False
        ).squeeze().numpy()

        # Image row
        ax_img = axes[row_base, col]
        ax_img.imshow(img_np)
        ax_img.axis("off")
        status = "✓" if true == pred else "✗"
        true_name = class_names[true] if class_names else str(true)
        pred_name = class_names[pred] if class_names else str(pred)
        color = "green" if true == pred else "red"
        ax_img.set_title(f"{status} T:{true_name}\nP:{pred_name}", fontsize=7, color=color)

        # Map row
        ax_map = axes[row_base + 1, col]
        ax_map.imshow(img_np)
        ax_map.imshow(saliency_resized, alpha=0.5, cmap="jet")
        ax_map.axis("off")

    # Hide empty axes
    for idx in range(n_total, nrows // 2 * ncols):
        row_base = (idx // ncols) * 2
        col = idx % ncols
        axes[row_base, col].axis("off")
        if row_base + 1 < nrows:
            axes[row_base + 1, col].axis("off")

    fig.suptitle(
        f"Attention Maps — {arch_family.upper()}  "
        f"(green=correct, red=wrong)",
        fontsize=11,
    )
    fig.tight_layout()
    return fig
