"""Grad-CAM and Grad-CAM++ saliency map generation."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Generate Grad-CAM saliency maps for any CNN layer.

    Grad-CAM (Selvaraju et al., 2017) computes a class-discriminative
    localization map by weighting each feature channel by its gradient
    with respect to the target class score, then applying ReLU.

    Args:
        model: Trained ``nn.Module``.
        target_layer: The convolutional layer to hook (e.g.
            ``model.layer4[-1]`` for ResNet).  Must produce spatial feature
            maps ``(B, C, H, W)``.
        use_cuda: Run on GPU if available (default: auto).

    Example::

        from fujicv.eval.gradcam import GradCAM, overlay_heatmap
        import cv2, numpy as np

        cam = GradCAM(model, target_layer=model.layer4[-1])
        image = np.array(Image.open("cat.jpg"))
        heatmap = cam.generate(image)            # (H, W) float32 in [0, 1]
        result  = overlay_heatmap(image, heatmap)
        cam.remove_hooks()
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        use_cuda: Optional[bool] = None,
    ) -> None:
        self.model = model
        self.target_layer = target_layer

        if use_cuda is None:
            use_cuda = next(model.parameters()).is_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = self.model.to(self.device).eval()

        self._features: Optional[torch.Tensor] = None
        self._grads:    Optional[torch.Tensor] = None
        self._hooks: List = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def _fwd_hook(module, input, output):
            self._features = output.detach()

        def _bwd_hook(module, grad_input, grad_output):
            self._grads = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(_fwd_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(_bwd_hook))

    def remove_hooks(self) -> None:
        """Remove forward/backward hooks. Call when done."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(
        self,
        image: Union[np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
        input_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """Compute a Grad-CAM heatmap for *image*.

        Args:
            image: RGB image as ``np.ndarray`` (H, W, 3) uint8 or float32, or
                a pre-processed tensor ``(1, C, H, W)``.
            target_class: Class index to explain.  If ``None``, uses the
                predicted class.
            input_size: Resize image to this size before inference
                (ignored when a tensor is supplied).

        Returns:
            Heatmap ``np.ndarray`` of shape ``(H, W)`` with values in ``[0, 1]``,
            resized to match the input image spatial dimensions.
        """
        tensor, orig_h, orig_w = self._preprocess(image, input_size)
        tensor = tensor.to(self.device)
        tensor.requires_grad_(False)

        self.model.zero_grad()
        logits = self.model(tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[0, target_class]
        self.model.zero_grad()
        score.backward()

        # Global average pool of gradients → weights
        weights = self._grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self._features).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam     = F.relu(cam)

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        denom = cam.max()
        if denom > 0:
            cam = cam / denom

        # Resize back to original image size
        cam = self._resize(cam, orig_h, orig_w)
        return cam.astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess(
        self, image: Union[np.ndarray, torch.Tensor], input_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, int, int]:
        if isinstance(image, torch.Tensor):
            if image.ndim == 3:
                image = image.unsqueeze(0)
            return image, image.shape[-2], image.shape[-1]

        import cv2
        orig_h, orig_w = image.shape[:2]
        img = cv2.resize(image, (input_size[1], input_size[0]))
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        img  = (img - mean) / std
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        return tensor, orig_h, orig_w

    @staticmethod
    def _resize(cam: np.ndarray, h: int, w: int) -> np.ndarray:
        try:
            import cv2
            return cv2.resize(cam, (w, h))
        except ImportError:
            from PIL import Image
            return np.array(
                Image.fromarray(cam).resize((w, h), Image.BILINEAR)
            )


class GradCAMPlusPlus(GradCAM):
    """Grad-CAM++ (Chattopadhyay et al., 2018) — improved localization.

    Uses second-order gradients to weight channels more accurately,
    especially when multiple instances of the same class are present.

    Same interface as :class:`GradCAM`.
    """

    def generate(
        self,
        image: Union[np.ndarray, torch.Tensor],
        target_class: Optional[int] = None,
        input_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        tensor, orig_h, orig_w = self._preprocess(image, input_size)
        tensor = tensor.to(self.device)

        self.model.zero_grad()
        logits = self.model(tensor)

        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[0, target_class]
        self.model.zero_grad()
        score.backward()

        grads    = self._grads                           # (1, C, h, w)
        features = self._features                        # (1, C, h, w)

        # Grad-CAM++ weights
        grads_sq  = grads ** 2
        grads_cu  = grads ** 3
        denom     = 2 * grads_sq + features * grads_cu
        denom     = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha     = grads_sq / denom
        weights   = (alpha * F.relu(grads)).mean(dim=(2, 3), keepdim=True)

        cam = (weights * features).sum(dim=1, keepdim=True)
        cam = F.relu(cam).squeeze().cpu().numpy()
        cam = cam - cam.min()
        denom_v = cam.max()
        if denom_v > 0:
            cam = cam / denom_v

        cam = self._resize(cam, orig_h, orig_w)
        return cam.astype(np.float32)


# ── Visualization helper ───────────────────────────────────────────────────────

def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = None,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an RGB image.

    Args:
        image: RGB ``np.ndarray`` (H, W, 3) uint8 or float32 in [0, 1].
        heatmap: Float heatmap (H, W) in [0, 1] from :meth:`GradCAM.generate`.
        alpha: Blend weight for the heatmap (default 0.5).
        colormap: OpenCV colormap constant (default ``cv2.COLORMAP_JET``).

    Returns:
        Blended RGB ``np.ndarray`` (H, W, 3) uint8.
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError("opencv-python is required for overlay_heatmap") from e

    if colormap is None:
        colormap = cv2.COLORMAP_JET

    if image.dtype != np.uint8:
        image = (image * 255).clip(0, 255).astype(np.uint8)

    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    colored    = cv2.applyColorMap(heatmap_u8, colormap)
    colored    = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    blended = (alpha * colored + (1 - alpha) * image).clip(0, 255).astype(np.uint8)
    return blended
