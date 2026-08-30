"""Backbone factory — wraps timm and torchvision model zoos."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

_VIT_PATTERNS = re.compile(r"vit|swin|deit|beit|coatnet|pit|tnt|cait|xcit|twins")


def _infer_arch_family(name: str) -> str:
    """Return 'vit' for transformer-based architectures, else 'cnn'."""
    return "vit" if _VIT_PATTERNS.search(name.lower()) else "cnn"


def _build_timm_backbone(
    name: str,
    pretrained: bool,
    in_chans: int,
    features_only: bool,
    out_indices: Optional[List[int]],
    drop_path_rate: Optional[float] = None,
) -> Dict[str, Any]:
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required for source='timm'. Install with: pip install timm") from exc

    kwargs: Dict[str, Any] = dict(
        pretrained=pretrained,
        num_classes=0,  # remove classifier head
        in_chans=in_chans,
    )
    if features_only:
        kwargs["features_only"] = True
        if out_indices is not None:
            kwargs["out_indices"] = out_indices
    if drop_path_rate is not None and drop_path_rate > 0.0:
        kwargs["drop_path_rate"] = drop_path_rate

    model = timm.create_model(name, **kwargs)

    # Determine output feature dimension
    if features_only:
        # features_only models return list; take last stage
        dummy = torch.zeros(1, in_chans, 224, 224)
        with torch.no_grad():
            outs = model(dummy)
        if isinstance(outs, (list, tuple)):
            out_features = outs[-1].shape[1]
        else:
            out_features = outs.shape[1]
    else:
        # num_classes=0 → model.num_features
        out_features = model.num_features

    return {
        "model": model,
        "out_features": out_features,
        "arch_family": _infer_arch_family(name),
    }


def _resolve_vision_encoder(model: nn.Module) -> nn.Module:
    """Return the vision tower of a (possibly multimodal) transformers model.

    CLIP / SigLIP / BLIP and similar load as multimodal models whose ``forward``
    needs both ``input_ids`` and ``pixel_values``. Their image encoder lives in a
    ``vision_model`` (or ``vision_tower`` / ``visual``) submodule that accepts
    ``pixel_values`` alone. Plain vision models (ViT, DINOv2, ConvNeXt, …) have no
    such attribute and are returned unchanged.
    """
    for attr in ("vision_model", "vision_tower", "visual"):
        sub = getattr(model, attr, None)
        if isinstance(sub, nn.Module):
            return sub
    return model


class _HFBackboneWrapper(nn.Module):
    """Adapt a ``transformers`` vision encoder to FujiCV's tensor-in/tensor-out API.

    Forwards ``pixel_values`` to the (already vision-tower-resolved) encoder and
    returns a feature tensor: ``(B, N, C)`` patch tokens (ViT/DINOv2/CLIP/SigLIP),
    ``(B, C, H, W)`` maps (ConvNeXt/ResNet), or ``(B, C)`` if only a pooled vector
    is available — which the assembled model then pools uniformly.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        # Prefer the full feature map/sequence; fall back to a pooled vector,
        # then to a raw tensor/tuple for exotic encoders.
        feats = getattr(out, "last_hidden_state", None)
        if feats is None:
            feats = getattr(out, "pooler_output", None)
        if feats is None:
            feats = out[0] if isinstance(out, (tuple, list)) else out
        return feats


def _build_hf_backbone(
    name: str,
    pretrained: bool,
    in_chans: int,
    image_size: int = 224,
) -> Dict[str, Any]:
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise ImportError(
            "transformers is required for source='hf'. "
            'Install with: pip install "fujicv[hf-models]"  (or: pip install transformers)'
        ) from exc

    if in_chans != 3:
        raise ValueError(
            f"source='hf' backbones support 3-channel input only, got in_chans={in_chans}."
        )

    if pretrained:
        model = AutoModel.from_pretrained(name)
    else:
        config = AutoConfig.from_pretrained(name)
        model = AutoModel.from_config(config)

    encoder = _resolve_vision_encoder(model)  # unwrap CLIP/SigLIP vision towers
    wrapper = _HFBackboneWrapper(encoder)

    # Infer output width with a dummy forward (robust across ViT/CNN outputs).
    wrapper.eval()
    with torch.no_grad():
        feats = wrapper(torch.zeros(1, in_chans, image_size, image_size))
    if feats.dim() == 3:          # (B, N, C) transformer tokens
        out_features = feats.shape[-1]
        arch_family = "vit"
    elif feats.dim() == 4:        # (B, C, H, W) feature map
        out_features = feats.shape[1]
        arch_family = "cnn"
    else:                          # (B, C) already pooled
        out_features = feats.shape[-1]
        arch_family = _infer_arch_family(name)
    wrapper.train()

    return {"model": wrapper, "out_features": out_features, "arch_family": arch_family}


def _build_torchvision_backbone(
    name: str,
    pretrained: bool,
) -> Dict[str, Any]:
    try:
        import torchvision.models as tv_models
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for source='torchvision'. Install with: pip install torchvision"
        ) from exc

    weights = "DEFAULT" if pretrained else None
    model = tv_models.get_model(name, weights=weights)

    # Strip the final classification layer(s)
    out_features: int
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        out_features = model.fc.in_features
        model.fc = nn.Identity()
    elif hasattr(model, "classifier"):
        classifier = model.classifier
        # Find the last Linear inside classifier
        if isinstance(classifier, nn.Linear):
            out_features = classifier.in_features
            model.classifier = nn.Identity()
        elif isinstance(classifier, nn.Sequential):
            last_linear: Optional[nn.Linear] = None
            for layer in classifier:
                if isinstance(layer, nn.Linear):
                    last_linear = layer
            if last_linear is not None:
                out_features = last_linear.in_features
                # Replace only the last linear
                layers = list(classifier.children())
                new_seq = nn.Sequential(*layers[:-1])
                model.classifier = new_seq
            else:
                raise ValueError(f"Cannot find a Linear layer in classifier for {name}")
        else:
            raise ValueError(f"Unsupported classifier type {type(classifier)} for {name}")
    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        out_features = model.head.in_features
        model.head = nn.Identity()
    elif hasattr(model, "heads"):
        # e.g. ViT-B/16 in torchvision
        head = model.heads
        if hasattr(head, "head") and isinstance(head.head, nn.Linear):
            out_features = head.head.in_features
            head.head = nn.Identity()
        else:
            raise ValueError(f"Cannot strip head for torchvision model {name}")
    else:
        raise ValueError(
            f"Don't know how to strip the classifier from torchvision model '{name}'. "
            "Supported models expose .fc, .classifier, or .head attributes."
        )

    return {
        "model": model,
        "out_features": out_features,
        "arch_family": _infer_arch_family(name),
    }


def build_backbone(
    name: str,
    source: str = "timm",
    pretrained: bool = True,
    in_chans: int = 3,
    features_only: bool = False,
    out_indices: Optional[List[int]] = None,
    drop_path_rate: Optional[float] = None,
    image_size: int = 224,
) -> Dict[str, Any]:
    """Build a backbone model from timm, torchvision, or Hugging Face.

    Args:
        name: Model name (e.g. ``'resnet50'``, ``'vit_tiny_patch16_224'``, or a
            HF repo id like ``'google/vit-base-patch16-224'``).
        source: Model zoo — ``'timm'`` (default), ``'torchvision'``, or
            ``'hf'`` (Hugging Face ``transformers``).
        pretrained: Load pretrained ImageNet weights when ``True``.
        in_chans: Number of input channels (default 3; ``'hf'`` requires 3).
        features_only: (timm only) Return intermediate feature maps.
        out_indices: (timm only) Which feature stages to return.
        image_size: Spatial size for the HF output-width probe (default 224).

    Returns:
        Dict with keys:

        * ``model`` — the ``nn.Module`` backbone (classifier head removed).
        * ``out_features`` — integer channel width of the last feature map.
        * ``arch_family`` — ``'cnn'`` or ``'vit'``.

    Raises:
        ValueError: If *source* is not recognised.
    """
    source = source.lower()
    if source == "timm":
        return _build_timm_backbone(name, pretrained, in_chans, features_only, out_indices, drop_path_rate)
    elif source == "torchvision":
        if features_only:
            raise ValueError("features_only is not supported for source='torchvision'")
        return _build_torchvision_backbone(name, pretrained)
    elif source in ("hf", "huggingface", "transformers"):
        if features_only:
            raise ValueError("features_only is not supported for source='hf'")
        return _build_hf_backbone(name, pretrained, in_chans, image_size=image_size)
    else:
        raise ValueError(
            f"Unknown source {source!r}. Choose 'timm', 'torchvision', or 'hf'."
        )


def list_available_backbones(
    source: str = "timm",
    filter: Optional[str] = None,
) -> List[str]:
    """List available backbone model names.

    Args:
        source: ``'timm'`` or ``'torchvision'``.
        filter: Substring filter applied to model names (case-insensitive).

    Returns:
        Sorted list of model name strings.
    """
    source = source.lower()
    if source == "timm":
        import timm

        names = timm.list_models()
    elif source == "torchvision":
        import torchvision.models as tv_models

        names = tv_models.list_models()
    else:
        raise ValueError(f"Unknown source {source!r}. Choose 'timm' or 'torchvision'.")

    if filter:
        names = [n for n in names if filter.lower() in n.lower()]

    return sorted(names)
