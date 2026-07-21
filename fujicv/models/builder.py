"""High-level ModelBuilder that assembles backbone + optional layers + head."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from fujicv.models.backbone import build_backbone
from fujicv.models.custom_layers import LinearBNDropout
from fujicv.models.head import ClassificationHead, MultiLabelHead, RegressionHead

_TASK_HEADS = {
    "classification": ClassificationHead,
    "multiclass": ClassificationHead,
    "regression": RegressionHead,
    "multilabel": MultiLabelHead,
}

_VALID_TASKS = set(_TASK_HEADS.keys())


class _AssembledModel(nn.Module):
    """Internal assembled model: backbone → pooling → custom layers → head."""

    def __init__(
        self,
        backbone: nn.Module,
        arch_family: str,
        custom_layers: nn.Sequential,
        head: nn.Module,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.arch_family = arch_family
        self.custom_layers = custom_layers
        self.head = head
        self._pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)

        # Backbone can return tensors of varying shapes
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]  # take last stage

        if feats.dim() == 4:
            # CNN spatial features (B, C, H, W) → (B, C)
            feats = self._pool(feats).flatten(1)
        elif feats.dim() == 3:
            # ViT patch tokens (B, N, C) — take CLS or mean
            feats = feats[:, 0] if feats.shape[1] > 1 else feats.squeeze(1)
        # else already (B, C) — num_classes=0 timm models do global pooling

        feats = self.custom_layers(feats)
        return self.head(feats)


class ModelBuilder:
    """Assemble a backbone + optional intermediate layers + task head.

    Args:
        backbone_name: Model name passed to ``build_backbone``.
        backbone_source: ``'timm'`` (default) or ``'torchvision'``.
        pretrained: Load pretrained weights (default ``True``).
        custom_layers: List of dicts specifying extra layers, each dict::

            {"type": "LinearBNDropout", "out_features": 512, "dropout": 0.3}

            Supported types: ``"LinearBNDropout"``.
        task: One of ``'classification'``, ``'regression'``, ``'multilabel'``,
            ``'multiclass'``.
        num_outputs: Number of output neurons (classes / regression targets).
        head_kwargs: Extra keyword arguments forwarded to the head constructor.
        image_size: Spatial size used for the validation dummy forward pass
            (default 224).
    """

    def __init__(
        self,
        backbone_name: str,
        backbone_source: str = "timm",
        pretrained: bool = True,
        custom_layers: Optional[List[Dict[str, Any]]] = None,
        task: str = "classification",
        num_outputs: int = 2,
        head_kwargs: Optional[Dict[str, Any]] = None,
        image_size: int = 224,
        drop_path_rate: float = 0.0,
    ) -> None:
        if task not in _VALID_TASKS:
            raise ValueError(f"task must be one of {sorted(_VALID_TASKS)}, got {task!r}")

        self.backbone_name = backbone_name
        self.backbone_source = backbone_source
        self.pretrained = pretrained
        self.custom_layers_cfg = custom_layers or []
        self.task = task
        self.num_outputs = num_outputs
        self.head_kwargs = head_kwargs or {}
        self.image_size = image_size
        self.drop_path_rate = drop_path_rate

    def build(self) -> _AssembledModel:
        """Build and validate the assembled model.

        Runs a dummy forward pass ``torch.zeros(1, 3, image_size, image_size)``
        to verify that all shapes are compatible.

        Returns:
            An ``nn.Module`` ready for training.
        """
        bb = build_backbone(
            name=self.backbone_name,
            source=self.backbone_source,
            pretrained=self.pretrained,
            drop_path_rate=self.drop_path_rate if self.drop_path_rate > 0.0 else None,
        )
        backbone: nn.Module = bb["model"]
        out_features: int = bb["out_features"]
        arch_family: str = bb["arch_family"]

        # Build optional intermediate layers
        current_features = out_features
        layer_mods: list[nn.Module] = []
        for spec in self.custom_layers_cfg:
            layer_type = spec.get("type", "LinearBNDropout")
            if layer_type == "LinearBNDropout":
                layer_out = int(spec.get("out_features", current_features))
                dropout = float(spec.get("dropout", 0.3))
                layer_mods.append(LinearBNDropout(current_features, layer_out, dropout))
                current_features = layer_out
            else:
                raise ValueError(f"Unknown custom layer type: {layer_type!r}")

        custom_seq = nn.Sequential(*layer_mods)

        # Build head
        head_cls = _TASK_HEADS[self.task]
        if self.task in ("classification", "multiclass"):
            count_kwarg = {"num_classes": self.num_outputs}
        elif self.task == "regression":
            count_kwarg = {"num_outputs": self.num_outputs}
        else:
            count_kwarg = {"num_labels": self.num_outputs}
        head = head_cls(in_features=current_features, **count_kwarg, **self.head_kwargs)

        model = _AssembledModel(backbone, arch_family, custom_seq, head)

        # Validate with dummy forward pass
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            model(dummy)  # raises if shapes are incompatible
        model.train()
        return model
