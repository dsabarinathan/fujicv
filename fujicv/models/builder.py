"""High-level ModelBuilder that assembles backbone + optional layers + head."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from fujicv.models.backbone import build_backbone
from fujicv.models.custom_layers import AttentionPool, GeM, LinearBNDropout
from fujicv.models.head import ClassificationHead, MultiLabelHead, RegressionHead

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "leakyrelu": nn.LeakyReLU,
}

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
        pool: Optional[nn.Module] = None,
        pool_type: str = "avg",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.arch_family = arch_family
        self.custom_layers = custom_layers
        self.head = head
        self.pool_type = pool_type
        self._pool = pool if pool is not None else nn.AdaptiveAvgPool2d(1)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pooled, pre-head embedding ``(B, D)``.

        This is the representation fed to the task head — the right vector to
        use for retrieval, kNN, or clustering. See
        :class:`fujicv.retrieval.Embedder`.
        """
        feats = self.backbone(x)

        # Backbone can return tensors of varying shapes
        if isinstance(feats, (list, tuple)):
            feats = feats[-1]  # take last stage

        if feats.dim() == 4:
            feats = self._pool_spatial(feats)
        elif feats.dim() == 3:
            feats = self._pool_tokens(feats)
        # else already (B, C) — timm with num_classes=0 does pooling internally

        return self.custom_layers(feats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def _pool_spatial(self, feats: torch.Tensor) -> torch.Tensor:
        """Pool CNN spatial features ``(B, C, H, W)`` → ``(B, C)``."""
        if self.pool_type in ("gem", "attention"):
            return self._pool(feats)  # GeM / AttentionPool already return (B, C)
        return self._pool(feats).flatten(1)  # avg / max adaptive pool → flatten

    def _pool_tokens(self, feats: torch.Tensor) -> torch.Tensor:
        """Pool transformer patch tokens ``(B, N, C)`` → ``(B, C)``.

        Mean over the sequence is architecture-agnostic (avoids hard-coding CLS
        position). Attention pooling uses the learned attention module.
        """
        if self.pool_type == "attention":
            return self._pool(feats)
        return feats.mean(dim=1)


class ModelBuilder:
    """Assemble a backbone + optional intermediate layers + task head.

    Args:
        backbone_name: Model name passed to ``build_backbone`` (a timm/torchvision
            name, or a Hugging Face repo id when ``backbone_source='hf'``).
        backbone_source: ``'timm'`` (default), ``'torchvision'``, or ``'hf'``.
        pretrained: Load pretrained weights (default ``True``).
        custom_layers: List of dicts specifying extra layers, each dict::

            {"type": "LinearBNDropout", "out_features": 512, "dropout": 0.3}

            Supported types: ``"LinearBNDropout"``, ``"Linear"``, ``"Dropout"``,
            ``"LayerNorm"``, ``"BatchNorm1d"``, ``"Activation"`` (with a ``"fn"``
            key: relu/gelu/silu/tanh/leakyrelu).
        task: One of ``'classification'``, ``'regression'``, ``'multilabel'``,
            ``'multiclass'``.
        num_outputs: Number of output neurons (classes / regression targets).
        head_kwargs: Extra keyword arguments forwarded to the head constructor.
        image_size: Spatial size used for the validation dummy forward pass
            (default 224).
        pooling: Feature pooling over backbone spatial maps — ``'avg'``
            (default), ``'max'``, ``'gem'`` (learnable Generalised-Mean, great
            for retrieval), or ``'attention'`` (learned attention pool).
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
        pooling: str = "avg",
    ) -> None:
        if task not in _VALID_TASKS:
            raise ValueError(f"task must be one of {sorted(_VALID_TASKS)}, got {task!r}")
        if pooling not in ("avg", "max", "gem", "attention"):
            raise ValueError(
                f"pooling must be 'avg', 'max', 'gem', or 'attention', got {pooling!r}"
            )

        self.backbone_name = backbone_name
        self.backbone_source = backbone_source
        self.pretrained = pretrained
        self.custom_layers_cfg = custom_layers or []
        self.task = task
        self.num_outputs = num_outputs
        self.head_kwargs = head_kwargs or {}
        self.image_size = image_size
        self.drop_path_rate = drop_path_rate
        self.pooling = pooling

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
            image_size=self.image_size,
        )
        backbone: nn.Module = bb["model"]
        out_features: int = bb["out_features"]
        arch_family: str = bb["arch_family"]

        # Build optional intermediate (head) layers. These run on the pooled
        # (B, C) feature vector, so they are all 1-D building blocks.
        current_features = out_features
        layer_mods: list[nn.Module] = []
        for spec in self.custom_layers_cfg:
            layer_type = spec.get("type", "LinearBNDropout")
            if layer_type == "LinearBNDropout":
                layer_out = int(spec.get("out_features", current_features))
                dropout = float(spec.get("dropout", 0.3))
                layer_mods.append(LinearBNDropout(current_features, layer_out, dropout))
                current_features = layer_out
            elif layer_type == "Linear":
                layer_out = int(spec.get("out_features", current_features))
                bias = bool(spec.get("bias", True))
                layer_mods.append(nn.Linear(current_features, layer_out, bias=bias))
                current_features = layer_out
            elif layer_type == "Dropout":
                layer_mods.append(nn.Dropout(p=float(spec.get("p", 0.5))))
            elif layer_type == "LayerNorm":
                layer_mods.append(nn.LayerNorm(current_features))
            elif layer_type == "BatchNorm1d":
                layer_mods.append(nn.BatchNorm1d(current_features))
            elif layer_type == "Activation":
                fn = str(spec.get("fn", "relu")).lower()
                if fn not in _ACTIVATIONS:
                    raise ValueError(
                        f"Unknown activation {fn!r}. Choose from {sorted(_ACTIVATIONS)}."
                    )
                layer_mods.append(_ACTIVATIONS[fn]())
            else:
                raise ValueError(f"Unknown custom layer type: {layer_type!r}")

        custom_seq = nn.Sequential(*layer_mods)

        # Feature pooling module (applied to 4-D backbone maps).
        if self.pooling == "avg":
            pool: nn.Module = nn.AdaptiveAvgPool2d(1)
        elif self.pooling == "max":
            pool = nn.AdaptiveMaxPool2d(1)
        elif self.pooling == "gem":
            pool = GeM()
        else:  # attention
            pool = AttentionPool(out_features)

        # Build head
        head_cls = _TASK_HEADS[self.task]
        if self.task in ("classification", "multiclass"):
            count_kwarg = {"num_classes": self.num_outputs}
        elif self.task == "regression":
            count_kwarg = {"num_outputs": self.num_outputs}
        else:
            count_kwarg = {"num_labels": self.num_outputs}
        head = head_cls(in_features=current_features, **count_kwarg, **self.head_kwargs)

        model = _AssembledModel(
            backbone, arch_family, custom_seq, head, pool=pool, pool_type=self.pooling
        )

        # Validate with dummy forward pass
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size)
            model(dummy)  # raises if shapes are incompatible
        model.train()
        return model
