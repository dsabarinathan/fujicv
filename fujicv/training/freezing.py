"""Layer freezing and gradual-unfreezing utilities for transfer learning.

A common fine-tuning recipe is to first train only the new head on top of a
frozen pretrained backbone, then progressively unfreeze the backbone from the
top down. These helpers make that pattern a few lines instead of manual
``requires_grad`` bookkeeping.

Example::

    from fujicv.training.freezing import freeze_backbone, GradualUnfreezing

    model = ModelBuilder("resnet50", task="classification", num_outputs=10).build()
    freeze_backbone(model)                     # train the head only, epoch 0

    unfreezer = GradualUnfreezing(model, unfreeze_epoch=2, layers_per_epoch=1)
    for epoch in range(epochs):
        unfreezer.step(epoch)                  # unfreeze one block from epoch 2 on
        train_one_epoch(...)
"""

from __future__ import annotations

import logging
from typing import List

import torch.nn as nn

logger = logging.getLogger(__name__)


def freeze(module: nn.Module) -> None:
    """Freeze every parameter in *module* (sets ``requires_grad=False``)."""
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze(module: nn.Module) -> None:
    """Unfreeze every parameter in *module* (sets ``requires_grad=True``)."""
    for p in module.parameters():
        p.requires_grad_(True)


def _resolve_backbone(model: nn.Module) -> nn.Module:
    """Return the backbone submodule of a FujiCV assembled model, else *model*."""
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        logger.warning(
            "freeze_backbone/unfreeze_backbone: model has no `.backbone` "
            "attribute; operating on the whole model instead."
        )
        return model
    return backbone


def freeze_backbone(model: nn.Module) -> None:
    """Freeze the backbone of a FujiCV model, leaving the head trainable.

    Works on any model exposing a ``.backbone`` attribute (as produced by
    :class:`~fujicv.models.builder.ModelBuilder`). If no ``.backbone`` is
    present the whole model is frozen and a warning is logged.
    """
    freeze(_resolve_backbone(model))


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze the backbone of a FujiCV model."""
    unfreeze(_resolve_backbone(model))


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of parameters with ``requires_grad=True``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_frozen_parameters(model: nn.Module) -> int:
    """Return the number of parameters with ``requires_grad=False``."""
    return sum(p.numel() for p in model.parameters() if not p.requires_grad)


def _top_level_children(module: nn.Module) -> List[nn.Module]:
    """Return direct child modules that actually own parameters."""
    return [child for child in module.children() if any(True for _ in child.parameters())]


class GradualUnfreezing:
    """Progressively unfreeze a backbone from the top (output) down.

    Call :meth:`step` once at the start of every epoch. Until
    ``unfreeze_epoch`` the backbone stays frozen (head-only warm-up). From
    ``unfreeze_epoch`` onward, ``layers_per_epoch`` backbone blocks are
    unfrozen each epoch, starting nearest the head, until the whole backbone
    is trainable.

    Args:
        model: A FujiCV model with a ``.backbone`` attribute (or any module —
            its top-level children are treated as the blocks).
        unfreeze_epoch: First epoch at which unfreezing begins (default 1).
        layers_per_epoch: Number of backbone blocks to unfreeze per epoch
            (default 1).
        freeze_on_init: If ``True`` (default), freeze the backbone immediately
            so epoch 0 trains the head only.

    Note:
        Newly unfrozen parameters already exist in the optimizer's param groups
        (their ``requires_grad`` simply flips to ``True``), so no optimizer
        rebuild is required as long as the optimizer was created over
        ``model.parameters()``.
    """

    def __init__(
        self,
        model: nn.Module,
        unfreeze_epoch: int = 1,
        layers_per_epoch: int = 1,
        freeze_on_init: bool = True,
    ) -> None:
        if unfreeze_epoch < 0:
            raise ValueError(f"unfreeze_epoch must be >= 0, got {unfreeze_epoch}")
        if layers_per_epoch < 1:
            raise ValueError(f"layers_per_epoch must be >= 1, got {layers_per_epoch}")

        self.model = model
        self.backbone = _resolve_backbone(model)
        self.unfreeze_epoch = unfreeze_epoch
        self.layers_per_epoch = layers_per_epoch

        # Blocks ordered top (near head) → bottom (near input).
        self._blocks: List[nn.Module] = list(reversed(_top_level_children(self.backbone)))
        self._num_unfrozen = 0

        if freeze_on_init:
            freeze(self.backbone)

    @property
    def num_blocks(self) -> int:
        """Total number of unfreezable backbone blocks."""
        return len(self._blocks)

    @property
    def fully_unfrozen(self) -> bool:
        """Return ``True`` once every backbone block is trainable."""
        return self._num_unfrozen >= len(self._blocks)

    def step(self, epoch: int) -> int:
        """Update which backbone blocks are trainable for *epoch*.

        Args:
            epoch: Current epoch (0-indexed).

        Returns:
            The total number of backbone blocks now unfrozen.
        """
        if epoch < self.unfreeze_epoch or self.fully_unfrozen:
            return self._num_unfrozen

        target = min(
            (epoch - self.unfreeze_epoch + 1) * self.layers_per_epoch,
            len(self._blocks),
        )
        while self._num_unfrozen < target:
            unfreeze(self._blocks[self._num_unfrozen])
            self._num_unfrozen += 1

        logger.info(
            "GradualUnfreezing: epoch %d — %d/%d backbone blocks trainable "
            "(%d trainable params)",
            epoch,
            self._num_unfrozen,
            len(self._blocks),
            count_trainable_parameters(self.model),
        )
        return self._num_unfrozen


def freeze_bn_stats(model: nn.Module, freeze: bool = True) -> None:
    """Freeze (or unfreeze) BatchNorm running-statistics updates.

    Useful when fine-tuning on a small dataset: keeping the pretrained BN
    statistics fixed often stabilises training. Puts every ``_BatchNorm``
    module into ``eval()`` mode (frozen) or ``train()`` mode (unfrozen)
    without touching parameter ``requires_grad``.

    Args:
        model: Model to modify.
        freeze: ``True`` to freeze BN stats, ``False`` to resume updating them.
    """
    from torch.nn.modules.batchnorm import _BatchNorm

    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.eval() if freeze else m.train()


__all__ = [
    "freeze",
    "unfreeze",
    "freeze_backbone",
    "unfreeze_backbone",
    "freeze_bn_stats",
    "count_trainable_parameters",
    "count_frozen_parameters",
    "GradualUnfreezing",
]
