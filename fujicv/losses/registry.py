"""Loss function registry."""

from __future__ import annotations

from typing import Any, Dict

from fujicv.utils.registry import Registry

LOSS_REGISTRY = Registry("losses")
register_loss = LOSS_REGISTRY.register


def get_loss(name: str, kwargs: Dict[str, Any] | None = None):
    """Instantiate a registered loss by name.

    Args:
        name: Registered loss name.
        kwargs: Constructor keyword arguments (default ``{}``).

    Returns:
        An ``nn.Module`` loss instance.
    """
    kwargs = kwargs or {}
    loss_cls = LOSS_REGISTRY.get(name)
    return loss_cls(**kwargs)
