"""Metric registry."""

from __future__ import annotations

from typing import Any, Dict

from fujicv.utils.registry import Registry

METRIC_REGISTRY = Registry("metrics")
register_metric = METRIC_REGISTRY.register


def get_metric(name: str, kwargs: Dict[str, Any] | None = None):
    """Instantiate a registered metric by name.

    Args:
        name: Registered metric name.
        kwargs: Constructor keyword arguments (default ``{}``).

    Returns:
        A callable metric instance.
    """
    kwargs = kwargs or {}
    metric_cls = METRIC_REGISTRY.get(name)
    return metric_cls(**kwargs)
