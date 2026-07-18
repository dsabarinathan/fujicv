"""Metrics for classification, regression, and multi-label tasks."""

import fujicv.metrics.classification  # noqa: F401
import fujicv.metrics.multilabel  # noqa: F401
import fujicv.metrics.regression  # noqa: F401
from fujicv.metrics.registry import METRIC_REGISTRY, get_metric, register_metric

__all__ = [
    "METRIC_REGISTRY",
    "register_metric",
    "get_metric",
]
