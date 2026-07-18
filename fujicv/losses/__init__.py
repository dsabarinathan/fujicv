"""Loss functions for classification, regression, and multi-label tasks."""

# Import modules to trigger registration
import fujicv.losses.classification  # noqa: F401
import fujicv.losses.multilabel  # noqa: F401
import fujicv.losses.regression  # noqa: F401
from fujicv.losses.registry import LOSS_REGISTRY, get_loss, register_loss

__all__ = [
    "LOSS_REGISTRY",
    "register_loss",
    "get_loss",
]
