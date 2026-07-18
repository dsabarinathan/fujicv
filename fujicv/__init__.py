"""FujiCV — image classification and regression built on timm + torchvision."""

__version__ = "0.1.0"
__author__ = "FujiCV Contributors"

from fujicv.data.datasets import get_default_dataset  # noqa: F401
from fujicv.utils.registry import Registry  # noqa: F401
from fujicv.utils.seed import get_device, set_seed  # noqa: F401

__all__ = ["__version__", "__author__", "get_device", "set_seed", "get_default_dataset"]
