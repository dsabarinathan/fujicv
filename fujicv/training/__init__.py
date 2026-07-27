"""High-level training utilities (k-fold CV, EMA, schedulers, LLRD, SWA, LR Finder)."""

from fujicv.training.ema import ModelEMA
from fujicv.training.sam import SAM  # noqa: F401 — re-exported
from fujicv.training.kfold import KFoldTrainer
from fujicv.training.llrd import get_layer_wise_lr_params, print_llrd_summary
from fujicv.training.schedulers import cosine_with_warmup, get_scheduler, linear_warmup_schedule
from fujicv.training.swa import SWA, get_swa_scheduler
from fujicv.training.lr_finder import LRFinder

__all__ = [
    "KFoldTrainer",
    "ModelEMA",
    "get_scheduler",
    "cosine_with_warmup",
    "linear_warmup_schedule",
    "get_layer_wise_lr_params",
    "print_llrd_summary",
    "SAM",
    "SWA",
    "get_swa_scheduler",
    "LRFinder",
]
