"""Inference utilities."""

from fujicv.inference.predictor import Predictor
from fujicv.inference.tta import TTAPredictor, tta_predict

__all__ = ["Predictor", "TTAPredictor", "tta_predict"]
