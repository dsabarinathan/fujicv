"""Evaluation utilities: reports, curves, t-SNE, attention maps."""

from fujicv.eval.attention_map import generate_attention_grid
from fujicv.eval.calibration import (  # noqa: F401
    TemperatureScaling,
    compute_ece,
    reliability_diagram,
)
from fujicv.eval.confusion import (  # noqa: F401 — re-exported
    per_class_metrics,
    plot_confusion_matrix,
)
from fujicv.eval.curves import plot_pr_curve, plot_roc_curve
from fujicv.eval.gradcam import (  # noqa: F401 — re-exported
    GradCAM,
    GradCAMPlusPlus,
    overlay_heatmap,
)
from fujicv.eval.plots import plot_loss_curves, plot_metric_curves
from fujicv.eval.report import classification_report
from fujicv.eval.tsne import extract_embeddings, plot_tsne

__all__ = [
    "classification_report",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_loss_curves",
    "plot_metric_curves",
    "plot_tsne",
    "extract_embeddings",
    "generate_attention_grid",
    "TemperatureScaling",
    "compute_ece",
    "reliability_diagram",
    "plot_confusion_matrix",
    "per_class_metrics",
    "GradCAM",
    "GradCAMPlusPlus",
    "overlay_heatmap",
]
