"""Evaluation utilities: reports, curves, t-SNE, attention maps."""

from fujicv.eval.attention_map import generate_attention_grid
from fujicv.eval.curves import plot_pr_curve, plot_roc_curve
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
]
