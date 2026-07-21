"""Model calibration utilities: Temperature Scaling, ECE, reliability diagram."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ── Expected Calibration Error ─────────────────────────────────────────────────

def compute_ece(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Args:
        confidences: 1-D array of predicted confidence (max softmax prob).
        correct: 1-D boolean / int array — 1 if prediction was correct.
        n_bins: Number of equal-width bins (default 15).

    Returns:
        ECE as a float in [0, 1] (lower is better; 0 = perfectly calibrated).

    Example::

        from fujicv.eval.calibration import compute_ece
        probs   = softmax(logits, axis=1)
        conf    = probs.max(axis=1)
        correct = (probs.argmax(axis=1) == targets)
        ece     = compute_ece(conf, correct)
    """
    confidences = np.asarray(confidences, dtype=np.float64)
    correct     = np.asarray(correct,     dtype=np.float64)
    if confidences.shape != correct.shape:
        raise ValueError("confidences and correct must have the same shape")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(confidences)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)

    return float(ece)


# ── Temperature Scaling ────────────────────────────────────────────────────────

class TemperatureScaling(nn.Module):
    """Post-hoc calibration via temperature scaling (Guo et al., 2017).

    A single scalar temperature T is learned on a held-out validation set by
    minimising NLL.  At inference, logits are divided by T before softmax,
    which reduces overconfidence without changing accuracy.

    Args:
        temperature: Initial temperature value (default 1.0 = no scaling).

    Example::

        cal = TemperatureScaling()
        cal.fit(model, val_loader, device='cpu')
        print(f"Optimal T = {cal.temperature.item():.3f}")

        # At inference
        logits = model(images)
        calibrated_probs = cal.calibrate(logits)
    """

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Return temperature-scaled logits."""
        return logits / self.temperature.clamp(min=1e-6)

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        """Return calibrated probabilities (softmax after scaling)."""
        return F.softmax(self.forward(logits), dim=-1)

    def fit(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        device: str | torch.device = "cpu",
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> "TemperatureScaling":
        """Optimise temperature on a validation set.

        Args:
            model: Trained model (weights are frozen).
            val_loader: Validation DataLoader yielding ``(images, targets)``.
            device: Compute device.
            lr: LBFGS learning rate (default 0.01).
            max_iter: LBFGS max iterations (default 50).

        Returns:
            ``self`` for chaining.
        """
        device  = torch.device(device)
        model   = model.to(device).eval()
        self.to(device)

        # Collect logits from the frozen model
        all_logits, all_targets = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                all_logits.append(model(images.to(device)).cpu())
                all_targets.append(targets.cpu())

        logits  = torch.cat(all_logits)
        targets = torch.cat(all_targets)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def _closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / self.temperature.clamp(min=1e-6), targets)
            loss.backward()
            return loss

        optimizer.step(_closure)
        logger.info("TemperatureScaling: T = %.4f", self.temperature.item())
        return self


# ── Reliability Diagram ────────────────────────────────────────────────────────

def reliability_diagram(
    confidences: np.ndarray,
    correct: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    save_path: Optional[str | Path] = None,
    show: bool = True,
):
    """Plot a reliability diagram (calibration curve).

    A well-calibrated model produces a near-diagonal bar chart.

    Args:
        confidences: 1-D array of max softmax confidence.
        correct: 1-D boolean / int array — 1 if prediction was correct.
        n_bins: Number of bins (default 15).
        title: Plot title.
        save_path: If given, save the figure to this path.
        show: Display the figure interactively (default True).

    Returns:
        ``(fig, ax)`` matplotlib objects.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for reliability_diagram") from e

    confidences = np.asarray(confidences, dtype=np.float64)
    correct     = np.asarray(correct,     dtype=np.float64)

    bin_edges   = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs    = np.zeros(n_bins)
    bin_confs   = np.zeros(n_bins)
    bin_counts  = np.zeros(n_bins, dtype=int)

    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() > 0:
            bin_accs[i]   = correct[mask].mean()
            bin_confs[i]  = confidences[mask].mean()
            bin_counts[i] = mask.sum()

    ece = compute_ece(confidences, correct, n_bins=n_bins)

    fig, ax = plt.subplots(figsize=(6, 6))
    width = 1.0 / n_bins

    ax.bar(bin_confs, bin_accs, width=width * 0.9, alpha=0.7, label="Accuracy", color="steelblue")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.fill_between(bin_confs, bin_accs, bin_confs, alpha=0.2, color="red", label=f"Gap (ECE={ece:.3f})")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{title}\nECE = {ece:.4f}")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax
