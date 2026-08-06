"""Learning Rate Finder: exponential range test to find the optimal LR."""
from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class LRFinder:
    """Exponential LR range test (Smith 2015 / fast.ai style).

    Usage::

        finder = LRFinder(model, optimizer, criterion)
        finder.range_test(train_loader, start_lr=1e-7, end_lr=10, num_iter=100)
        fig = finder.plot()
        best_lr = finder.suggestion()
        finder.reset()   # restore model + optimizer before training
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model_state  = copy.deepcopy(model.state_dict())
        self._optim_state  = copy.deepcopy(optimizer.state_dict())
        # Remember where the model started so we can restore it after the sweep
        self._origin_device = next(model.parameters()).device

        self.history: dict = {"lr": [], "loss": []}

    def range_test(
        self,
        loader: DataLoader,
        start_lr: float = 1e-7,
        end_lr: float = 10.0,
        num_iter: int = 100,
        smooth_f: float = 0.05,
        diverge_th: float = 5.0,
    ) -> None:
        """Run the range test.

        Args:
            loader: Training data loader (cycled automatically when exhausted).
            start_lr: Starting learning rate.
            end_lr: Ending learning rate.
            num_iter: Number of update steps.
            smooth_f: Exponential smoothing factor for loss (0 = no smooth).
            diverge_th: Stop early when loss > diverge_th * best_loss.
        """
        self.history = {"lr": [], "loss": []}
        best_loss = float("inf")

        self.model.load_state_dict(self._model_state)
        self.optimizer.load_state_dict(self._optim_state)
        self.model.to(self.device).train()

        for pg in self.optimizer.param_groups:
            pg["lr"] = start_lr

        mult = (end_lr / start_lr) ** (1.0 / max(num_iter - 1, 1))
        avg_loss = 0.0
        loader_iter = iter(loader)

        for step in range(num_iter):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            inputs  = batch[0].to(self.device)
            targets = batch[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss    = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            raw = loss.item()
            avg_loss = smooth_f * raw + (1 - smooth_f) * avg_loss
            # Bias correction (guard against smooth_f=0 giving 0-denominator)
            bias_correction = 1 - (1 - smooth_f) ** (step + 1)
            smoothed = avg_loss / bias_correction if bias_correction > 0 else raw

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["lr"].append(current_lr)
            self.history["loss"].append(smoothed)

            if smoothed < best_loss:
                best_loss = smoothed

            if smoothed > diverge_th * best_loss:
                break

            for pg in self.optimizer.param_groups:
                pg["lr"] *= mult

        # Restore state so the caller can train normally afterward
        self.model.load_state_dict(self._model_state)
        self.model.to(self._origin_device)
        self.optimizer.load_state_dict(self._optim_state)

    def suggestion(self, skip_start: int = 10, skip_end: int = 5) -> float:
        """Return the LR at the point of steepest loss descent.

        Args:
            skip_start: Ignore the first N steps (noisy warm-up).
            skip_end: Ignore the last N steps (diverging region).

        Returns:
            Suggested learning rate.
        """
        lrs    = self.history["lr"][skip_start: -skip_end or None]
        losses = self.history["loss"][skip_start: -skip_end or None]

        if len(losses) < 2:
            return lrs[0] if lrs else 1e-3

        min_grad_idx = int(
            min(range(1, len(losses)), key=lambda i: losses[i] - losses[i - 1])
        )
        return lrs[min_grad_idx]

    def plot(self, skip_start: int = 10, skip_end: int = 5, log_lr: bool = True):
        """Plot smoothed loss vs learning rate.

        Args:
            skip_start: Trim first N noisy steps.
            skip_end: Trim last N diverging steps.
            log_lr: Use log scale on the LR axis.

        Returns:
            ``matplotlib.figure.Figure``.
        """
        import matplotlib.pyplot as plt

        lrs    = self.history["lr"][skip_start: -skip_end or None]
        losses = self.history["loss"][skip_start: -skip_end or None]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(lrs, losses)
        if log_lr:
            ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Loss (smoothed)")
        ax.set_title("LR Finder — Loss vs Learning Rate")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        # Mark the suggested LR
        best_lr = self.suggestion(skip_start, skip_end)
        ax.axvline(best_lr, color="red", linestyle="--", label=f"Suggestion: {best_lr:.2e}")
        ax.legend()
        return fig

    def reset(self) -> None:
        """Restore model and optimizer to their pre-range-test state."""
        self.model.load_state_dict(self._model_state)
        self.optimizer.load_state_dict(self._optim_state)
