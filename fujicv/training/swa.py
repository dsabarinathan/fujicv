"""Stochastic Weight Averaging (SWA) utilities."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class SWA:
    """Stochastic Weight Averaging wrapper around ``torch.optim.swa_utils``.

    Maintains a running average of model weights across SWA epochs.  After
    training, call :meth:`finalize` with the training loader to refresh
    BatchNorm running statistics before evaluation.

    Usage::

        swa = SWA(model, swa_lr=1e-4)
        swa_scheduler = swa.get_scheduler(optimizer)

        for epoch in range(total_epochs):
            train_one_epoch(model, ...)
            if epoch >= swa_start:
                swa.update()
                swa_scheduler.step()

        swa.finalize(train_loader)          # update BN stats
        evaluate(swa.averaged_model, val_loader)
    """

    def __init__(
        self,
        model: nn.Module,
        swa_lr: Optional[float] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model  = model
        self.swa_lr = swa_lr
        self.device = device or next(model.parameters()).device
        self._averaged_model = torch.optim.swa_utils.AveragedModel(model)
        self._n_averaged     = 0

    @property
    def averaged_model(self) -> nn.Module:
        """The exponentially-averaged model (use for evaluation after :meth:`finalize`)."""
        return self._averaged_model

    @property
    def n_averaged(self) -> int:
        """Number of times :meth:`update` has been called."""
        return self._n_averaged

    def update(self) -> None:
        """Accumulate current model weights into the running SWA average."""
        self._averaged_model.update_parameters(self.model)
        self._n_averaged += 1

    def finalize(self, loader: DataLoader) -> None:
        """Update BatchNorm running stats using the training data.

        Must be called before evaluating :attr:`averaged_model` when the
        model contains BatchNorm layers.

        Args:
            loader: Training data loader (a single pass is sufficient).
        """
        torch.optim.swa_utils.update_bn(loader, self._averaged_model, device=self.device)

    def get_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        swa_lr: Optional[float] = None,
        anneal_epochs: int = 10,
        anneal_strategy: str = "cos",
    ) -> torch.optim.swa_utils.SWALR:
        """Return a SWALR learning-rate scheduler that anneals towards ``swa_lr``.

        Args:
            optimizer: The optimizer to schedule.
            swa_lr: Target SWA learning rate. Defaults to ``self.swa_lr``.
            anneal_epochs: Steps over which to anneal the LR.
            anneal_strategy: ``'cos'`` (default) or ``'linear'``.

        Returns:
            Configured :class:`torch.optim.swa_utils.SWALR` instance.
        """
        lr = swa_lr or self.swa_lr
        if lr is None:
            raise ValueError("swa_lr must be provided either at init or in get_scheduler()")
        return torch.optim.swa_utils.SWALR(
            optimizer,
            swa_lr=lr,
            anneal_epochs=anneal_epochs,
            anneal_strategy=anneal_strategy,
        )

    def state_dict(self) -> dict:
        return {
            "averaged_model": self._averaged_model.state_dict(),
            "n_averaged":     self._n_averaged,
        }

    def load_state_dict(self, state: dict) -> None:
        self._averaged_model.load_state_dict(state["averaged_model"])
        self._n_averaged = state["n_averaged"]


def get_swa_scheduler(
    optimizer: torch.optim.Optimizer,
    swa_lr: float,
    anneal_epochs: int = 10,
    anneal_strategy: str = "cos",
) -> torch.optim.swa_utils.SWALR:
    """Module-level helper: return a SWALR scheduler annealing to ``swa_lr``.

    Args:
        optimizer: The optimizer to schedule.
        swa_lr: Target SWA learning rate.
        anneal_epochs: Steps over which to anneal.
        anneal_strategy: ``'cos'`` or ``'linear'``.

    Returns:
        Configured :class:`torch.optim.swa_utils.SWALR`.
    """
    return torch.optim.swa_utils.SWALR(
        optimizer,
        swa_lr=swa_lr,
        anneal_epochs=anneal_epochs,
        anneal_strategy=anneal_strategy,
    )
