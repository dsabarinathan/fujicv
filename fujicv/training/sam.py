"""SAM (Sharpness-Aware Minimization) optimizer wrapper."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Optional, Type

import torch
from torch.optim import Optimizer


class SAM(Optimizer):
    """Sharpness-Aware Minimization (Foret et al., 2021).

    SAM simultaneously minimizes the loss value and the sharpness of the loss
    landscape, improving generalisation.  It wraps any base optimizer and
    requires **two forward-backward passes** per step.

    Each training step has two phases:

    1. ``first_step(zero_grad=True)``  — perturb weights toward sharp region
    2. Forward + backward on perturbed weights
    3. ``second_step(zero_grad=True)`` — take the base optimizer step and
       restore original weights

    Args:
        params: Model parameters (same as any optimizer).
        base_optimizer: Optimizer class to wrap (e.g. ``torch.optim.AdamW``).
        rho: Neighbourhood size for perturbation (default 0.05).  Larger →
            stronger regularisation; typical range 0.01–0.2.
        adaptive: Use Adaptive SAM (ASAM) which normalises per-parameter
            (default False).
        **kwargs: Passed to *base_optimizer* constructor.

    Example::

        from fujicv.training.sam import SAM

        optimizer = SAM(model.parameters(), torch.optim.AdamW,
                        rho=0.05, lr=1e-3, weight_decay=0.05)

        for images, targets in train_loader:
            # First pass
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # Second pass (perturbed weights)
            criterion(model(images), targets).backward()
            optimizer.second_step(zero_grad=True)
    """

    def __init__(
        self,
        params,
        base_optimizer: Type[Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs: Any,
    ) -> None:
        if rho < 0:
            raise ValueError(f"rho must be >= 0, got {rho}")

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups   = self.base_optimizer.param_groups

    # ------------------------------------------------------------------

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Perturb weights to the local sharp region.

        Call this after the **first** backward pass.  Then run a second
        forward+backward on the perturbed model before calling
        :meth:`second_step`.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                # ASAM: scale by |w| per parameter (ASAM paper uses absolute value, not square)
                e_w = (torch.abs(p) if group["adaptive"] else torch.tensor(1.0)) * p.grad
                p.add_(e_w, alpha=float(scale))

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Restore weights and take the base optimizer step.

        Call this after the **second** backward pass (on perturbed weights).
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # restore original weights

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):  # type: ignore[override]
        """Not recommended — use :meth:`first_step` / :meth:`second_step` instead.

        If *closure* is provided (re-evaluates the model), this method handles
        both SAM passes automatically.
        """
        if closure is None:
            raise RuntimeError(
                "SAM.step() requires a closure that re-evaluates the model. "
                "Prefer the explicit first_step() / second_step() pattern."
            )
        closure = torch.enable_grad()(closure)
        loss = closure()
        loss.backward()
        self.first_step(zero_grad=True)
        closure().backward()
        self.second_step()
        return loss

    def _grad_norm(self) -> torch.Tensor:
        # All params on the same device — norm across all
        device = next(
            (p for group in self.param_groups for p in group["params"] if p.grad is not None),
            None,
        )
        if device is None:
            return torch.tensor(0.0)
        target_device = device.device

        norms = [
            ((torch.abs(p) if group["adaptive"] else torch.tensor(1.0, device=target_device))
             * p.grad).norm(2).to(target_device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        return torch.stack(norms).norm(2)

    def load_state_dict(self, state_dict: dict) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
