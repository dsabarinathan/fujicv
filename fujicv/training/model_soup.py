"""Model soups — average the weights of several fine-tuned models.

"Model soups" (Wortsman et al., 2022) show that averaging the weights of
multiple models fine-tuned with different hyper-parameters often matches or
beats the best single model **and** the cost of an ensemble at inference time
(one forward pass instead of N).

Two recipes are provided:

* :func:`uniform_soup` — plain average of every ingredient. Simple, no held-out
  data required.
* :func:`greedy_soup` — sort ingredients by validation score, then add each one
  only if it improves the running soup (Wortsman §3). Needs an eval callback.

Both operate on state dicts, so the ingredients may come from separate runs,
checkpoints on disk, or SWA/EMA snapshots.

Example::

    from fujicv.training.model_soup import uniform_soup, greedy_soup

    states = [torch.load(p)["model_state_dict"] for p in checkpoint_paths]

    # Uniform
    uniform_soup(model, states)          # loads averaged weights in-place

    # Greedy (keeps only ingredients that help)
    def eval_fn(m):
        return evaluate_accuracy(m, val_loader)   # higher is better
    kept = greedy_soup(model, states, eval_fn)

Reference:
    Wortsman et al., "Model soups: averaging weights of multiple fine-tuned
    models improves accuracy without increasing inference time" (ICML 2022).
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

StateDict = Dict[str, torch.Tensor]


def _average_states(states: List[StateDict]) -> StateDict:
    """Return the element-wise mean of a non-empty list of state dicts."""
    if not states:
        raise ValueError("Cannot average an empty list of state dicts.")

    ref_state = states[0]
    keys = set(ref_state.keys())
    for i, s in enumerate(states[1:], start=1):
        if set(s.keys()) != keys:
            raise ValueError(
                f"State dict {i} has mismatched keys; all ingredients must share "
                "the same architecture."
            )
        for key in keys:
            if s[key].shape != ref_state[key].shape:
                raise ValueError(
                    f"State dict {i} has a shape mismatch for '{key}': "
                    f"{tuple(s[key].shape)} vs {tuple(ref_state[key].shape)}; "
                    "all ingredients must share the same architecture."
                )

    averaged: StateDict = {}
    n = len(states)
    for key, ref in states[0].items():
        if ref.is_floating_point():
            acc = torch.zeros_like(ref, dtype=torch.float64)
            for s in states:
                acc += s[key].to(torch.float64)
            averaged[key] = (acc / n).to(ref.dtype)
        else:
            # Integer buffers (e.g. num_batches_tracked) — keep the first.
            averaged[key] = ref.clone()
    return averaged


def uniform_soup(model: nn.Module, states: List[StateDict], strict: bool = True) -> nn.Module:
    """Average *states* uniformly and load the result into *model* in-place.

    Args:
        model: Model to receive the averaged weights.
        states: List of state dicts (all sharing *model*'s architecture).
        strict: Passed to ``load_state_dict`` (default ``True``).

    Returns:
        The same *model*, with soup weights loaded.
    """
    if not states:
        raise ValueError("uniform_soup requires at least one state dict.")
    averaged = _average_states(states)
    model.load_state_dict(averaged, strict=strict)
    logger.info("uniform_soup: averaged %d ingredients.", len(states))
    return model


def greedy_soup(
    model: nn.Module,
    states: List[StateDict],
    eval_fn: Callable[[nn.Module], float],
    higher_is_better: bool = True,
    strict: bool = True,
) -> List[int]:
    """Build a greedy soup, keeping only ingredients that improve the soup.

    Ingredients are first ranked by their individual ``eval_fn`` score. Starting
    from the best one, each remaining ingredient is tentatively added (as a
    running average); it is kept only if the averaged model scores at least as
    well as the current soup on ``eval_fn``.

    Args:
        model: Model used for evaluation and to receive the final soup weights.
        states: Candidate state dicts.
        eval_fn: Callable ``fn(model) -> float`` returning a validation score.
        higher_is_better: Whether a larger ``eval_fn`` value is better
            (default ``True``; set ``False`` for a loss).
        strict: Passed to ``load_state_dict``.

    Returns:
        Indices (into *states*) of the ingredients kept in the soup.
    """
    if not states:
        raise ValueError("greedy_soup requires at least one state dict.")

    def _better(a: float, b: float) -> bool:
        return a > b if higher_is_better else a < b

    # 1 — score each ingredient individually.
    scores: List[float] = []
    for i, s in enumerate(states):
        model.load_state_dict(s, strict=strict)
        scores.append(float(eval_fn(model)))
        logger.info("greedy_soup: ingredient %d score=%.4f", i, scores[-1])

    order = sorted(range(len(states)), key=lambda i: scores[i], reverse=higher_is_better)

    # 2 — greedily grow the soup.
    soup: List[StateDict] = [states[order[0]]]
    kept = [order[0]]
    model.load_state_dict(soup[0], strict=strict)
    best_score = scores[order[0]]
    logger.info("greedy_soup: seed ingredient %d (score=%.4f)", order[0], best_score)

    for idx in order[1:]:
        candidate = _average_states(soup + [states[idx]])
        model.load_state_dict(candidate, strict=strict)
        score = float(eval_fn(model))
        if _better(score, best_score) or score == best_score:
            soup.append(states[idx])
            kept.append(idx)
            best_score = score
            logger.info("greedy_soup: + ingredient %d → soup score=%.4f", idx, score)
        else:
            logger.info("greedy_soup: skip ingredient %d (score=%.4f)", idx, score)

    # 3 — load the final soup.
    model.load_state_dict(_average_states(soup), strict=strict)
    logger.info("greedy_soup: kept %d/%d ingredients, final score=%.4f",
                len(kept), len(states), best_score)
    return kept


__all__ = ["uniform_soup", "greedy_soup"]
