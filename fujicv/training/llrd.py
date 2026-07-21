"""Layer-wise Learning Rate Decay (LLRD) for fine-tuning pretrained models."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import torch.nn as nn


def get_layer_wise_lr_params(
    model: nn.Module,
    base_lr: float,
    decay_rate: float = 0.75,
    num_layers: Optional[int] = None,
    head_lr_scale: float = 1.0,
    no_decay_names: Tuple[str, ...] = ("bias", "norm", "bn"),
) -> List[Dict]:
    """Build parameter groups with layer-wise learning rate decay.

    Assigns a smaller LR to earlier (lower) layers and a larger LR to later
    layers — a key trick when fine-tuning large pretrained models (ViT, Swin,
    ConvNeXt).  The head always uses ``base_lr * head_lr_scale``.

    The function works by assigning each parameter a *layer depth index*
    derived from its module name:

    * Parameters with ``layer.N`` or ``blocks.N`` in their name are assigned
      depth ``N``.
    * Parameters without a depth indicator (embedding, patch_embed, cls_token,
      pos_embed, head, fc, classifier) are assigned fixed depths.
    * LR for depth ``d`` = ``base_lr * decay_rate ** (num_layers - d)``.

    Args:
        model: The model whose parameters to group.
        base_lr: Base (maximum) learning rate, applied to the final layer.
        decay_rate: Multiplicative decay per layer going toward the input
            (default 0.75).  0.75 means each earlier layer gets 75% of the
            next layer's LR.
        num_layers: Total number of transformer / residual layers to consider.
            If ``None``, inferred as one more than the deepest layer index
            found in the parameter names.
        head_lr_scale: Multiplier on ``base_lr`` for the classification head
            parameters (default 1.0 — same as the deepest layer).
        no_decay_names: Sub-strings that indicate a parameter should have
            ``weight_decay=0`` (default: bias, norm, bn).

    Returns:
        A list of dicts suitable for passing to any PyTorch optimizer's
        ``param_groups`` argument.

    Example::

        from fujicv.training.llrd import get_layer_wise_lr_params
        param_groups = get_layer_wise_lr_params(model, base_lr=1e-4,
                                                 decay_rate=0.75)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
    """
    # Patterns that signal "head" or "input stem"
    HEAD_PATTERNS   = re.compile(r"(head|fc|classifier|linear)")
    STEM_PATTERNS   = re.compile(r"(patch_embed|cls_token|pos_embed|stem|conv_stem)")
    LAYER_PATTERNS  = re.compile(r"(?:blocks?|layer|layers?|stage)[\._](\d+)")

    # 1 — collect parameter names and infer layer depths
    name_depth: List[Tuple[str, nn.Parameter, int, bool]] = []
    max_depth = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        no_wd = any(nd in name.lower() for nd in no_decay_names)

        m = LAYER_PATTERNS.search(name)
        if m:
            depth = int(m.group(1)) + 1   # 1-indexed
        elif HEAD_PATTERNS.search(name):
            depth = -1                     # sentinel → handled below
        elif STEM_PATTERNS.search(name):
            depth = 0
        else:
            depth = 0

        max_depth = max(max_depth, depth)
        name_depth.append((name, param, depth, no_wd))

    if num_layers is None:
        num_layers = max_depth + 1

    # 2 — build groups keyed by (depth, no_wd)
    groups: Dict[Tuple[int, bool], List[nn.Parameter]] = {}
    for name, param, depth, no_wd in name_depth:
        groups.setdefault((depth, no_wd), []).append(param)

    # 3 — compute LR per depth
    param_groups = []
    for (depth, no_wd), params in groups.items():
        if depth == -1:
            lr = base_lr * head_lr_scale
        else:
            lr = base_lr * (decay_rate ** (num_layers - depth))

        param_groups.append({
            "params": params,
            "lr": lr,
            "weight_decay": 0.0 if no_wd else None,   # None → optimizer default
        })

    # Remove None weight_decay entries so the optimizer uses its own default
    for g in param_groups:
        if g["weight_decay"] is None:
            del g["weight_decay"]

    return param_groups


def print_llrd_summary(param_groups: List[Dict]) -> None:
    """Pretty-print a summary of the LLRD parameter groups."""
    total_params = sum(p.numel() for g in param_groups for p in g["params"])
    print(f"{'LR':>12}  {'WD':>8}  {'Params':>12}  {'%':>6}")
    print("-" * 46)
    for g in sorted(param_groups, key=lambda x: x["lr"]):
        n   = sum(p.numel() for p in g["params"])
        wd  = g.get("weight_decay", "default")
        pct = 100.0 * n / total_params if total_params else 0
        print(f"{g['lr']:>12.2e}  {str(wd):>8}  {n:>12,}  {pct:>5.1f}%")
    print(f"{'Total':>12}  {'':>8}  {total_params:>12,}  100.0%")
