"""Class-imbalance-aware sampling utilities."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def make_weighted_sampler(
    labels: Union[List[int], np.ndarray],
    num_samples: Optional[int] = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Build a :class:`torch.utils.data.WeightedRandomSampler` from class labels.

    Each sample is assigned a weight equal to the inverse class frequency,
    so that every class contributes equally in expectation — regardless of
    how many samples each class has in the dataset.

    Args:
        labels: Integer class label for every sample in the dataset
            (length == ``len(dataset)``).
        num_samples: Number of samples to draw per epoch.  Defaults to
            ``len(labels)`` (same as a standard epoch).
        replacement: Whether to sample with replacement (default ``True``).
            Must be ``True`` when oversampling minority classes.

    Returns:
        A ``WeightedRandomSampler`` ready to pass as the ``sampler`` argument
        to :class:`torch.utils.data.DataLoader`.

    Example::

        from fujicv.data.sampler import make_weighted_sampler
        from torch.utils.data import DataLoader

        sampler = make_weighted_sampler(train_df["label"].tolist())
        loader  = DataLoader(dataset, batch_size=32, sampler=sampler)

    Note:
        When using a sampler, set ``shuffle=False`` in DataLoader — the two
        are mutually exclusive.
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    classes, counts = np.unique(labels_arr, return_counts=True)

    class_weights = np.zeros(int(classes.max()) + 1, dtype=np.float64)
    for cls, cnt in zip(classes, counts):
        class_weights[cls] = 1.0 / cnt

    sample_weights = torch.tensor(class_weights[labels_arr], dtype=torch.double)

    n = num_samples if num_samples is not None else len(labels_arr)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=n,
        replacement=replacement,
    )


def class_weights_from_labels(
    labels: Union[List[int], np.ndarray],
    num_classes: Optional[int] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Compute inverse-frequency class weights for use in loss functions.

    Args:
        labels: Integer class labels for the training set.
        num_classes: Total number of classes.  If ``None``, inferred from
            ``max(labels) + 1``.
        normalize: Scale weights so they sum to ``num_classes`` (default
            ``True``), keeping the overall gradient magnitude stable.

    Returns:
        Float tensor of shape ``(num_classes,)`` suitable for the
        ``weight`` argument of ``nn.CrossEntropyLoss``.

    Example::

        from fujicv.data.sampler import class_weights_from_labels
        weights  = class_weights_from_labels(train_df["label"])
        loss_fn  = nn.CrossEntropyLoss(weight=weights.to(device))
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    n_cls = num_classes if num_classes is not None else int(labels_arr.max()) + 1

    counts = np.bincount(labels_arr, minlength=n_cls).astype(np.float64)
    counts = np.where(counts == 0, 1, counts)     # avoid divide-by-zero
    weights = 1.0 / counts

    if normalize:
        weights = weights * n_cls / weights.sum()

    return torch.tensor(weights, dtype=torch.float32)
