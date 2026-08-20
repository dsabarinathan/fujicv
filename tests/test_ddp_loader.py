"""Regression tests for DDP data-sharding (the v1.14.1 hang fix).

The NCCL deadlock itself needs 2 GPUs to reproduce, but the loader-distribution
logic that accompanies the fix is testable on CPU via a single-process gloo group.
"""
from __future__ import annotations

import os

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from fujicv.engine.trainer import Trainer


@pytest.fixture()
def gloo_group():
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29513")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    created = False
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        created = True
    yield
    if created:
        dist.destroy_process_group()


def test_distribute_loader_wraps_with_distributed_sampler(gloo_group):
    from torch.utils.data.distributed import DistributedSampler

    ds = TensorDataset(torch.randn(20, 3), torch.randint(0, 2, (20,)))
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

    trainer = Trainer.__new__(Trainer)  # bypass __init__; method uses no other state
    new_loader, sampler = trainer._distribute_loader(loader, shuffle=True)

    assert isinstance(sampler, DistributedSampler)
    assert new_loader.batch_size == 4
    assert new_loader.dataset is ds
    # DistributedSampler replaces shuffle; the loader must be iterable end-to-end.
    batches = list(new_loader)
    assert len(batches) == 5  # 20 samples / batch 4, world_size 1


def test_distribute_loader_preserves_drop_last_and_workers(gloo_group):
    ds = TensorDataset(torch.randn(18, 3), torch.randint(0, 2, (18,)))
    loader = DataLoader(ds, batch_size=4, num_workers=0, drop_last=True)

    trainer = Trainer.__new__(Trainer)
    new_loader, _ = trainer._distribute_loader(loader, shuffle=False)
    assert new_loader.drop_last is True
    # 18 // 4 = 4 full batches when dropping the remainder.
    assert len(list(new_loader)) == 4
