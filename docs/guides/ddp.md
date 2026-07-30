# Multi-GPU Training (DDP)

FujiCV uses `DistributedDataParallel` (DDP) for multi-GPU training — the PyTorch-recommended approach.

!!! warning "Not DataParallel"
    FujiCV no longer auto-wraps in `nn.DataParallel`. DDP is faster, uses memory more efficiently,
    and scales across nodes. Always use `torchrun` for multi-GPU runs.

## Launch with torchrun

```bash
# 4 GPUs on a single node
torchrun --nproc_per_node=4 train.py
```

## Python changes

Pass `use_ddp=True` to `Trainer`. That is the only change required.

```python
trainer = Trainer(
    model=model,
    train_loader=train_loader,   # use DistributedSampler in your DataLoader
    val_loader=val_loader,
    ...,
    use_ddp=True,                # ← only new argument
)
```

## DistributedSampler

When using DDP, each process must see a different shard of the data:

```python
from torch.utils.data import DataLoader, DistributedSampler

sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
```

## What happens under the hood

1. `Trainer.__init__` detects `use_ddp=True` and reads `LOCAL_RANK` from the environment (set by `torchrun`).
2. It calls `torch.distributed.init_process_group(backend="nccl")` if not already initialized.
3. The model is moved to `cuda:{LOCAL_RANK}` and wrapped in `DistributedDataParallel`.
4. Checkpointing unwraps the model via `_model_core` so saved weights are DDP-free.
