"""E2 (DDP row) — multi-GPU throughput: raw DistributedDataParallel vs. FujiCV.

Runs BOTH a hand-written DDP loop and a FujiCV ``Trainer(use_ddp=True)`` in the
same job (sharing one process group) with equal functionality (train + validate
+ accuracy + checkpoint), and reports global throughput (images/s summed across
ranks). Requires >= 2 GPUs; launch with torchrun.

    torchrun --nproc_per_node=2 bench_ddp.py --epochs 3 --batch-size 256

Rank 0 writes results/e2_ddp.json. Validates the v1.14.1 DDP fix at scale and
completes the DDP row of Table 4.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.amp import GradScaler, autocast
    def _amp(en):
        return autocast("cuda", enabled=en)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    def _amp(en):
        return autocast(enabled=en)


def make_model():
    from fujicv.models.builder import ModelBuilder
    return ModelBuilder("resnet18", backbone_source="timm", pretrained=False,
                        task="classification", num_outputs=10, image_size=64).build()


def synth(n, img):
    return TensorDataset(torch.randn(n, 3, img, img), torch.randint(0, 10, (n,)))


def run_raw_ddp(train_ds, val_ds, device, rank, world, epochs, bs, use_amp):
    model = make_model().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp)
    tsampler = DistributedSampler(train_ds, shuffle=True)
    vsampler = DistributedSampler(val_ds, shuffle=False)
    tl = DataLoader(train_ds, batch_size=bs, sampler=tsampler, num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=bs, sampler=vsampler, num_workers=2, pin_memory=True)

    dist.barrier()
    t0 = time.perf_counter()
    for ep in range(epochs):
        tsampler.set_epoch(ep)
        model.train()
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with _amp(use_amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        model.eval()
        preds, tgts = [], []
        with torch.no_grad():
            for x, y in vl:
                x = x.to(device, non_blocking=True)
                with _amp(use_amp):
                    logits = model(x)
                preds.append(logits.float().cpu().numpy()); tgts.append(y.numpy())
        gathered_p = [None] * world; gathered_t = [None] * world
        dist.all_gather_object(gathered_p, np.concatenate(preds))
        dist.all_gather_object(gathered_t, np.concatenate(tgts))
        if rank == 0:
            (np.concatenate(gathered_p).argmax(1) == np.concatenate(gathered_t)).mean()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return len(train_ds) * epochs / elapsed  # global images/s


def run_fujicv_ddp(train_ds, val_ds, device, epochs, bs, use_amp):
    import tempfile

    from fujicv.engine.trainer import Trainer
    from fujicv.losses.classification import CrossEntropyLoss
    from fujicv.metrics.classification import Accuracy
    # FujiCV auto-adds DistributedSampler under use_ddp=True.
    tl = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    vl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    model = make_model()
    dist.barrier()
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        Trainer(model=model, train_loader=tl, val_loader=vl,
                loss_fn=CrossEntropyLoss(), metrics={"accuracy": Accuracy()},
                optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
                epochs=epochs, task="classification", output_dir=tmp,
                mixed_precision=use_amp, use_ddp=True).train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return len(train_ds) * epochs / elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=8192)
    ap.add_argument("--img", type=int, default=64)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    use_amp = True

    train_ds, val_ds = synth(args.n_train, args.img), synth(args.n_train // 4, args.img)

    raw_tp = run_raw_ddp(train_ds, val_ds, device, rank, world, args.epochs, args.batch_size, use_amp)
    fuji_tp = run_fujicv_ddp(train_ds, val_ds, device, args.epochs, args.batch_size, use_amp)

    if rank == 0:
        from common import save_result
        overhead = 1.0 - fuji_tp / raw_tp
        save_result("e2_ddp", {
            "config": vars(args) | {"world_size": world},
            "raw_pytorch_ddp": {"throughput_img_s": raw_tp},
            "fujicv_ddp": {"throughput_img_s": fuji_tp},
            "fujicv_throughput_overhead_frac": overhead,
        })
        print(f"[DDP x{world}] raw {raw_tp:.1f} img/s | fujicv {fuji_tp:.1f} img/s | overhead {overhead*100:+.1f}%")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
