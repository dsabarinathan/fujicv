"""E2 — Throughput & peak-memory overhead: FujiCV Trainer vs. a raw PyTorch loop.

Runs the SAME model and data through (a) an equivalent hand-written AMP loop and
(b) FujiCV's Trainer, measuring images/sec and peak GPU memory. Uses a warmup
epoch and reports mean +/- std over several timed runs to control for noise.

By default it uses a synthetic dataset so the harness runs anywhere; for the
paper, pass real data (e.g. CelebA) via --data-dir and adapt `build_loaders`.

Example:
    python bench_throughput.py --runs 3 --epochs 1 --batch-size 256
"""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from common import Timer, save_result, set_seed

try:
    from torch.amp import GradScaler, autocast
    _AMP = lambda enabled: autocast("cuda", enabled=enabled)  # noqa: E731
except ImportError:  # PyTorch 2.0.x
    from torch.cuda.amp import GradScaler, autocast
    _AMP = lambda enabled: autocast(enabled=enabled)  # noqa: E731


def build_loaders(n_train: int, batch_size: int, img: int, num_workers: int):
    X = torch.randn(n_train, 3, img, img)
    y = torch.randint(0, 10, (n_train,))
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def make_model() -> nn.Module:
    from fujicv.models.builder import ModelBuilder
    return ModelBuilder("resnet18", backbone_source="timm", pretrained=False,
                        task="classification", num_outputs=10, image_size=64).build()


def run_raw(model, train_loader, val_loader, device, epochs, use_amp):
    """Equivalent-functionality raw loop: FujiCV also tracks train+val accuracy
    and checkpoints every epoch, so the raw baseline must do the same to be fair."""
    import tempfile
    from pathlib import Path

    import numpy as np
    model = model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=use_amp)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    n_train = len(train_loader.dataset)

    def evaluate(loader):
        preds, tgts = [], []
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with _AMP(use_amp):
                logits = model(x)
            preds.append(logits.detach().float().cpu().numpy())
            tgts.append(y.numpy())
        return float((np.concatenate(preds).argmax(1) == np.concatenate(tgts)).mean())

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        best = 0.0
        with Timer() as t:
            for _ in range(epochs):
                model.train()
                tr_preds, tr_tgts = [], []
                for x, y in train_loader:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    with _AMP(use_amp):
                        logits = model(x)
                        loss = crit(logits, y)
                    scaler.scale(loss).backward()
                    scaler.step(opt); scaler.update()
                    tr_preds.append(logits.detach().float().cpu().numpy())
                    tr_tgts.append(y.detach().cpu().numpy())
                (np.concatenate(tr_preds).argmax(1) == np.concatenate(tr_tgts)).mean()  # train acc
                model.eval()
                with torch.no_grad():
                    val_acc = evaluate(val_loader)
                # Checkpoint (best + last) like FujiCV.
                torch.save({"model_state_dict": model.state_dict()}, out / "last.pt")
                if val_acc > best:
                    best = val_acc
                    torch.save({"model_state_dict": model.state_dict()}, out / "best.pt")
    peak = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0
    return n_train * epochs / t.elapsed, peak


def run_fujicv(model, train_loader, val_loader, device, epochs, use_amp):
    import tempfile

    from fujicv.engine.trainer import Trainer
    from fujicv.losses.classification import CrossEntropyLoss
    from fujicv.metrics.classification import Accuracy
    n_train = len(train_loader.dataset)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(
            model=model, train_loader=train_loader, val_loader=val_loader,
            loss_fn=CrossEntropyLoss(), metrics={"accuracy": Accuracy()},
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
            epochs=epochs, task="classification", output_dir=tmp,
            mixed_precision=use_amp,
        )
        with Timer() as t:
            trainer.train()
    peak = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0.0
    return n_train * epochs / t.elapsed, peak


def timed(fn, runs, **kw):
    tputs, mems = [], []
    for r in range(runs + 1):  # +1 warmup
        set_seed(1000 + r)
        model = make_model()
        tp, mem = fn(model, **kw)
        if r > 0:  # discard warmup
            tputs.append(tp); mems.append(mem)
    return tputs, mems


def build_val_loader(n_val, batch_size, img, num_workers):
    X = torch.randn(n_val, 3, img, img)
    y = torch.randint(0, 10, (n_val,))
    return DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--n-train", type=int, default=8192)
    ap.add_argument("--img", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    train_loader = build_loaders(args.n_train, args.batch_size, args.img, args.num_workers)
    val_loader = build_val_loader(args.n_train // 4, args.batch_size, args.img, args.num_workers)

    kw = dict(train_loader=train_loader, val_loader=val_loader, device=device,
              epochs=args.epochs, use_amp=use_amp)
    raw_tp, raw_mem = timed(run_raw, args.runs, **kw)
    fuji_tp, fuji_mem = timed(run_fujicv, args.runs, **kw)

    def stat(v):
        return {"mean": statistics.mean(v), "std": statistics.pstdev(v) if len(v) > 1 else 0.0}

    overhead = 1.0 - stat(fuji_tp)["mean"] / stat(raw_tp)["mean"]
    result = {
        "config": vars(args) | {"device": str(device), "amp": use_amp},
        "raw_pytorch": {"throughput_img_s": stat(raw_tp), "peak_mem_mb": stat(raw_mem)},
        "fujicv": {"throughput_img_s": stat(fuji_tp), "peak_mem_mb": stat(fuji_mem)},
        "fujicv_throughput_overhead_frac": overhead,
    }
    save_result("e2_throughput", result)
    print(f"raw:    {stat(raw_tp)['mean']:.1f} img/s   peak {stat(raw_mem)['mean']:.1f} MB")
    print(f"fujicv: {stat(fuji_tp)['mean']:.1f} img/s   peak {stat(fuji_mem)['mean']:.1f} MB")
    print(f"FujiCV throughput overhead: {overhead*100:+.1f}%")


if __name__ == "__main__":
    main()
