"""Raw-PyTorch CIFAR-10 baseline — the same pipeline as examples/train_cifar10.py.

Manual AMP, cosine schedule, accuracy tracking, best-checkpoint selection, early
stopping, and history logging — all by hand. Used for the paper's E1 (lines of
code) and E3 (accuracy parity) comparisons against FujiCV.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18

try:
    from torch.amp import GradScaler, autocast
    def _amp(enabled):
        return autocast("cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    def _amp(enabled):
        return autocast(enabled=enabled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--image-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="runs/baseline_cifar10")
    args = ap.parse_args()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = CIFAR10("data", train=True, download=True, transform=train_tf)
    val_ds = CIFAR10("data", train=False, download=True, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = resnet18(num_classes=10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    Path(args.out).mkdir(parents=True, exist_ok=True)
    best_acc, patience, bad = 0.0, 3, 0
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with _amp(use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with _amp(use_amp):
                    logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        history.append({"epoch": epoch, "val_accuracy": val_acc})
        print(f"[baseline] epoch {epoch} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, bad = val_acc, 0
            torch.save({"model_state_dict": model.state_dict()}, f"{args.out}/best.pt")
        else:
            bad += 1
            if bad >= patience:
                print("[baseline] early stopping")
                break

    elapsed = time.time() - t0
    result = {"framework": "raw_pytorch", "best_val_accuracy": best_acc,
              "elapsed_sec": elapsed, "history": history, "config": vars(args)}
    with open(f"{args.out}/result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[baseline] best val acc {best_acc:.4f} in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
