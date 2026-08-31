"""E3 — Accuracy parity on a standard dataset (CIFAR-10 by default).

Trains a FujiCV model and reports best validation accuracy. Run with a matched
hand-written loop (not included here) to complete the parity comparison; the
point is that the abstraction does not cost accuracy.

Example:
    python bench_accuracy.py --backbone resnet18 --epochs 10 --image-size 32
"""

from __future__ import annotations

import argparse
import tempfile

import torch
from torch.utils.data import DataLoader

from common import save_result, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "mnist"])
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--image-size", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true",
                    help="Train the backbone from scratch (match a from-scratch baseline).")
    ap.add_argument("--aug-level", default="medium", choices=["light", "medium", "heavy"],
                    help="Augmentation strength; use 'light' to match a resize+flip baseline.")
    args = ap.parse_args()
    set_seed(args.seed)

    from fujicv.data.datasets import get_default_dataset
    from fujicv.data.transforms import get_train_transforms, get_val_transforms
    from fujicv.engine.trainer import Trainer
    from fujicv.losses.classification import CrossEntropyLoss
    from fujicv.metrics.classification import Accuracy
    from fujicv.models.builder import ModelBuilder

    train_ds, val_ds, c2i = get_default_dataset(
        args.dataset, root="data",
        train_transform=get_train_transforms(args.image_size, level=args.aug_level),
        val_transform=get_val_transforms(args.image_size),
    )
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = ModelBuilder(args.backbone, backbone_source="timm",
                         pretrained=not args.no_pretrained,
                         task="classification", num_outputs=len(c2i),
                         image_size=args.image_size).build()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(
            model=model, train_loader=tl, val_loader=vl,
            loss_fn=CrossEntropyLoss(), metrics={"accuracy": Accuracy()},
            optimizer=opt, scheduler=sched, epochs=args.epochs,
            task="classification", output_dir=tmp, monitor_metric="val_accuracy",
        )
        hist = trainer.train()

    best = max(hist.metrics.get("val_accuracy", [0.0]))
    save_result(f"e3_accuracy_{args.dataset}", {
        "config": vars(args),
        "best_val_accuracy": best,
        "history": hist.metrics,
    })
    print(f"{args.dataset} best val accuracy: {best:.4f}")


if __name__ == "__main__":
    main()
