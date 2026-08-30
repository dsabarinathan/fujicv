"""E4 — Value of built-in techniques (ablation).

Trains the same model/budget with each built-in technique toggled on, to
quantify the accuracy each buys "for free". Reports val accuracy per config.

Example:
    python bench_ablation.py --epochs 15 --image-size 32
"""

from __future__ import annotations

import argparse
import tempfile

import torch
from torch.utils.data import DataLoader

from common import save_result, set_seed


def _train(configs_kw, train_ds, val_ds, num_classes, args):
    from fujicv.engine.trainer import Trainer
    from fujicv.losses.classification import CrossEntropyLoss
    from fujicv.metrics.classification import Accuracy
    from fujicv.models.builder import ModelBuilder

    set_seed(args.seed)
    model = ModelBuilder(args.backbone, backbone_source="timm", pretrained=True,
                         task="classification", num_outputs=num_classes,
                         image_size=args.image_size).build()
    tl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    vl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    with tempfile.TemporaryDirectory() as tmp:
        trainer = Trainer(
            model=model, train_loader=tl, val_loader=vl,
            loss_fn=CrossEntropyLoss(), metrics={"accuracy": Accuracy()},
            optimizer=opt, epochs=args.epochs, task="classification",
            output_dir=tmp, monitor_metric="val_accuracy", **configs_kw,
        )
        hist = trainer.train()
    return max(hist.metrics.get("val_accuracy", [0.0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--image-size", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from fujicv.data.datasets import get_default_dataset
    from fujicv.data.transforms import get_train_transforms, get_val_transforms
    train_ds, val_ds, c2i = get_default_dataset(
        args.dataset, root="data",
        train_transform=get_train_transforms(args.image_size, level="medium"),
        val_transform=get_val_transforms(args.image_size),
    )
    nc = len(c2i)

    # Each entry is a set of Trainer kwargs enabling one technique.
    variants = {
        "baseline": {},
        "+EMA": {"use_ema": True, "ema_warmup_steps": 100},
        "+SWA is external (see docs)": {},  # SWA runs outside Trainer; documented separately
    }

    results = {}
    baseline_acc = None
    for name, kw in variants.items():
        if name.startswith("+SWA"):
            continue  # skip placeholder; SWA has its own recipe
        acc = _train(kw, train_ds, val_ds, nc, args)
        results[name] = acc
        if name == "baseline":
            baseline_acc = acc
        delta = "" if baseline_acc is None else f"  (delta {acc - baseline_acc:+.4f})"
        print(f"  {name:12} val_acc={acc:.4f}{delta}")

    save_result(f"e4_ablation_{args.dataset}", {
        "config": vars(args),
        "results": results,
        "baseline": baseline_acc,
    })


if __name__ == "__main__":
    main()
