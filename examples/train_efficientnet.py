"""Train EfficientNet-B0 on CIFAR-10 for 5 epochs.

EfficientNet uses compound scaling to uniformly scale network width, depth,
and resolution with a fixed set of scaling coefficients, achieving better
accuracy and efficiency than prior ConvNets at the same FLOP budget.
Reference: Tan & Le, 2019 (https://arxiv.org/abs/1905.11946)
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

import fujicv
from fujicv.data.datasets import get_default_dataset
from fujicv.data.transforms import get_train_transforms, get_val_transforms
from fujicv.engine.trainer import Trainer
from fujicv.losses.classification import CrossEntropyLoss
from fujicv.metrics.classification import Accuracy
from fujicv.models.builder import ModelBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Config ────────────────────────────────────────────────────────────────────
BACKBONE   = "efficientnet_b0"
IMAGE_SIZE = 224      # EfficientNet-B0 native resolution
BATCH_SIZE = 64
EPOCHS     = 5
LR         = 1e-3
OUTPUT_DIR = "runs/efficientnet_b0"

# ── Device ────────────────────────────────────────────────────────────────────
device = fujicv.get_device()   # auto: CUDA → MPS → CPU

# ── Data ──────────────────────────────────────────────────────────────────────
train_ds, val_ds, class_to_idx = get_default_dataset(
    name="cifar10",
    root="data",
    train_transform=get_train_transforms(IMAGE_SIZE, level="light"),
    val_transform=get_val_transforms(IMAGE_SIZE),
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Classes: {list(class_to_idx.keys())}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = ModelBuilder(
    backbone_name=BACKBONE,
    backbone_source="timm",
    pretrained=True,
    task="classification",
    num_outputs=10,
    image_size=IMAGE_SIZE,
).build()

total = sum(p.numel() for p in model.parameters()) / 1e6
print(f"{BACKBONE} | {total:.1f}M parameters")

# ── Optimizer & scheduler ─────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=CrossEntropyLoss(),
    metrics={"accuracy": Accuracy()},
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=EPOCHS,
    task="classification",
    output_dir=OUTPUT_DIR,
    class_to_idx=class_to_idx,
    monitor_metric="val_accuracy",
    mixed_precision=True,
)

history = trainer.train()

best_acc = max(history.metrics.get("val_accuracy", [0]))
print(f"\nDone! Best val accuracy: {best_acc:.4f} ({best_acc*100:.1f}%)")
print(f"Checkpoint saved to: {OUTPUT_DIR}/best.pt")
