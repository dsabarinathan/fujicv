"""Quickstart: train a ConvNeXt-Tiny on CIFAR-10 using the built-in default dataset."""

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
BACKBONE = "resnet18"        # small backbone for a quick demo
IMAGE_SIZE = 32              # CIFAR-10 native resolution
BATCH_SIZE = 128
EPOCHS = 10
LR = 3e-4
OUTPUT_DIR = "runs/cifar10"

# ── Device (auto-detected) ────────────────────────────────────────────────────
device = fujicv.get_device()

# ── Data ──────────────────────────────────────────────────────────────────────
train_transform = get_train_transforms(IMAGE_SIZE, level="medium")
val_transform = get_val_transforms(IMAGE_SIZE)

train_ds, val_ds, class_to_idx = get_default_dataset(
    name="cifar10",
    root="data",
    train_transform=train_transform,
    val_transform=val_transform,
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")
print(f"Classes: {list(class_to_idx.keys())}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = ModelBuilder(
    backbone_name=BACKBONE,
    backbone_source="timm",
    pretrained=True,
    task="classification",
    num_outputs=10,
    image_size=IMAGE_SIZE,
).build()

# ── Optimizer & scheduler ─────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Loss & metrics ────────────────────────────────────────────────────────────
loss_fn = CrossEntropyLoss()
metrics = {"accuracy": Accuracy()}

# ── Train ─────────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    metrics=metrics,
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
print(f"\nBest val accuracy: {max(history.metrics.get('val_accuracy', [0])):.4f}")
print(f"Checkpoints saved to: {OUTPUT_DIR}/")
