"""
Train ResNet-18 on the HuggingFace 'beans' dataset (3-class plant disease).

Dataset: https://huggingface.co/datasets/beans
Classes: angular_leaf_spot | bean_rust | healthy
~1000 training images, downloads automatically (~60 MB).

Usage:
    pip install "fujicv[hf]"
    python examples/train_hf_beans.py
"""

import logging
import os

import torch
import torch.optim as optim

import fujicv
from fujicv.data.hf_dataset import load_hf_dataset
from fujicv.data.transforms import get_train_transforms, get_val_transforms
from fujicv.engine.trainer import Trainer
from fujicv.losses.classification import CrossEntropyLoss
from fujicv.metrics.classification import Accuracy, F1
from fujicv.models.builder import ModelBuilder
from fujicv.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ── Config ────────────────────────────────────────────────────────────────────
BACKBONE   = "resnet18"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 1e-3
OUTPUT_DIR = "runs/hf_beans"
SEED       = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
set_seed(SEED)
device = fujicv.get_device()
print(f"Device: {device}")

# ── Load dataset from HuggingFace Hub ─────────────────────────────────────────
train_ds, val_ds, test_ds, class_to_idx = load_hf_dataset(
    repo_id="beans",
    image_col="image",
    label_col="labels",
    task="classification",
    train_split="train",
    val_split="validation",
    test_split="test",
    train_transform=get_train_transforms(IMAGE_SIZE, level="medium"),
    val_transform=get_val_transforms(IMAGE_SIZE),
)

print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
print(f"Classes: {class_to_idx}")

train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False
)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False
)

# ── Build model ───────────────────────────────────────────────────────────────
model = ModelBuilder(
    backbone_name=BACKBONE,
    backbone_source="timm",
    pretrained=True,
    task="classification",
    num_outputs=len(class_to_idx),
    image_size=IMAGE_SIZE,
).build()

total = sum(p.numel() for p in model.parameters()) / 1e6
print(f"{BACKBONE} | {total:.1f}M parameters | {len(class_to_idx)} classes")

# ── Train ─────────────────────────────────────────────────────────────────────
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=CrossEntropyLoss(),
    metrics={"accuracy": Accuracy(), "f1": F1()},
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=EPOCHS,
    task="classification",
    output_dir=OUTPUT_DIR,
    class_to_idx=class_to_idx,
    monitor_metric="val_accuracy",
    mixed_precision=False,
    early_stopping_patience=5,
)

history = trainer.train()
best_acc = max(history.metrics.get("val_accuracy", [0]))
print(f"\nBest val accuracy: {best_acc*100:.2f}%  →  {OUTPUT_DIR}/best.pt")
