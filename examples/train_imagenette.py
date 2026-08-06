"""Train on Tiny ImageNet (200-class ImageNet subset, 64×64 px, auto-downloads).

Tiny ImageNet is a well-known benchmark: 100 K training images across 200
ImageNet classes. Images are 64×64 pixels, so training is fast even on CPU.

Expected val accuracy after 10 epochs with ResNet-18 (pretrained):
  64×64 images  → ~55-60%   (limited by small resolution)
  224×224 upscaled → ~68-72%

Run:
    python examples/train_imagenette.py
"""

from __future__ import annotations

import logging

import torch
from torch.utils.data import DataLoader

import fujicv
from fujicv.data.hf_dataset import load_hf_dataset
from fujicv.data.transforms import get_train_transforms, get_val_transforms
from fujicv.engine.trainer import Trainer
from fujicv.losses.classification import CrossEntropyLoss
from fujicv.metrics.classification import Accuracy
from fujicv.models.builder import ModelBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
BACKBONE     = "resnet18"      # swap to "efficientnet_b0" for +5-8% accuracy
IMAGE_SIZE   = 64              # native Tiny ImageNet resolution (fast)
BATCH_SIZE   = 128
EPOCHS       = 10
LR           = 1e-3
WEIGHT_DECAY = 1e-4
OUTPUT_DIR   = "runs/tiny-imagenet"
NUM_WORKERS  = 0               # set to 4 on Linux / Colab for speed

# ── Device ────────────────────────────────────────────────────────────────────
device = fujicv.get_device()
logger.info("Device: %s", device)

# ── Data (downloads ~130 MB, cached after first run) ─────────────────────────
logger.info("Loading Tiny ImageNet from HuggingFace Hub …")

train_ds, val_ds, class_to_idx = load_hf_dataset(
    "Maysee/tiny-imagenet",
    image_col="image",
    label_col="label",
    task="classification",
    train_split="train",
    val_split="valid",
    test_split=None,
    train_transform=get_train_transforms(IMAGE_SIZE, level="medium"),
    val_transform=get_val_transforms(IMAGE_SIZE),
)

logger.info("Train: %d | Val: %d | Classes: %d", len(train_ds), len(val_ds), len(class_to_idx))

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"),
)

# ── Model ─────────────────────────────────────────────────────────────────────
model = ModelBuilder(
    backbone_name=BACKBONE,
    backbone_source="timm",
    pretrained=True,
    task="classification",
    num_outputs=len(class_to_idx),   # 200 classes
    image_size=IMAGE_SIZE,
).build()

total_params = sum(p.numel() for p in model.parameters()) / 1e6
logger.info("Model: %s | Parameters: %.1f M | Classes: %d", BACKBONE, total_params, len(class_to_idx))

# ── Optimizer & schedule ──────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── Trainer ───────────────────────────────────────────────────────────────────
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
    mixed_precision=(device.type == "cuda"),
    early_stopping_patience=5,
)

history = trainer.train()

# ── Summary ───────────────────────────────────────────────────────────────────
best_acc = max(history.metrics.get("val_accuracy", [0]))
print(f"\n{'='*55}")
print(f"  Dataset   : Tiny ImageNet (200 classes, 64×64 px)")
print(f"  Model     : {BACKBONE}  ({total_params:.1f} M params)")
print(f"  Best acc  : {best_acc:.4f}  ({best_acc*100:.1f}%)")
print(f"  Saved to  : {OUTPUT_DIR}/")
print(f"{'='*55}")
