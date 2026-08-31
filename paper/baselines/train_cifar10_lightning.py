"""PyTorch Lightning CIFAR-10 baseline — same pipeline as the FujiCV example.

Used for the paper's E1 (lines-of-code) comparison. Requires
`pip install pytorch-lightning torchmetrics`.
"""
from __future__ import annotations

import argparse

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18


class LitClassifier(pl.LightningModule):
    def __init__(self, num_classes=10, lr=3e-4, epochs=5):
        super().__init__()
        self.save_hyperparameters()
        self.model = resnet18(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        x, y = batch
        loss = self.criterion(self(x), y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        self.val_acc.update(logits, y)
        self.log("val_accuracy", self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.hparams.epochs)
        return {"optimizer": opt, "lr_scheduler": sched}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--image-size", type=int, default=32)
    args = ap.parse_args()

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

    model = LitClassifier(num_classes=10, epochs=args.epochs)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        callbacks=[
            ModelCheckpoint(monitor="val_accuracy", mode="max", save_top_k=1),
            EarlyStopping(monitor="val_accuracy", mode="max", patience=3),
        ],
        default_root_dir="runs/lightning_cifar10",
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
