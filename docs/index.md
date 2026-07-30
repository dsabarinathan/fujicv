# FujiCV

**Lightweight image classification & regression for PyTorch — from prototype to production in 10 lines.**

[![PyPI version](https://img.shields.io/pypi/v/fujicv?color=blue)](https://pypi.org/project/fujicv/)
[![PyPI downloads](https://img.shields.io/pypi/dm/fujicv)](https://pypi.org/project/fujicv/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](https://github.com/dsabarinathan/fujicv/blob/main/LICENSE)
[![CI](https://github.com/dsabarinathan/fujicv/actions/workflows/ci.yml/badge.svg)](https://github.com/dsabarinathan/fujicv/actions)

---

## Why FujiCV?

You want to train a ResNet50 on your image dataset — with AMP, early stopping, and a best-model checkpoint.
Here is how much code that takes:

| Framework | Lines of code | What you get |
|---|---|---|
| **Raw PyTorch** | ~90 lines | Full control, full boilerplate |
| **PyTorch Lightning** | ~45 lines | Great for large-scale training; heavy for a weekend project |
| **FujiCV** | **~12 lines** | AMP · Early stopping · Checkpointing · Metrics · W&B · HPO · Grad-CAM |

FujiCV is not trying to replace Lightning for distributed training at scale.
It is designed for **rapid prototyping, Kaggle competitions, students, and researchers** who need results — not boilerplate.

---

## 30-second install

```bash
pip install fujicv
# GPU users: also install PyTorch for your CUDA version first
# https://pytorch.org/get-started/locally/
```

---

## 12-line quick start

```python
from fujicv.models.builder import ModelBuilder
from fujicv.losses import get_loss
from fujicv.metrics import get_metric
from fujicv.engine.trainer import Trainer
from fujicv.data import build_splits, build_dataloaders
from fujicv.utils import set_seed
import torch.optim as optim

set_seed(42)
train_df, val_df, _ = build_splits({"csv_path": "data.csv", "image_col": "path", "label_col": "label"})
train_loader, val_loader, _ = build_dataloaders(train_df, val_df, None, {}, {})

model = ModelBuilder("resnet50", task="classification", num_outputs=10).build()
trainer = Trainer(
    model=model, train_loader=train_loader, val_loader=val_loader,
    loss_fn=get_loss("LabelSmoothingCE", {"smoothing": 0.1}),
    metrics={"accuracy": get_metric("Accuracy")},
    optimizer=optim.AdamW(model.parameters(), lr=3e-4),
    epochs=30, task="classification", output_dir="outputs/",
)
history = trainer.train()   # → best.pt  last.pt  history.csv
```

---

## Feature highlights

- **700+ backbones** from `timm` and `torchvision`, head auto-stripped.
- **15 losses** · **16 metrics** across classification, regression, and multi-label.
- **Grad-CAM / Grad-CAM++** for model explainability.
- **Optuna HPO** with pruning strategies.
- **ONNX + TorchScript export** with INT8 quantization.
- **LR Finder**, **SWA**, **EMA**, **SAM**, **Mixup/CutMix**, **RandAugment**.
- **DDP** multi-GPU training via `torchrun`.
- **K-Fold cross-validation** with out-of-fold predictions.

[Get started →](getting-started/quickstart.md){ .md-button .md-button--primary }
[View on GitHub →](https://github.com/dsabarinathan/fujicv){ .md-button }
