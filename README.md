# FujiCV

**Open-source Python package for image classification and regression, built on timm + torchvision.**

[![CI](https://github.com/dsabarinathan/fujicv/actions/workflows/ci.yml/badge.svg)](https://github.com/dsabarinathan/fujicv/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)

---

## Features

- **Backbone factory** — instantly load any timm or torchvision model with a
  single call; classifier head auto-stripped, output dimension auto-detected.
- **Task heads** — ClassificationHead, RegressionHead, MultiLabelHead.
- **Custom layers** — LinearBNDropout, GeM pooling, AttentionPool, SqueezeExcite.
- **ModelBuilder** — assemble backbone + custom layers + head; dummy forward
  pass validates shapes at build time.
- **Albumentations pipelines** — light / medium / heavy augmentation presets,
  ImageNet normalisation, deterministic val transforms.
- **CSVImageDataset** — reads any CSV-driven image dataset; pre-validates files,
  skips missing with a warning; supports classification, regression, multilabel.
- **Loss functions** — 13 losses across classification, regression, and multilabel
  tasks, all registered in `LOSS_REGISTRY`.
- **Metrics** — 16 metrics across all tasks, registered in `METRIC_REGISTRY`.
- **Trainer** — AMP, gradient clipping, early stopping, best/last checkpointing,
  history CSV.
- **WandbLogger** — W&B integration via env-var only (`WANDB_API_KEY`); graceful
  no-op when W&B is absent.
- **Evaluation** — confusion matrix, ROC/PR curves, t-SNE, Grad-CAM (CNN),
  attention rollout (ViT).
- **Inference** — `Predictor.from_checkpoint` for single image and batch inference.
- **ONNX export** — export and numeric verification.

---

## Installation

```bash
# Core (CPU)
pip install fujicv

# With W&B logging
pip install "fujicv[wandb]"

# With ONNX export
pip install "fujicv[onnx]"

# Dev / testing
pip install "fujicv[dev]"
```

> **PyTorch is not included as a transitive dependency on PyPI.**
> Install it separately following the [official instructions](https://pytorch.org/get-started/locally/)
> to pick the right CUDA version.

---

## Quick Start

### 1. Classification

```python
from fujicv.models.builder import ModelBuilder
from fujicv.losses import get_loss
from fujicv.metrics import get_metric
from fujicv.engine.trainer import Trainer
from fujicv.data import build_splits, build_dataloaders
from fujicv.utils import load_config, set_seed

set_seed(42)
cfg = load_config("examples/configs/classification_example.yaml")

train_df, val_df, test_df = build_splits(cfg["dataset"])
train_loader, val_loader, _ = build_dataloaders(
    train_df, val_df, test_df, cfg["dataset"], cfg["augmentation"]
)

model = ModelBuilder(
    backbone_name="resnet50",
    task="classification",
    num_outputs=10,
    pretrained=True,
).build()

import torch.optim as optim

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=get_loss("LabelSmoothingCE", {"smoothing": 0.1}),
    metrics={"Accuracy": get_metric("Accuracy"), "F1": get_metric("F1")},
    optimizer=optim.AdamW(model.parameters(), lr=3e-4),
    epochs=30,
    task="classification",
    output_dir="outputs/",
)
history = trainer.train()
```

### 2. Using the example scripts

```bash
# Train
python examples/train.py --config examples/configs/classification_example.yaml

# Evaluate (with attention maps)
python examples/evaluate.py \
    --config examples/configs/classification_example.yaml \
    --checkpoint outputs/classification/best.pt \
    --with-attention-maps
```

### 3. Inference

```python
from fujicv.inference import Predictor
from fujicv.models.builder import ModelBuilder

model_skeleton = ModelBuilder(
    backbone_name="resnet50", task="classification", num_outputs=10, pretrained=False
).build()

predictor = Predictor.from_checkpoint("outputs/classification/best.pt", model=model_skeleton)
label, confidence = predictor.predict("path/to/image.jpg")
print(f"Predicted: {label}  ({confidence:.1%})")
```

### 4. ONNX Export

```python
from fujicv.export import to_onnx, verify_onnx

to_onnx(model, "model.onnx")
verify_onnx(model, "model.onnx")
```

---

## Package Layout

```
fujicv/
  models/       backbone factory, heads, custom layers, ModelBuilder
  data/         CSVImageDataset, transforms, dataloader factory
  losses/       13 loss functions + registry
  metrics/      16 metric callables + registry
  engine/       Trainer, WandbLogger, callbacks
  eval/         plots, report, ROC/PR curves, t-SNE, attention maps
  utils/        Registry, set_seed, config loader
  inference/    Predictor
  export/       ONNX export & verification
tests/
examples/
  configs/      YAML configs for each task type
  train.py
  evaluate.py
```

---

## Supported Tasks

| Task | Loss examples | Metric examples |
|------|--------------|----------------|
| Classification / Multiclass | CrossEntropyLoss, FocalLoss, LabelSmoothingCE | Accuracy, F1, AUROC |
| Regression | MSELoss, HuberLoss, QuantileLoss | MAE, RMSE, R2Score |
| Multi-label | BCEWithLogitsLoss, AsymmetricLoss, FocalBCELoss | HammingLoss, mAP, PerLabelAUROC |

---

## Security

See [SECURITY.md](SECURITY.md) for the full policy.  Key points:

- No hardcoded credentials anywhere in the codebase.
- W&B API key read from `WANDB_API_KEY` environment variable only.
- Loss/metric functions are pure tensor operations with no network calls.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Install in editable mode: `pip install -e ".[dev]"`
3. Run the checks locally:
   ```bash
   ruff check fujicv/ tests/
   mypy fujicv/
   pytest tests/
   detect-secrets scan
   ```
4. Open a pull request — CI will run automatically.

---

## Validated Results

| Dataset | Model | Epochs | Device | Val Accuracy |
|---------|-------|--------|--------|-------------|
| MNIST | ResNet-18 (scratch) | 5 | CPU | **98.6%** |

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full release history.

---

## License

[Apache 2.0](LICENSE) — Copyright (c) 2025 FujiCV Contributors.
