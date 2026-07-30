# FujiCV

**Lightweight image classification & regression for PyTorch — from prototype to production in 10 lines.**

[![PyPI version](https://img.shields.io/pypi/v/fujicv?color=blue)](https://pypi.org/project/fujicv/)
[![PyPI downloads](https://img.shields.io/pypi/dm/fujicv)](https://pypi.org/project/fujicv/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![CI](https://github.com/dsabarinathan/fujicv/actions/workflows/ci.yml/badge.svg)](https://github.com/dsabarinathan/fujicv/actions)
[![GitHub Discussions](https://img.shields.io/github/discussions/dsabarinathan/fujicv)](https://github.com/dsabarinathan/fujicv/discussions)

---

> **Try it in your browser →** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dsabarinathan/fujicv/blob/main/examples/quickstart.ipynb)

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
It is designed for: **rapid prototyping, Kaggle competitions, students, and researchers** who need results — not boilerplate.

---

## 12-Line Quick Start

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
    metrics={"accuracy": get_metric("Accuracy"), "f1": get_metric("F1")},
    optimizer=optim.AdamW(model.parameters(), lr=3e-4),
    epochs=30, task="classification", output_dir="outputs/",
)
history = trainer.train()   # → best.pt  last.pt  history.csv
```

---

## Feature Matrix

| Category | What's included |
|---|---|
| **Backbones** | Any `timm` or `torchvision` model — 700+ architectures |
| **Task heads** | Classification · Regression · Multi-label |
| **Custom layers** | LinearBNDropout · GeM Pooling · AttentionPool · SqueezeExcite |
| **Losses** | 15 losses — CrossEntropy · Focal · LabelSmoothing · CORAL · Ordinal · Huber · Quantile · … |
| **Metrics** | 16 metrics — Accuracy · F1 · AUROC · mAP · MAE · RMSE · R² · … |
| **Augmentation** | Albumentations presets · RandAugment · Mixup · CutMix |
| **Trainer** | AMP · Gradient clipping · EMA · SWA · Early stopping · Checkpointing · History CSV |
| **LR utilities** | LR Finder · Cosine warmup · OneCycleLR · LLRD |
| **Multi-GPU** | `DistributedDataParallel` via `torchrun` (`use_ddp=True`) |
| **HPO** | Optuna hyperparameter search with pruning (`pip install "fujicv[hpo]"`) |
| **Explainability** | Grad-CAM · Grad-CAM++ · Attention rollout · Confusion matrix |
| **Export** | ONNX · ONNX INT8 quantization · TorchScript trace/script |
| **Inference** | `Predictor.from_checkpoint` · `EnsemblePredictor` · TTA |
| **Logging** | W&B (`WANDB_API_KEY` env var only) |
| **CV / Imbalance** | K-Fold · Stratified K-Fold · WeightedRandomSampler |
| **Distillation** | `DistillationTrainer` with temperature scaling |

---

## Installation

```bash
# Core
pip install fujicv

# With W&B logging
pip install "fujicv[wandb]"

# With ONNX export + quantization
pip install "fujicv[onnx]"

# With Optuna HPO
pip install "fujicv[hpo]"

# Everything
pip install "fujicv[wandb,onnx,hpo]"

# Dev / testing
pip install "fujicv[dev]"
```

> **Install PyTorch separately** following [pytorch.org/get-started](https://pytorch.org/get-started/locally/) to pick the correct CUDA build.

---

## Selected Code Recipes

### Grad-CAM visualization

```python
from fujicv.eval.gradcam import GradCAM, overlay_heatmap
import torchvision.transforms as T
from PIL import Image

cam    = GradCAM(model, target_layer=model.backbone.layer4[-1])
image  = T.ToTensor()(Image.open("dog.jpg")).unsqueeze(0)
heatmap = cam.generate(image)
result  = overlay_heatmap(image.squeeze(), heatmap)
result.save("cam_result.png")
```

### Optuna HPO with pruning

```python
from fujicv.hpo import run_hpo

def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    wd = trial.suggest_float("weight_decay", 0, 1e-2)
    # ... build model + trainer, return val_accuracy

result = run_hpo(objective, n_trials=30, pruner="median")
print(result["best_params"])
```

### TorchScript export for production

```python
from fujicv.export import export_torchscript, verify_torchscript
import torch

scripted = export_torchscript(model, "model.pt", torch.randn(1, 3, 224, 224))
verify_torchscript(scripted, torch.randn(1, 3, 224, 224), original_model=model)
```

### LR Finder

```python
from fujicv.training import LRFinder

finder = LRFinder(model, optimizer, criterion)
finder.range_test(train_loader, start_lr=1e-7, end_lr=10, num_iter=100)
finder.plot()
best_lr = finder.suggestion()
finder.reset()   # restore model before training
```

### Stochastic Weight Averaging

```python
from fujicv.training import SWA

swa = SWA(model, swa_lr=1e-4)
for epoch in range(total_epochs):
    train_one_epoch(...)
    if epoch >= swa_start:
        swa.update()
swa.finalize(train_loader)   # update BN stats
evaluate(swa.averaged_model, val_loader)
```

### Multi-GPU training (DDP)

```bash
# Launch with torchrun
torchrun --nproc_per_node=4 train.py
```

```python
# In train.py, just add use_ddp=True
trainer = Trainer(model, ..., use_ddp=True)
```

---

## Package Layout

```
fujicv/
  models/       backbone factory · heads · custom layers · ModelBuilder
  data/         CSVImageDataset · transforms · Mixup/CutMix · RandAugment · sampler
  losses/       15 losses + LOSS_REGISTRY
  metrics/      16 metrics + METRIC_REGISTRY
  engine/       Trainer · DistillationTrainer · callbacks · WandbLogger
  training/     EMA · SWA · SAM · LR Finder · schedulers · LLRD · K-Fold
  eval/         Grad-CAM · confusion matrix · calibration · Ensemble
  hpo/          Optuna search + pruning + visualization
  export/       ONNX · ONNX quantization · TorchScript
  inference/    Predictor · EnsemblePredictor · TTA
  utils/        Registry · set_seed · config loader
tests/          255 tests covering every module
```

---

## Validated Results

| Dataset | Model | Epochs | Val Accuracy |
|---|---|---|---|
| MNIST | ResNet-18 (scratch) | 5 | **98.6%** |

*More benchmarks coming — see [GitHub Discussions](https://github.com/dsabarinathan/fujicv/discussions).*

---

## Supported Tasks

| Task | Loss examples | Metric examples |
|---|---|---|
| Classification / Multiclass | CrossEntropyLoss · FocalLoss · LabelSmoothingCE | Accuracy · F1 · AUROC |
| Regression | MSELoss · HuberLoss · QuantileLoss | MAE · RMSE · R² |
| Multi-label | BCEWithLogitsLoss · AsymmetricLoss · FocalBCELoss | HammingLoss · mAP · PerLabelAUROC |

---

## Security

- No hardcoded credentials anywhere in the codebase.
- W&B API key read from `WANDB_API_KEY` environment variable only.
- `detect-secrets` pre-commit hook blocks accidental credential commits.

See [SECURITY.md](SECURITY.md) for the full policy.

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide.

```bash
# Quick setup
git clone https://github.com/dsabarinathan/fujicv.git
cd fujicv && pip install -e ".[dev]"
pre-commit install
pytest   # all 255 tests should pass
```

---

## Community

- **Questions / ideas**: [GitHub Discussions](https://github.com/dsabarinathan/fujicv/discussions)
- **Bug reports**: [GitHub Issues](https://github.com/dsabarinathan/fujicv/issues)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for full release history.

---

## License

[Apache 2.0](LICENSE) — Copyright (c) 2025 FujiCV Contributors.
