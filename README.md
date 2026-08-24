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

## Quick Start — CIFAR-10 (auto-downloads, no setup needed)

```python
import torch
from torch.utils.data import DataLoader

import fujicv
from fujicv.data.datasets import get_default_dataset
from fujicv.data.transforms import get_train_transforms, get_val_transforms
from fujicv.engine.trainer import Trainer
from fujicv.losses.classification import CrossEntropyLoss
from fujicv.metrics.classification import Accuracy
from fujicv.models.builder import ModelBuilder

# Download CIFAR-10 automatically
train_ds, val_ds, class_to_idx = get_default_dataset(
    "cifar10", root="data",
    train_transform=get_train_transforms(32, level="medium"),
    val_transform=get_val_transforms(32),
)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=2)

model = ModelBuilder("resnet18", task="classification", num_outputs=10, image_size=32).build()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

trainer = Trainer(
    model=model, train_loader=train_loader, val_loader=val_loader,
    loss_fn=CrossEntropyLoss(),
    metrics={"accuracy": Accuracy()},
    optimizer=optimizer,
    epochs=10, task="classification", output_dir="runs/cifar10",
    monitor_metric="val_accuracy",
)
history = trainer.train()   # → best.pt  last.pt  history.csv
# Best val accuracy after 10 epochs on CIFAR-10: ~80%
```

---

## Feature Matrix

| Category | What's included |
|---|---|
| **Backbones** | Any `timm`, `torchvision`, or Hugging Face `transformers` model |
| **Task heads** | Classification · Regression · Multi-label |
| **Metric learning** | ArcFace · CosFace · Sub-center ArcFace margin heads |
| **Retrieval** | Embedding extraction · cosine/FAISS kNN index · Recall@K · mAP@K |
| **Custom layers** | LinearBNDropout · GeM Pooling · AttentionPool · SqueezeExcite |
| **Head builder** | Pluggable pooling (avg/max/gem/attention) · Linear/Dropout/LayerNorm/BatchNorm/Activation blocks |
| **Losses** | 15 losses — CrossEntropy · Focal · LabelSmoothing · CORAL · Ordinal · Huber · Quantile · … |
| **Metrics** | 16 metrics — Accuracy · F1 · AUROC · mAP · MAE · RMSE · R² · … |
| **Augmentation** | Albumentations presets · RandAugment · Mixup · CutMix |
| **Trainer** | AMP · Gradient clipping · Gradient accumulation · EMA · SWA · Model soups · Early stopping · Checkpointing · History CSV |
| **Fine-tuning** | Layer freezing · Gradual unfreezing · Frozen BN stats · LLRD |
| **LR utilities** | LR Finder · Cosine warmup · OneCycleLR · LLRD |
| **Multi-GPU** | `DistributedDataParallel` via `torchrun` (`use_ddp=True`) · rank-guarded checkpoints · all-gathered metrics |
| **HPO** | Optuna hyperparameter search with pruning (`pip install "fujicv[hpo]"`) |
| **Explainability** | Grad-CAM · Grad-CAM++ · Attention rollout · Confusion matrix |
| **Export** | ONNX · TorchScript · INT8 quantization (ONNX · dynamic PTQ · static FX PTQ) |
| **Inference** | `Predictor.from_checkpoint` · batch predict with IDs · built-in TTA · `EnsemblePredictor` |
| **Logging** | W&B · TensorBoard (offline) · MLflow · pluggable `BaseLogger` |
| **Performance** | `torch.compile` (graph optimisation, PyTorch 2.x) |
| **CV / Imbalance** | K-Fold · Stratified K-Fold · WeightedRandomSampler |
| **Distillation** | `DistillationTrainer` with temperature scaling |

---

## Installation

```bash
# Core
pip install fujicv

# With experiment logging (W&B / TensorBoard / MLflow)
pip install "fujicv[wandb]"  "fujicv[tensorboard]"  "fujicv[mlflow]"

# With ONNX export + quantization
pip install "fujicv[onnx]"

# With Optuna HPO
pip install "fujicv[hpo]"

# With Hugging Face transformers backbones
pip install "fujicv[hf-models]"

# Everything
pip install "fujicv[wandb,tensorboard,mlflow,onnx,hpo,hf-models]"

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

### Gradient accumulation (large effective batch on a small GPU)

```python
# Effective batch = batch_size × grad_accum_steps = 32 × 4 = 128
trainer = Trainer(model, ..., grad_accum_steps=4)
history = trainer.train()
```

### TensorBoard logging (offline, no account)

```python
from fujicv.engine import TensorBoardLogger

tb = TensorBoardLogger(log_dir="runs/exp1", config={"lr": 1e-3})
trainer = Trainer(model, ..., tb_logger=tb)
trainer.train()
# then:  tensorboard --logdir runs/exp1
```

### Staged fine-tuning (freeze → gradual unfreeze)

```python
from fujicv.training import freeze_backbone, GradualUnfreezing

freeze_backbone(model)                 # epoch 0: train the head only
unfreezer = GradualUnfreezing(model, unfreeze_epoch=2, layers_per_epoch=1)
for epoch in range(epochs):
    unfreezer.step(epoch)              # unfreeze one backbone block per epoch
    train_one_epoch(...)
```

### ArcFace margin head (fine-grained / retrieval)

```python
import torch.nn.functional as F
from fujicv.models import ArcMarginProduct

head = ArcMarginProduct(in_features=512, num_classes=5000, s=30.0, m=0.5)

# training — margin applied to the ground-truth class
logits = head(embeddings, labels)
loss   = F.cross_entropy(logits, labels)

# inference / retrieval — plain scaled cosine
scores = head(embeddings)              # labels=None
```

### Model soups (ensemble accuracy, single-model inference)

```python
import torch
from fujicv.training import uniform_soup, greedy_soup

states = [torch.load(p)["model_state_dict"] for p in checkpoint_paths]

uniform_soup(model, states)            # plain weight average
# or keep only ingredients that help a validation metric:
kept = greedy_soup(model, states, eval_fn=lambda m: evaluate(m, val_loader))
```

### Hugging Face backbone

```python
from fujicv.models.builder import ModelBuilder

model = ModelBuilder(
    backbone_name="google/vit-base-patch16-224",  # any HF vision repo id
    backbone_source="hf",
    task="classification", num_outputs=10, image_size=224,
).build()
```

### MLflow logging + torch.compile

```python
from fujicv.engine import MLflowLogger

mlf = MLflowLogger("fujicv-experiments", run_name="run1", params={"lr": 1e-3})
trainer = Trainer(
    model, ..., loggers=[mlf],   # pluggable — also WandbLogger/TensorBoardLogger
    compile_model=True,          # torch.compile graph optimisation (PyTorch 2.x)
)
trainer.train()
```

### INT8 quantization for edge deployment

```python
from fujicv.export import quantize_dynamic, quantize_static, measure_model_size

fp32_mb = measure_model_size(model)

# Dynamic PTQ — no calibration data required (great for transformer/MLP heads)
qmodel = quantize_dynamic(model)

# Static FX PTQ — calibrate on a few batches (best for CNNs)
qmodel = quantize_static(model, calibration_loader, backend="x86")

print(f"{fp32_mb:.1f} MB → {measure_model_size(qmodel):.1f} MB")
```

### Custom head + GeM pooling

```python
model = ModelBuilder(
    "resnet50", task="classification", num_outputs=100,
    pooling="gem",                                   # learnable Generalised-Mean pool
    custom_layers=[
        {"type": "Linear", "out_features": 512},
        {"type": "LayerNorm"},
        {"type": "Activation", "fn": "gelu"},
        {"type": "Dropout", "p": 0.2},
    ],
).build()
```

### Image retrieval (embeddings + kNN + Recall@K)

```python
from fujicv.retrieval import Embedder, RetrievalIndex, evaluate_retrieval

# 1. Extract L2-normalized embeddings from a trained model
emb, labels = Embedder(model).embed(gallery_loader, return_labels=True)

# 2. Evaluate retrieval quality
print(evaluate_retrieval(emb, labels, ks=(1, 5, 10)))   # {'recall@1': .., 'mAP@10': ..}

# 3. Nearest-neighbour search (optional FAISS backend for big galleries)
index = RetrievalIndex(emb, labels)
sims, idx, hit_labels = index.search(query_emb, k=5)
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
