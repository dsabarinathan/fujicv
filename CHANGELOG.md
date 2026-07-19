# Changelog

All notable changes to FujiCV are documented here.

---

## [1.0.0] — 2026-07-19

First stable release. Validated end-to-end on MNIST (98.6% val accuracy, 5 epochs, CPU).

### Features

**Models**
- Unified backbone loader for `timm` and `torchvision` — supports ResNet, EfficientNet, ConvNeXt, ViT, Swin, DeiT, and 1000+ architectures
- Classification, regression, and multi-label heads with configurable dropout
- Custom layer insertion: `LinearBNDropout`, `GeM`, `AttentionPool`, `SqueezeExcite`
- `ModelBuilder` validates full model with a dummy forward pass at construction time
- Auto device detection: CUDA → MPS (Apple Silicon) → CPU with informative logging

**Data**
- `CSVImageDataset` — unified dataset for classification, regression, and multi-label tasks
- Automatic stratified train/val/test splits, saves reproducible `split_assignment.csv`
- ImageFolder support for pre-split directory layouts
- Built-in datasets: `get_default_dataset("mnist")` (11 MB) and `get_default_dataset("cifar10")`
- Albumentations pipelines: `light`, `medium`, `heavy` augmentation presets

**Losses** (13 total, all registry-selectable by name)
- Classification: `CrossEntropyLoss`, `WeightedCrossEntropyLoss`, `LabelSmoothingCE`, `FocalLoss`, `ClassBalancedLoss`
- Regression: `MSELoss`, `MAELoss`, `HuberLoss`, `LogCoshLoss`, `QuantileLoss`
- Multi-label: `BCEWithLogitsLoss`, `WeightedBCELoss`, `FocalBCELoss`, `AsymmetricLoss`

**Metrics** (16 total, all registry-selectable by name)
- Classification: `Accuracy`, `BalancedAccuracy`, `Precision`, `Recall`, `F1`, `TopKAccuracy`, `CohenKappa`, `MCC`, `AUROC`
- Regression: `MAE`, `MSE`, `RMSE`, `R2Score`, `MAPE`, `PearsonCorr`, `SpearmanCorr`
- Multi-label: `SubsetAccuracy`, `HammingLoss`, `mAP`, `PerLabelAUROC`

**Training Engine**
- `Trainer` with AMP, gradient clipping, best/last checkpointing, early stopping
- Auto `history.csv` when W&B is not used
- `WandbLogger` — reads `WANDB_API_KEY` from environment only, fully optional
- Callbacks: `EarlyStopping`, `CheckpointCallback`, `LRSchedulerCallback`

**Evaluation**
- Loss/metric training curves, confusion matrix heatmap, ROC/PR curves, t-SNE plots
- Grad-CAM (CNN) and attention rollout (ViT) for correct vs. wrong prediction grids

**Inference & Export**
- `Predictor.from_checkpoint(path)` for single-image and batch inference
- ONNX export with round-trip numerical verification

**Config & Utils**
- YAML-driven experiment config with validation
- `@register_loss` / `@register_metric` decorators for user-defined extensions
- `set_seed` for full reproducibility; resolved config saved every run

**Examples**
- `train.py`, `evaluate.py` CLI scripts
- `train_mnist.py` quickstart → **98.6% val accuracy in 5 epochs on CPU**
- `train_cifar10.py` quickstart
- 3 example YAML configs (classification, regression, multi-label)

**Security & CI**
- Apache 2.0 license
- No hardcoded credentials; W&B key via environment only
- GitHub Actions: ruff lint + pytest (Python 3.10, 3.11)

### Validated Results
| Dataset | Model | Epochs | Device | Val Accuracy |
|---------|-------|--------|--------|-------------|
| MNIST   | ResNet-18 (scratch) | 5 | CPU | **98.6%** |

---

## [0.1.0] — 2026-07-18

Initial scaffold release.
