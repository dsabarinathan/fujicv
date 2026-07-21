# Changelog

All notable changes to FujiCV are documented here.

---

## [1.2.0] — 2026-07-21

### New Features

**HuggingFace Datasets Integration**
- New `fujicv.data.hf_dataset` module with `HFImageDataset` and `load_hf_dataset`
- `HFImageDataset` — wraps any `datasets.Dataset` object; supports PIL images, file paths, and raw arrays; handles classification (int + string labels, HF `ClassLabel` feature), regression, and multi-label tasks
- `load_hf_dataset(repo_id, ...)` — one-call download + split + wrap for any HuggingFace Hub dataset; automatically creates a val split if none exists
- Auto class-to-idx from HF `ClassLabel` feature when available
- Optional dependency: `pip install "fujicv[hf]"` installs `datasets>=2.14`
- Example script: `examples/train_hf_beans.py` (3-class plant disease, ~1K images)

**Bug Fixes**
- `colab_multilabel.ipynb`: fixed wrong import paths (`BCEWithLogitsLoss` was imported from `losses.classification` instead of `losses.multilabel`; `HammingScore`/`MeanAveragePrecision` renamed to `HammingLoss`/`mAP`; `CSVImageDataset` import and constructor corrected)

---

## [1.1.0] — 2026-07-20

### New Features

**Ordinal Regression Losses**
- `CoralLoss` — CORAL ordinal regression (Cao et al., 2020); converts rank targets to binary tasks and applies BCE across `K-1` cumulative thresholds
- `CornLoss` — CORN conditional ordinal regression (Shi et al., 2023); mask-based conditional training for each rank boundary
- Both registered in `LOSS_REGISTRY` and selectable by name

**Hyperparameter Optimisation**
- New `fujicv.hpo` module with `run_hpo(objective_fn, n_trials, direction, study_name)` wrapper around Optuna
- Optional dependency: `pip install "fujicv[hpo]"` installs `optuna>=3.0`
- Raises a clean `ImportError` with install instructions when Optuna is absent

**Multi-GPU Training**
- `Trainer` now automatically wraps the model in `nn.DataParallel` when `torch.cuda.device_count() > 1`
- Checkpoint saving correctly unwraps `.module` before serialisation

**Backbone Example Scripts**
- `examples/train_efficientnet.py` — EfficientNet-B0 on CIFAR-10 (224px, pretrained)
- `examples/train_convnext.py` — ConvNeXt-Tiny on CIFAR-10 (224px, pretrained)
- `examples/train_vit.py` — ViT-Tiny patch16/224 on CIFAR-10 (pretrained)

**Colab Notebooks**
- `examples/colab_cifar10.ipynb` — added attention map grid cell (cell 11b) using `generate_attention_grid`
- `examples/colab_regression.ipynb` — ResNet-18 regression on synthetic brightness dataset; scatter plot + residual distribution
- `examples/colab_multilabel.ipynb` — ResNet-18 multi-label on synthetic 5-label dataset; per-label AP bar chart + label co-occurrence heatmap

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
