# Changelog

All notable changes to FujiCV are documented here.

---

## [1.5.0] — 2026-07-22

### New Features

**Mixup / CutMix Batch Augmentation**
- New `fujicv.data.mixup` module with `MixupCollator`, `CutMixCollator`, `MixupCutMixCollator`
- Drop-in `collate_fn` for any DataLoader — no changes to dataset or model needed
- `MixupCollator`: linearly interpolates image pairs and soft labels (Zhang et al., 2018)
- `CutMixCollator`: cuts and pastes random rectangular patches, mixes labels by patch area (Yun et al., 2019)
- `MixupCutMixCollator`: randomly selects Mixup or CutMix each batch with configurable per-method probability
- All collators output one-hot soft targets compatible with cross-entropy or BCE losses
- Exported from `fujicv.data`; 11 unit tests

**EMA (Exponential Moving Average)**
- New `fujicv.training.ema.ModelEMA` — shadow weight tracker for SOTA training pipelines
- Bias-corrected warmup schedule for the first N steps (timm/EfficientNet style)
- `update(model)` — call after every optimizer step
- `average_parameters(model)` context manager — swaps EMA weights in for eval, restores on exit
- `apply_to(model)` — permanently overwrite model with EMA weights
- `state_dict` / `load_state_dict` for checkpoint serialisation
- Exported from `fujicv.training`; 8 unit tests

**LR Warmup + Advanced Schedulers**
- New `fujicv.training.schedulers` module with `linear_warmup_schedule`, `cosine_with_warmup`, `get_scheduler`
- `cosine_with_warmup`: ViT/Swin recipe — linear ramp then cosine decay, configurable min LR ratio
- `linear_warmup_schedule`: chain any scheduler after warmup via `SequentialLR`
- `get_scheduler(name, optimizer, ...)`: factory supporting `cosine`, `cosine_warmup`, `step`, `onecycle`, `plateau`, `linear_warmup`
- Exported from `fujicv.training`; 11 unit tests

**Layer-wise LR Decay (LLRD)**
- New `fujicv.training.llrd.get_layer_wise_lr_params` — builds AdamW param groups with per-layer LR
- Infers layer depth from `blocks.N`, `layer.N`, `stage.N` naming patterns; handles stem, head, and bias/norm no-decay
- `decay_rate` controls how steeply LR falls toward the input (typical: 0.65–0.85)
- `print_llrd_summary(param_groups)` pretty-prints LR/WD/count table
- Exported from `fujicv.training`; 6 unit tests

**Model Calibration**
- New `fujicv.eval.calibration` module
- `compute_ece(confidences, correct, n_bins)` — Expected Calibration Error metric
- `TemperatureScaling` — post-hoc calibration; `fit(model, val_loader)` learns T via LBFGS on NLL; `calibrate(logits)` returns calibrated probabilities
- `reliability_diagram(confidences, correct)` — bar chart vs diagonal with ECE annotation, save or display
- Exported from `fujicv.eval`; 9 unit tests

---

## [1.4.0] — 2026-07-22

### New Features

**Stochastic Depth (DropPath)**
- New `fujicv.models.stochastic_depth` module with `DropPath` layer and `build_stochastic_depth_schedule`
- `DropPath(drop_prob)` drops entire residual branches per-sample during training; identity at eval
- `build_stochastic_depth_schedule(num_stages, max_drop_rate)` returns linearly-spaced rates for stacked models
- Integrated into `ModelBuilder` via `drop_path_rate` kwarg — passed through to timm natively
- Re-exported from `fujicv.models.custom_layers` for convenience
- 11 unit tests covering identity, drop behaviour, shape, invalid prob, schedule, and builder integration

**Knowledge Distillation**
- New `fujicv.losses.distillation` module with `DistillationLoss` and `FeatureDistillationLoss`
- `DistillationLoss(alpha, temperature)` — Hinton-style soft + hard loss; KL divergence scaled by T²
- `FeatureDistillationLoss(projector)` — MSE between student and teacher feature maps with optional projection layer
- Both losses registered in the LOSS_REGISTRY; retrievable via `get_loss("DistillationLoss", {...})`
- New `fujicv.engine.distillation_trainer.DistillationTrainer` — extends `Trainer` with teacher freezing, teacher forward pass, and distillation loss dispatch
- Teacher is automatically frozen and moved to device at construction; raises `TypeError` early if wrong loss type
- Supports full AMP, grad clipping, checkpointing — same as base `Trainer`
- 14 unit tests covering losses, registry, backward pass, trainer smoke, wrong-loss rejection, teacher frozen

**K-Fold Cross Validation**
- New `fujicv.training.kfold.KFoldTrainer` for robust model evaluation
- Factory pattern: `model_factory`, `dataset_factory`, `trainer_factory` — caller controls all hyperparameters
- Uses `StratifiedKFold` (set `stratify_col`) or plain `KFold` from scikit-learn
- Per-fold checkpoints saved under `<output_dir>/fold_N/`
- Returns `fold_histories`, `fold_metrics`, `summary` (DataFrame with mean/std/min/max), `oof_preds`, `oof_targets`
- OOF predictions initialised lazily — no need to specify logit dim in advance
- 5 unit tests covering fold count, summary shape, OOF coverage, directory creation, missing sklearn error

---

## [1.3.0] — 2026-07-21

### New Features

**Test-Time Augmentation (TTA)**
- New `fujicv.inference.tta` module with `TTAPredictor` and `tta_predict`
- `TTAPredictor` wraps any trained model and averages predictions over multiple augmented views
- 7 built-in augmentation presets: `hflip` (2 views), `hflip_vflip` (3), `rotate` (4), `hflip_rotate` (5), `brightness` (3), `standard` (6), `full` (8)
- Custom augments supported: pass any list of `fn(np.ndarray) -> np.ndarray` callables
- Two merge strategies: `mean` (default) and `max` over probability distributions
- Supports all three tasks: classification, regression, multilabel
- Three prediction entry points: `predict(image)`, `predict_dataset(paths)`, `predict_batch(dataloader)`
- `predict_proba()` returns raw probability array for downstream ensembling
- `tta_predict()` one-shot convenience function (no class instantiation needed)
- Exported from `fujicv.inference`: `from fujicv.inference import TTAPredictor, tta_predict`
- 18 unit tests, all passing

**Other**
- `Registry` added to `fujicv.__all__` for cleaner public API

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
