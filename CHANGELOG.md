# Changelog

All notable changes to FujiCV will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-01-01

### Added
- Initial package scaffold with full support for image classification, regression,
  and multi-label classification.
- `models/` — backbone factory (timm + torchvision), task heads
  (ClassificationHead, RegressionHead, MultiLabelHead), custom layers
  (LinearBNDropout, GeM, AttentionPool, SqueezeExcite), and high-level
  ModelBuilder.
- `data/` — CSVImageDataset, stratified/random split builder, albumentations
  transform pipelines (light / medium / heavy), DataLoader factory.
- `losses/` — CrossEntropyLoss, WeightedCrossEntropyLoss, LabelSmoothingCE,
  FocalLoss, ClassBalancedLoss, MSELoss, MAELoss, HuberLoss, LogCoshLoss,
  QuantileLoss, BCEWithLogitsLoss, WeightedBCELoss, FocalBCELoss,
  AsymmetricLoss — all registered in LOSS_REGISTRY.
- `metrics/` — Accuracy, BalancedAccuracy, Precision, Recall, F1, TopKAccuracy,
  CohenKappa, MCC, AUROC, MAE, MSE, RMSE, R2Score, MAPE, PearsonCorr,
  SpearmanCorr, SubsetAccuracy, HammingLoss, mAP, PerLabelAUROC — all
  registered in METRIC_REGISTRY.
- `engine/` — Trainer with AMP, gradient clipping, checkpointing, early stopping;
  WandbLogger (env-var auth only); EarlyStopping, CheckpointCallback,
  LRSchedulerCallback.
- `eval/` — confusion matrix reports, ROC/PR curves, t-SNE embeddings,
  Grad-CAM (CNN) and attention rollout (ViT) visualisations.
- `inference/` — Predictor.from_checkpoint for single-image and batch inference.
- `export/` — ONNX export and verification utilities.
- `utils/` — generic Registry, set_seed, YAML config loader/validator.
- Example configs for classification, regression, and multi-label tasks.
- CI workflow: ruff, mypy, detect-secrets, pytest on Python 3.9–3.11.
