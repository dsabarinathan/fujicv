# API Reference

A map of the main entry points. Every class and function has a full docstring —
use `help(obj)` or your IDE for complete signatures.

## Models — `fujicv.models`

| Object | Purpose |
|---|---|
| `ModelBuilder(backbone_name, backbone_source, task, num_outputs, image_size, pooling, custom_layers)` | Assemble backbone + pooling + head. `.build()` returns the model. |
| `ArcMarginProduct`, `AddMarginProduct` / `CosMarginProduct`, `SubCenterArcMarginProduct` | Metric-learning margin heads. |
| `GeM`, `AttentionPool`, `SqueezeExcite`, `LinearBNDropout` | Custom layers. |

Assembled models expose `forward_features(x)` — the pooled, pre-head embedding.

## Training — `fujicv.engine` / `fujicv.training`

| Object | Purpose |
|---|---|
| `Trainer(...)` | Full training loop: AMP, grad accumulation, EMA, DDP, `torch.compile`, checkpointing, incremental history. |
| `DistillationTrainer(teacher, ...)` | Knowledge distillation. |
| `WandbLogger`, `TensorBoardLogger`, `MLflowLogger`, `BaseLogger` | Experiment logging (pass via `loggers=[...]`). |
| `LRFinder`, `SWA`, `SAM`, `ModelEMA` | LR range test, weight averaging, sharpness-aware minimization, EMA. |
| `KFoldTrainer` | Stratified k-fold cross-validation. |
| `freeze_backbone`, `GradualUnfreezing` | Staged fine-tuning. |
| `uniform_soup`, `greedy_soup` | Model soups. |

## Losses & metrics — `fujicv.losses` / `fujicv.metrics`

Classification: `CrossEntropyLoss`, `FocalLoss`, `LabelSmoothingCE` · `Accuracy`, `F1Score`, `AUROC`.
Regression: `MSELoss`, `HuberLoss`, `LogCoshLoss`, `QuantileLoss`, `CoralLoss`, `CornLoss` · `MAE`, `RMSE`, `R2`.
Multi-label: `BCEWithLogitsLoss`, `FocalBCELoss`, `AsymmetricLoss` · `mAP`, `PerLabelAUROC`, `HammingLoss`.

## Data — `fujicv.data`

| Object | Purpose |
|---|---|
| `get_default_dataset(name, ...)` | Built-in CIFAR-10 / MNIST (auto-download). |
| `CSVImageDataset(df, image_dir, image_col, label_col, task, ...)` | CSV-driven dataset. |
| `load_hf_dataset(name, ...)` | Hugging Face Hub datasets. |
| `get_train_transforms`, `get_val_transforms` | Albumentations presets. |
| `MixupCollator`, `CutMixCollator`, `RandAugment` | Augmentation. |
| `make_weighted_sampler` | Imbalanced sampling. |

## Inference & explainability — `fujicv.inference` / `fujicv.eval`

| Object | Purpose |
|---|---|
| `Predictor.from_checkpoint(path, model=...)` | Single / batch prediction, TTA, preserved IDs. |
| `EnsemblePredictor`, `TTAPredictor` | Ensembling and test-time augmentation. |
| `GradCAM`, `GradCAMPlusPlus`, `overlay_heatmap` | Saliency maps. |
| `plot_confusion_matrix`, `TemperatureScaling` | Diagnostics & calibration. |

## Retrieval — `fujicv.retrieval`

| Object | Purpose |
|---|---|
| `Embedder(model)` | Extract L2-normalized embeddings. |
| `RetrievalIndex(embeddings, labels)` | Cosine / FAISS kNN search. |
| `evaluate_retrieval`, `recall_at_k`, `mean_average_precision_at_k` | Retrieval metrics. |

## Export — `fujicv.export`

| Object | Purpose |
|---|---|
| `to_onnx`, `quantize_onnx` | ONNX export + INT8. |
| `export_torchscript`, `load_torchscript` | TorchScript. |
| `quantize_dynamic`, `quantize_static`, `measure_model_size` | Post-training quantization. |

## HPO — `fujicv.hpo`

| Object | Purpose |
|---|---|
| `run_hpo(objective, n_trials, pruner=...)` | Optuna search with pruning. |
| `OptunaPruningCallback` | Report intermediate values for pruning. |
