# FujiCV — Project Specification

Open-source Python package for **image classification** and **image regression**, built on
`timm` + `torchvision` pretrained backbones, with full training/experiment-tracking, rich
evaluation reporting, and explainability.

This document is the source of truth for a coding agent (e.g. local Claude Code / Sonnet agent)
to scaffold and implement the package. Each section = one implementable module. Build in the
order listed; each phase should be runnable/testable before moving to the next.

---

## 0. Package Layout

```
fujicv/
  __init__.py
  models/
    __init__.py
    backbone.py          # timm + torchvision backbone loader
    head.py               # classification / regression / multi-label heads
    custom_layers.py       # user-insertable custom layers
    builder.py             # assembles backbone + custom layer(s) + head
  data/
    __init__.py
    datasets.py            # generic ImageFolder / CSV / multi-label dataset classes
    transforms.py          # Albumentations pipelines (train/val/test)
    dataloader.py
  losses/
    __init__.py
    classification.py      # CE, Focal, Label Smoothing, etc.
    regression.py           # MSE, MAE, Huber, LogCosh, etc.
    multilabel.py           # BCE variants, Asymmetric Loss, etc.
  metrics/
    __init__.py
    classification.py
    regression.py
    multilabel.py
  engine/
    __init__.py
    trainer.py              # Trainer class: train/val loop, checkpointing
    callbacks.py             # early stopping, LR scheduler hooks, checkpoint saving
    logger.py                # W&B integration wrapper
  eval/
    __init__.py
    report.py                # classification_report, confusion matrix
    curves.py                # ROC, PR curves
    tsne.py                  # t-SNE embedding plots
    attention_map.py         # Grad-CAM / attention rollout for correct & wrong preds
    plots.py                 # train/val loss & metric curve plots
  utils/
    __init__.py
    seed.py
    config.py                # YAML/dict-based config loader
    registry.py               # name->class registries for losses/metrics/models
tests/
examples/
  configs/
    classification_example.yaml
    regression_example.yaml
    multilabel_example.yaml
  train.py
  evaluate.py
pyproject.toml
README.md
```

---

## 1. Backbones — timm + torchvision pretrained

**Goal:** one unified interface to load *any* image classification backbone from `timm` or
`torchvision.models`, pretrained, with the classifier head stripped so it can be used as a
feature extractor.

- `fujicv.models.backbone.build_backbone(name: str, source: Literal["timm","torchvision"]="timm", pretrained=True, in_chans=3, features_only=False, out_indices=None)`
  - For `timm`: use `timm.create_model(name, pretrained=pretrained, num_classes=0, ...)` to get a pooled feature vector, or `features_only=True` for multi-scale feature maps (needed for attention maps / Grad-CAM on conv nets).
  - For `torchvision`: use `torchvision.models.get_model(name, weights="DEFAULT" if pretrained else None)`, then strip the final `fc`/`classifier` to expose `in_features`.
  - Return a dict/dataclass: `{model: nn.Module, out_features: int, out_indices/feature_info}`.
- Provide `list_available_backbones(source)` wrapping `timm.list_models(pretrained=True)` and torchvision's model registry, with optional substring filter.
- Support both CNNs (resnet, efficientnet, convnext) and transformers (vit, swin, deit) — since attention-map extraction differs per architecture family, tag each backbone with `arch_family: "cnn" | "vit"` (infer from timm's `default_cfg`/model name heuristics) so the explainability module knows which method to use (Grad-CAM for CNN, attention-rollout for ViT).

---

## 2. Custom Layer Insertion

**Goal:** let the user inject one or more custom `nn.Module` layers between the backbone and
the head, without hand-editing backbone internals.

- `fujicv.models.builder.ModelBuilder`:
  ```python
  ModelBuilder(
      backbone_name="convnext_tiny",
      backbone_source="timm",
      pretrained=True,
      custom_layers: list[nn.Module] | None = None,   # inserted in sequence after backbone pooled output
      task="classification" | "regression" | "multilabel" | "multiclass",
      num_outputs=...,
      head_kwargs={...},
  )
  ```
- Internally: `nn.Sequential(backbone_features -> *custom_layers -> head)`.
- `custom_layers` accepts arbitrary `nn.Module`s (e.g. an extra `nn.Linear`+`nn.BatchNorm1d`+`nn.Dropout` block, an attention pooling module, a projection layer for embeddings). Validate in/out dim compatibility at build time and raise a clear error if shapes mismatch (do a dummy forward pass with a `torch.zeros(1,3,H,W)` tensor at construction to sanity check).
- Also support insertion **inside** the backbone at a named layer (advanced/optional) via forward hooks, for users who want to splice into intermediate feature maps rather than only after pooling. Document as an advanced/optional feature (v2), not required for v1.

---

## 2.5 Dataset Config & CSV-Driven Data Loading

**Goal:** the user should be able to describe an entire dataset + experiment purely through a
YAML file, and separately just hand over a CSV with image paths + labels — FujiCV builds the
train/val/test splits and dataloaders automatically for both classification and regression.

### 2.5.1 Dataset/Experiment YAML

A single YAML file fully describes: where the data lives, how to augment it, what loss/model
output to use. This is the same config referenced in Section 7, but the `dataset` block is
specified in detail here:

```yaml
dataset:
  task: classification              # classification | multiclass | multilabel | regression
  csv_path: data/labels.csv          # single CSV; splits derived per §2.5.2
  image_dir: data/images/            # root dir images are resolved relative to
  image_col: filename
  label_col: label                   # classification: class name/int | regression: float target(s)
                                       # multilabel: comma/pipe-separated string OR list of binary cols
  split_col: split                   # optional — if present, must contain train/val/test values
  split_ratios: {train: 0.7, val: 0.15, test: 0.15}   # used only if split_col absent
  stratify: true                     # stratify split by label_col (classification/multilabel only)
  random_seed: 42
  image_size: 224
augmentation:
  name: medium                       # light | medium | heavy | custom
  custom_pipeline: null              # optional path to a python file exposing get_transforms()
train:
  lr: 3e-4
  optimizer: {name: adamw}
  scheduler: {name: cosine}
  epochs: 30
  batch_size: 32
loss:
  name: focal_loss                   # any registered name, incl. user-registered custom losses
  kwargs: {gamma: 2.0}
model:
  backbone: {name: convnext_tiny, source: timm, pretrained: true}
  custom_layers: []
  num_outputs: 10                    # classes for classification/multilabel, target-dim for regression
```

- `fujicv.utils.config.load_dataset_config(path)` parses and validates this block (required
  keys per task type, e.g. `regression` requires `label_col` to resolve to numeric column(s);
  `multilabel` requires either a delimiter-separated label column or a list of one-hot columns).
- Validation errors must be explicit and human-readable (e.g. "label_col 'target' not found in
  CSV columns: [...]" or "task=regression but label_col contains non-numeric values").

### 2.5.2 CSV → Train/Val/Test Split Logic

`fujicv.data.datasets.build_splits(dataset_cfg)`:

- **If `split_col` is present in the CSV** (e.g. a `split` column already marked `train`/`val`/`test`
  per row): use it directly, no re-splitting. Validate all three values are present at least once;
  warn if `test` is missing (allowed — test can be optional, val is not).
- **If `split_col` is absent**: perform the split using `split_ratios`:
  - **Classification / multiclass / multilabel** — use `sklearn.model_selection.train_test_split`
    with `stratify=label_col` (or a combined multilabel stratification via
    `iterstrat.ml_stratifiers.MultilabelStratifiedShuffleSplit` when `task=multilabel`) so class
    balance is preserved across train/val/test.
  - **Regression** — plain random split (no stratification by default) using `random_seed`;
    optionally support binning the continuous target into quantile buckets and stratifying on
    the buckets if the user sets `stratify: true` for regression (useful for skewed targets).
  - Persist the resulting split assignment back out as `output_dir/split_assignment.csv` (adds a
    `split` column to the original rows) so runs are reproducible and auditable.
- Both paths return three `pandas.DataFrame`s: `train_df, val_df, test_df` (test_df may be empty).

### 2.5.3 Dataset Classes

- `fujicv.data.datasets.CSVImageDataset(df, image_dir, image_col, label_col, task, transform)`:
  - Single unified class for classification (returns int class index — builds/reuses a
    `class_to_idx` mapping), regression (returns float or float-vector target), and multilabel
    (returns a multi-hot float vector).
  - Handles missing/corrupt image files gracefully: skip with a logged warning at dataset-build
    time rather than failing mid-epoch (pre-validate file existence when the dataset is built).
  - `class_to_idx` / target scaler (for regression, optional `StandardScaler`/`MinMaxScaler` on
    the label column, fit on train only, applied to val/test) is saved alongside checkpoints so
    inference-time decoding matches training.
- `fujicv.data.dataloader.build_dataloaders(train_df, val_df, test_df, dataset_cfg, augmentation_cfg)`
  → returns `train_loader, val_loader, test_loader` (test may be `None`), wiring in the correct
  Albumentations pipeline per split (train gets augmentation, val/test get only resize+normalize).

### 2.5.4 Also Support Pre-Split Folder Layout (no CSV)

For users who already have `train/`, `val/`, `test/` folders (e.g. `ImageFolder`-style, one
subfolder per class) — keep supporting this as an alternative to the CSV path (classification
only, since folder-per-class doesn't extend cleanly to regression):

```yaml
dataset:
  task: classification
  format: imagefolder
  train_dir: data/train/
  val_dir: data/val/
  test_dir: data/test/     # optional
```

Both entry points (CSV-based and folder-based) converge on the same `train_loader, val_loader,
test_loader` output so the rest of the pipeline (Trainer, eval, reporting) is agnostic to which
was used.

---

## 3. Training Loop, W&B logging, Albumentations

### 3.1 Augmentation
- All transforms via `albumentations` + `albumentations.pytorch.ToTensorV2`.
- Provide preset pipelines in `fujicv.data.transforms`:
  - `get_train_transforms(image_size, level="light"|"medium"|"heavy")`
  - `get_val_transforms(image_size)`
  - Standard set: `Resize`, `HorizontalFlip`, `RandomBrightnessContrast`, `ShiftScaleRotate`, `CoarseDropout`, `Normalize` (ImageNet mean/std by default, override via config), `ToTensorV2`.
  - Let users pass their own `A.Compose(...)` object to override presets entirely.

### 3.2 Trainer
- `fujicv.engine.trainer.Trainer`:
  - Args: model, train_loader, val_loader, loss_fn, metrics (list), optimizer, scheduler, device, epochs, task type, output_dir, wandb config.
  - Standard loop: per-epoch train step + val step, accumulate loss + each metric, log per-batch and per-epoch.
  - Checkpointing: save best (by a configurable monitor metric) + last checkpoint.
  - Mixed precision (`torch.cuda.amp`) support, gradient clipping, resume-from-checkpoint.
  - Return a `History` object storing per-epoch train/val loss and metrics for later plotting (see Section 4).

### 3.3 Weights & Biases integration
- `fujicv.engine.logger.WandbLogger`:
  - Init run with project/entity/config from user.
  - Log per-epoch: `train/loss`, `val/loss`, `train/<metric>`, `val/<metric>` for every configured metric.
  - Log learning rate, epoch time.
  - Log final artifacts: classification report, confusion matrix image, ROC curve image, t-SNE plot image, attention-map grid, best checkpoint (as W&B artifact).
  - Must be fully optional — if `use_wandb=False` or `wandb` not installed/configured, everything else still works and plots are saved locally instead.

---

## 4. Custom Plots & Final Reporting

Implement each of these as a standalone function taking a `History` object and/or model +
dataloader, returning a `matplotlib` Figure (so they work identically whether saved to disk or
logged to W&B):

- `plots.plot_loss_curves(history)` — train vs val loss over epochs.
- `plots.plot_metric_curves(history, metric_name)` — train vs val for a given metric, one call per tracked metric.
- `report.classification_report(y_true, y_pred, class_names)` — sklearn's `classification_report` as text + a rendered table figure, plus `confusion_matrix` heatmap.
- `curves.plot_roc_curve(y_true, y_probs, class_names, multi_class="ovr")` — supports binary, multiclass (one-vs-rest, per-class + micro/macro average), and multilabel.
- `curves.plot_pr_curve(...)` — precision-recall, same multi-class support (useful for imbalanced sets — nice complement to ROC).
- `tsne.plot_tsne(embeddings, labels, class_names)` — extract penultimate-layer embeddings for a val/test set, run `sklearn.manifold.TSNE`, scatter-plot colored by class.
- Everything should run end-to-end from a single `evaluate.py` entrypoint that takes a trained checkpoint + val/test loader and produces all of the above into `output_dir/reports/`.

---

## 5. Attention Maps for Correct vs. Wrong Predictions

**Goal:** after evaluation, generate an attention/saliency visualization grid split into
"correctly classified" and "misclassified" examples, to help debug the model.

- `fujicv.eval.attention_map`:
  - For CNN backbones: Grad-CAM (or Grad-CAM++) using the last conv feature map.
  - For ViT/Swin backbones: attention rollout or attention-map visualization from the last transformer block.
  - `generate_attention_grid(model, dataloader, arch_family, n_correct=8, n_wrong=8, class_names=None)` → saves/returns a figure with two rows/sections: correct predictions (image + overlay + predicted/true label) and wrong predictions (image + overlay + predicted vs true label).
  - Should work for classification and multilabel (top-1 predicted label used for CAM target in multilabel case, or let user pick which label to target).
  - Regression: optional — use saliency/Grad-CAM w.r.t. the scalar output instead of a class logit (document as best-effort, since "correct vs wrong" doesn't map as cleanly — bucket by prediction error, e.g. top-K best vs top-K worst absolute error).
  - This is explicitly called out as an **optional, user-triggerable** step (`evaluate.py --with-attention-maps`), not run by default (compute cost).

---

## 6. Losses & Metrics — full coverage across tasks

Implement as a registry (`fujicv.utils.registry`) so users/config files can select by string name,
and so custom losses/metrics can be registered by users too (`@register_loss("my_loss")`).

### 6.1 Classification (binary / multiclass, single-label)
- Losses: CrossEntropy, Weighted CrossEntropy, Label Smoothing CE, Focal Loss, Class-Balanced Loss.
- Metrics: Accuracy, Balanced Accuracy, Precision/Recall/F1 (macro/micro/weighted), Top-K Accuracy, Cohen's Kappa, MCC, AUROC (ovr/ovo), Confusion Matrix.

### 6.2 Multi-label classification
- Losses: BCEWithLogits, Weighted BCE, Focal Loss (multi-label variant), Asymmetric Loss (ASL).
- Metrics: per-label Precision/Recall/F1 (macro/micro/samples-averaged), Subset Accuracy (exact match), Hamming Loss, mAP (mean average precision), per-label AUROC.

### 6.3 Regression
- Losses: MSE, MAE (L1), Huber/SmoothL1, LogCosh, Quantile/Pinball Loss.
- Metrics: MAE, MSE, RMSE, R², MAPE, Pearson/Spearman correlation.

### 6.4 Ordinal / multi-output regression (nice-to-have, note as v2)
- CORAL/CORN ordinal loss, multi-target MSE with per-target weighting.

### 6.5 Custom Loss Functions (user-defined, pip-installable package)

Since FujiCV ships as a **pip-installable package** (not a private repo), the loss/metric
registry must let users register their own custom loss functions/classes without forking the
library, and without ever needing to hardcode anything sensitive.

- `fujicv.losses.registry.register_loss(name: str)` — decorator users apply to their own
  `nn.Module` or callable to add it to the global registry, e.g.:
  ```python
  from fujicv.losses.registry import register_loss
  import torch.nn as nn

  @register_loss("my_custom_loss")
  class MyCustomLoss(nn.Module):
      def __init__(self, alpha=0.5):
          super().__init__()
          self.alpha = alpha
      def forward(self, preds, targets):
          ...
          return loss
  ```
  Then selectable from config exactly like built-ins: `loss: {name: my_custom_loss, kwargs: {alpha: 0.7}}`.
- Same pattern for custom metrics (`fujicv.metrics.registry.register_metric`).
- Also support passing an already-instantiated loss object directly to `Trainer(loss_fn=...)`,
  bypassing the registry/config entirely, for users who prefer plain Python over YAML.
- Document this clearly in the README as the primary extension point for the library.

**Security / no-secrets policy (applies to the whole package, not just losses):**
- The loss/metric modules (and the package as a whole) must **never read, log, serialize, or
  hardcode** API keys, tokens, credentials, or any other confidential data. This includes:
  - No hardcoded W&B API keys, cloud storage keys, or dataset URLs with embedded credentials
    anywhere in `fujicv/` source code, example configs, or tests.
  - W&B auth is the user's responsibility via `wandb login` / `WANDB_API_KEY` env var — the
    `WandbLogger` wrapper must only ever *read* an already-configured API key from the
    environment (never accept it as a config/YAML field, never print or log it, never write it
    to checkpoints or artifacts).
  - Loss/metric functions operate purely on tensors (`preds`, `targets`) — they must not accept,
    store, or transmit any data outside the local process (no telemetry, no network calls) unless
    explicitly and separately implemented as an opt-in logging integration (W&B/Trainer only).
  - No user PII, file paths with usernames, or dataset contents should ever be embedded in
    checkpoints, config dumps, or W&B-logged artifacts by default.
  - Add a `SECURITY.md` at the repo root stating this policy, and a CI lint step (e.g. `gitleaks`
    or `detect-secrets`) to catch accidental secret commits before publishing to PyPI.

All losses/metrics should:
- Accept raw logits (apply softmax/sigmoid internally where relevant) for consistent trainer usage.
- Be selectable in the YAML config purely by name plus kwargs, e.g.:
  ```yaml
  loss:
    name: focal_loss
    kwargs: {gamma: 2.0, alpha: 0.25}
  metrics:
    - {name: f1, kwargs: {average: macro}}
    - {name: accuracy}
    - {name: auroc, kwargs: {multi_class: ovr}}
  ```

---

## 7. Config-Driven Usage (example end-to-end flow)

```yaml
task: classification          # classification | multiclass | multilabel | regression
backbone: {name: convnext_tiny, source: timm, pretrained: true}
custom_layers:
  - {type: linear_bn_dropout, out_features: 256, dropout: 0.3}
head: {num_outputs: 10}
data:
  train_csv: data/train.csv
  val_csv: data/val.csv
  image_size: 224
  augmentation_level: medium
train:
  epochs: 30
  batch_size: 32
  optimizer: {name: adamw, lr: 3e-4}
  scheduler: {name: cosine}
  loss: {name: cross_entropy}
  metrics: [{name: accuracy}, {name: f1, kwargs: {average: macro}}]
  wandb: {project: fujicv-demo, entity: null, use_wandb: true}
output_dir: runs/exp001
```

Then:
```bash
python examples/train.py --config examples/configs/classification_example.yaml
python examples/evaluate.py --config examples/configs/classification_example.yaml --checkpoint runs/exp001/best.pt --with-attention-maps
```

---

## 8. Build Order for the Agent

1. Package skeleton + `pyproject.toml` (deps: `torch`, `torchvision`, `timm`, `albumentations`, `wandb`, `scikit-learn`, `matplotlib`, `seaborn`, `pandas`, `pyyaml`, `grad-cam` or custom Grad-CAM impl).
2. `models/backbone.py` + `models/custom_layers.py` + `models/head.py` + `models/builder.py` — with a unit test that builds 2-3 backbones (1 CNN, 1 ViT) and runs a dummy forward pass.
3. `data/transforms.py` + `data/datasets.py` + `data/dataloader.py`.
4. `losses/*` + `metrics/*` + registries.
5. `engine/trainer.py` + `engine/logger.py` + `engine/callbacks.py` — test on a tiny toy dataset (e.g. CIFAR-10 subset) end-to-end.
6. `eval/plots.py`, `eval/report.py`, `eval/curves.py`, `eval/tsne.py`.
7. `eval/attention_map.py` (Grad-CAM for CNN, rollout for ViT).
8. `examples/train.py`, `examples/evaluate.py`, example YAML configs, README with quickstart.
9. Tests + CI (GitHub Actions: lint + pytest on push).

---

## 9. Non-Goals for v1
- Object detection / segmentation (classification & regression only).
- Distributed/multi-GPU training (single-GPU/CPU only for v1; note as future work).
- Hyperparameter search / AutoML.

---

## 10. Suggested Improvements (added for completeness)

These weren't explicitly requested but round out the package for real-world use and safe
publishing to PyPI.

### 10.1 Inference / Prediction Module
The spec covers train + evaluate, but not "run the trained model on new, unlabeled images."
Add `fujicv.inference.predictor.Predictor`:
- `Predictor.from_checkpoint(path)` — loads model architecture + weights + `class_to_idx`
  mapping / regression label scaler saved alongside the checkpoint (per §2.5.3), so decoding is
  automatic and consistent with training.
- `.predict(image_or_path_or_folder)` → returns class label + probability (classification),
  scalar/vector in original units (regression, auto-inverse-scaled), or multi-hot labels above a
  configurable threshold (multilabel).
- `.predict_batch(dataloader)` for bulk inference on a folder/CSV of new images, writing results
  to a CSV (`image, prediction, confidence`).
- This is the natural "day 2" entrypoint once a model is trained — without it, the package only
  supports experimentation, not actual deployment/use of the trained model.

### 10.2 Local (no-W&B) Logging Fallback
Section 3.3 already says W&B must be optional — make this concrete: when `use_wandb=False`,
`Trainer` should still write a `history.csv` (one row per epoch, all train/val loss + metrics)
and all plots (Section 4) to `output_dir/reports/` automatically. This way the package is fully
usable offline / without any external account, and W&B is purely an add-on, not a dependency for
core functionality.

### 10.3 Reproducibility
- `fujicv.utils.seed.set_seed(seed)` — seeds `random`, `numpy`, `torch`, and sets
  `torch.backends.cudnn.deterministic = True` (with a documented perf trade-off note).
- On every run, dump the fully-resolved config (after defaults are applied) to
  `output_dir/resolved_config.yaml`, plus package version and git commit hash (if run from a
  repo) — so any run's exact settings can be reconstructed later.

### 10.4 Model Export
- `fujicv.export.to_onnx(model, path, input_size)` — export a trained model to ONNX for
  deployment outside PyTorch. Include a round-trip test (ONNX Runtime output matches PyTorch
  output within tolerance) in the test suite.
- Optional: TorchScript export as a lighter-weight alternative.

### 10.5 Packaging & Release Hygiene (PyPI-specific)
Since this ships as a public pip package:
- `pyproject.toml` with pinned minimum versions for `torch`/`timm`/`albumentations` (avoid
  silent breakage from upstream API changes), and clearly separated `[project.optional-dependencies]`
  for `wandb`, `onnx`, `dev`/`test` extras — so a minimal `pip install fujicv` doesn't force
  every optional dependency on every user.
- Semantic versioning (`0.1.0` to start), with a `CHANGELOG.md`.
- `LICENSE` (MIT/Apache-2.0 — pick one; note that "Fuji" as a name carries a mild trademark risk
  as flagged earlier, but no code/legal blocker for shipping under an OSS license).
- CI: lint (`ruff`), type-check (`mypy` on public APIs at least), tests (`pytest`), secret scan
  (`detect-secrets`/`gitleaks`, per §6.5), then publish via PyPI Trusted Publishing (GitHub
  Actions OIDC — no long-lived PyPI token stored as a repo secret).
- `README.md` quickstart should mirror Section 7's example end-to-end so a first-time user can
  copy-paste from `pip install` to a finished report in under 10 lines.
