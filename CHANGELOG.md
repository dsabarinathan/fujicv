# Changelog

All notable changes to FujiCV are documented here.

---

## [1.16.0] — 2026-08-30

### New: Generic modern HF image-encoder support

- **Vision-tower extraction** — `backbone_source="hf"` now works with multimodal
  encoders (CLIP, SigLIP, BLIP, …). `_resolve_vision_encoder` unwraps the
  `.vision_model` / `.vision_tower` / `.visual` submodule so the image tower can
  be used standalone; plain vision models (ViT, DINOv2, ConvNeXt) pass through
  unchanged.
- **Robust output extraction** — the wrapper now falls back
  `last_hidden_state` → `pooler_output` → raw tensor/tuple, covering token,
  pooled, and feature-map encoders.
- **`get_hf_transforms(model_name, train=...)`** — builds albumentations
  pipelines from the encoder's own `AutoImageProcessor` (correct mean/std/size).
  Using the wrong normalization badly degrades pretrained encoders like CLIP
  (≈0.48/0.26) and SigLIP (0.5/0.5); this makes it automatic.
- `get_train_transforms` / `get_val_transforms` now accept custom `mean`/`std`.

### New: Qualitative inspection plots (`fujicv.eval`)

- `plot_image_grid`, `plot_predictions` (green/red correct/wrong),
  `plot_top_losses` (fast.ai-style worst-error triage), `plot_class_distribution`
  (imbalance check, multi-split), `plot_confidence_histogram`,
  `plot_augmentations` (visualize a transform pipeline). All headless-safe.

### Tests

- +25 tests (`test_hf_encoder.py`, `test_inspect_plots.py`) — HF routing tested
  with stubs (runs without transformers). 375 passing.

---

## [1.15.1] — 2026-08-25

### Bug Fixes

- **Checkpoint loading on PyTorch ≥ 2.6 (`weights_only`)** — `torch.load` defaults
  to `weights_only=True` on modern PyTorch, which refuses to unpickle the
  non-tensor objects FujiCV checkpoints embed (`History`, `class_to_idx`,
  `task`). `Trainer(resume_from=...)`, `Predictor.from_checkpoint`, and model
  soups now pass `weights_only=False` explicitly (these are the user's own
  trusted checkpoints). Previously raised `UnpicklingError` on load.

### CI / Docs

- **Fixed the docs build** (`mkdocs build --strict`) — the `nav` referenced 13
  pages that didn't exist. Added the missing guides (Classification, Regression,
  Multi-Label, Configuration) and a hand-written API Reference index; removed the
  unused mkdocstrings plugin (which needed imports the docs job doesn't have).
- These two issues had been failing every CI run (tests + docs) despite the
  suite passing locally on older torch.

### Tests

- +2 regression tests (`test_checkpoint_compat.py`) that simulate torch ≥ 2.6's
  strict `weights_only=True` default on any torch version. 350 passing.

---

## [1.15.0] — 2026-08-25

### New Feature: Image-Retrieval Toolkit

Completes the metric-learning stack (ArcFace/CosFace/Sub-center heads + GeM
pooling added in 1.11.0) with the tools to actually *use* learned embeddings —
`fujicv.retrieval`:

- **`Embedder`** — extract pre-head embeddings from a trained model. Uses the
  new `model.forward_features(x)` when available (all `ModelBuilder` models now
  expose it), else falls back to the plain forward output. L2-normalizes by
  default and unwraps `torch.compile`.
- **`RetrievalIndex`** — cosine nearest-neighbour search over a gallery. Pure
  PyTorch matmul backend by default; optional FAISS (`pip install
  "fujicv[retrieval]"`) for large galleries. `search(queries, k)` returns
  similarities, indices, and gallery labels.
- **Retrieval metrics** — `recall_at_k`, `precision_at_k`,
  `mean_average_precision_at_k`, and an `evaluate_retrieval` report. Standard
  deep-metric-learning definitions; support both leave-one-out (single set) and
  query-vs-gallery modes.

### Model change

- `ModelBuilder` models (`_AssembledModel`) now expose **`forward_features(x)`**
  returning the pooled, pre-head embedding; `forward` is `head(forward_features(x))`.

### Tests

- +13 tests (`test_retrieval.py`) with hand-computed metric values; 348 passing.

---

## [1.14.2] — 2026-08-24

Release-hardening pass: static audit + runtime stress testing.

### Bug Fixes

- **`DistillationTrainer` parity (HIGH)** — its overridden `_run_epoch` silently
  diverged from the base `Trainer`: it ignored `grad_accum_steps`, never called
  the DDP loss-reduction / prediction-gather (so under DDP every rank computed
  metrics on its own shard, corrupting checkpoint/early-stop selection), and
  omitted the float32 cast before numpy metrics (feeding fp16 arrays to sklearn
  under AMP). Rewritten to mirror the base loop exactly while injecting teacher
  logits.
- **ArcFace / Sub-center ArcFace fp16 NaN (MEDIUM)** — `sqrt(1 - cosine²)` could
  round negative under autocast/fp16 (the `1e-7` clamp bound isn't representable
  in float16), producing NaN loss. Now `sqrt((1 - cosine²).clamp_min(1e-7))`.
- **`EnsemblePredictor` multilabel squeeze (LOW)** — bare `.squeeze()` in the
  multilabel `predict`/`predict_proba` paths collapsed the batch dim for
  batch-size-1 or single-label outputs; now `.squeeze(0)`.

### Tests

- +12 tests: `test_stress.py` (grad-accum×EMA, single-sample/single-class
  batches, extreme-value losses/metrics, ArcFace fp16 stability, soup with BN
  buffers) and a `DistillationTrainer` grad-accum regression. 335 passing.

---

## [1.14.1] — 2026-08-20

### Critical DDP Fix

- **Fixed multi-GPU hang/deadlock** (`use_ddp=True`). The DDP init never called
  `torch.cuda.set_device(local_rank)`, so the object collectives added in 1.12.0
  (`all_gather_object`) allocated their transport tensors on `cuda:0` for *every*
  rank — NCCL then deadlocked during communicator setup and timed out after 10
  minutes. Discovered on a real 2×T4 benchmark. Now `set_device` is called
  before `init_process_group`.
- **Automatic `DistributedSampler`** — under DDP the `Trainer` now rebuilds the
  train/val loaders with a `DistributedSampler` (train shuffled, val not),
  preserving batch size / workers / pinning / `drop_last` / collate. Previously
  both ranks iterated the *entire* dataset, which gave no throughput speedup and
  caused `_gather_concat` to double-count the validation set. `set_epoch` is now
  called each epoch for correct reshuffling.

### Tests

- +2 tests: `test_ddp_loader.py` (single-process gloo) — 324 passing.

---

## [1.14.0] — 2026-08-19

### New Features

- **Post-Training Quantization** (`fujicv.export.quantization`) for edge/CPU
  deployment:
  - `quantize_dynamic(model)` — INT8 weight quantization, no calibration needed
    (best for Linear/transformer-heavy models). Deep-copies the model so the
    original is untouched.
  - `quantize_static(model, calibration_data, backend=...)` — full static INT8
    via FX graph mode with calibration (best for CNNs); raises a clear,
    actionable error if the model isn't FX-traceable.
  - `measure_model_size(model)` — report the on-disk MB for before/after
    comparisons.
- **Expanded `ModelBuilder` head blocks** — `custom_layers` now supports
  `Linear`, `Dropout`, `LayerNorm`, `BatchNorm1d`, and `Activation`
  (relu/gelu/silu/tanh/leakyrelu) in addition to `LinearBNDropout`.
- **Pluggable feature pooling** — `ModelBuilder(..., pooling=...)` selects
  `'avg'` (default), `'max'`, `'gem'` (learnable Generalised-Mean, ideal for
  retrieval), or `'attention'` (learned attention pool). Works for both CNN
  spatial maps and transformer token sequences.

### Tests

- +14 tests: `test_quantization.py`, `test_builder_head_blocks.py`
  (322 passing, 8 skipped).

---

## [1.13.0] — 2026-08-18

### Extensibility

- **Hugging Face backbones** — `ModelBuilder(backbone_source="hf", ...)` (also
  `build_backbone(source="hf")`) loads any `transformers` vision model by repo
  id (ViT, Swin, DeiT, ConvNeXt, …). A wrapper adapts the HF output
  (`last_hidden_state`, `(B, N, C)` tokens or `(B, C, H, W)` maps) to FujiCV's
  pooling, and the feature width is auto-probed. Install with
  `pip install "fujicv[hf-models]"`.
- **Generic logger interface** — new `BaseLogger` ABC; `Trainer(..., loggers=[...])`
  drives any implementation. Added `MLflowLogger` (`pip install "fujicv[mlflow]"`).
  The existing `WandbLogger`/`TensorBoardLogger` already follow the same
  `log_epoch`/`finish`/`active` contract.
- **`torch.compile` support** — `Trainer(..., compile_model=True, compile_mode=...)`
  wraps the model with `torch.compile` for graph optimisation on PyTorch 2.x.
  Gracefully falls back to eager (with a warning) where compilation is
  unavailable or unsupported (e.g. Windows). `_model_core` now robustly unwraps
  any nesting of `torch.compile` + DDP for clean checkpoints.

### Tests

- +12 tests: `test_hf_backbone.py`, `test_loggers.py`, `test_compile.py`
  (308 passing, 8 skipped).

---

## [1.12.0] — 2026-08-18

### DDP Correctness (High priority)

- **Rank-guarded checkpointing** — `best.pt`, `last.pt`, and history files are
  now written only on rank 0 (`_is_main_process`), eliminating the multi-process
  race that could corrupt checkpoints under `torchrun`.
- **Distributed-correct metrics** — `_run_epoch` now all-gathers predictions and
  targets (`all_gather_object`) and all-reduces the loss across ranks, so
  `val_accuracy`/`val_loss` reflect the *entire* validation set rather than one
  GPU's shard. This makes `EarlyStopping` and `ReduceLROnPlateau` behave
  identically and correctly on every rank.

### Inference Pipeline

- **`Predictor.predict_batch` preserves identifiers** — resolves IDs from
  loader-yielded `(images, ids)` batches (`yields_ids=True`), an explicit `ids`
  list, or a running `sample_<idx>` fallback (was: dummy `batchN_sampleI`).
  Ready for a Kaggle `submission.csv`.
- **Vectorized decoding** — `_decode_scores` computes softmax/argmax (or
  sigmoid, or regression outputs) for the whole batch at once instead of a
  per-item Python loop.
- **Built-in TTA** — `predict(..., use_tta=True)` and
  `predict_batch(..., use_tta=True)` average the original image with its
  horizontal flip (probability space for classification/multilabel, output
  space for regression).

### Resilience

- **Incremental history** — `history.csv` **and** a new `history.json` are
  written after *every* epoch, so a preemption/crash at epoch 90/100 no longer
  loses the metric history. Added `History.to_json`.

### Tests

- +11 tests: `test_predictor_batch.py`, `test_trainer_resilience.py`
  (296 total, all green).

---

## [1.11.0] — 2026-08-11

### New Features

- **Metric-learning margin heads** (`fujicv.models.metric_heads`) — the
  workhorse of fine-grained classification and image-retrieval competitions:
  - `ArcMarginProduct` (ArcFace, Deng et al. 2019) — additive angular margin,
    with numerically safe hard-margin default and `easy_margin` option.
  - `AddMarginProduct` / `CosMarginProduct` (CosFace, Wang et al. 2018) —
    additive cosine margin.
  - `SubCenterArcMarginProduct` (Sub-center ArcFace, Deng et al. 2020) — `K`
    sub-centers per class for noisy-label robustness.
  - Each head takes `(features, labels)`; with `labels=None` it returns plain
    scaled cosine similarities for inference/retrieval.
- **Model soups** (`fujicv.training.model_soup`, Wortsman et al. 2022) —
  average the weights of several fine-tuned models for ensemble-level accuracy
  at single-model inference cost:
  - `uniform_soup(model, states)` — plain weight average.
  - `greedy_soup(model, states, eval_fn)` — keep only ingredients that improve
    the running soup on a validation callback.

### Tests

- +16 tests: `test_metric_heads.py`, `test_model_soup.py` (285 total, green).

---

## [1.10.0] — 2026-08-11

### New Features

- **Gradient accumulation** — `Trainer(..., grad_accum_steps=N)` splits the
  effective batch across `N` micro-batches, letting you train with large
  effective batch sizes on limited GPU memory. Loss is scaled by `1/N`, the
  optimizer steps every `N` micro-batches (and always on the final partial
  window), gradient clipping is applied before each step, and EMA updates only
  on real optimizer steps. Verified numerically equivalent to a single
  big-batch step.
- **TensorBoard logging** — new `TensorBoardLogger` (offline, no API key). Pass
  `Trainer(..., tb_logger=TensorBoardLogger(log_dir="runs/exp1"))`. Metrics are
  grouped into shared charts (`loss/train` + `loss/val`). Install with
  `pip install "fujicv[tensorboard]"`.
- **Layer freezing / gradual unfreezing** — `fujicv.training.freezing` adds
  `freeze`, `unfreeze`, `freeze_backbone`, `unfreeze_backbone`, `freeze_bn_stats`,
  `count_trainable_parameters`, and a `GradualUnfreezing` helper that unfreezes
  the backbone top-down over epochs for staged fine-tuning.

### Tests

- +14 tests: `test_grad_accum.py`, `test_freezing.py`, `test_tensorboard_logger.py`
  (269 total, all green).

---

## [1.9.0] — 2026-07-30

### Bug Fixes & Robustness

- **`engine/trainer.py`** — Replaced silent `nn.DataParallel` auto-wrap with proper
  `DistributedDataParallel` (DDP) support.  Pass `use_ddp=True` to `Trainer` when
  launching via `torchrun`.  On single-GPU or CPU, a clear `UserWarning` is emitted
  if multiple GPUs are detected but `use_ddp=False`.  Added `_model_core` property
  that unwraps both DDP and legacy DataParallel uniformly in checkpoint saving.
- **`models/builder.py`** — Fixed fragile ViT feature extraction: the 3-D branch
  `(B, N, C)` now uses `feats.mean(dim=1)` instead of `feats[:, 0]`.  Mean pooling
  over the token sequence is architecture-agnostic and avoids hard-coding CLS position
  (which varies across ViT variants and is not guaranteed for CAIT, XCiT, etc.).

### New: Integration Tests

- `tests/test_integration.py` — 6 end-to-end tests covering:
  - `best.pt`, `last.pt`, `history.csv` artifact creation after 2-epoch run.
  - Checkpoint loadability and key validation.
  - `history.csv` column correctness.
  - `resume_from` continuity (cumulative history).
  - `early_stopping_patience` halt verification (frozen model, patience=1).
  - Single-GPU / CPU `use_ddp=False` safety.

### New: Community Files

- `CONTRIBUTING.md` — dev environment setup, test/lint commands, branching model,
  feature-addition checklist, and security policy.
- `.pre-commit-config.yaml` — enforces `ruff` linting + formatting, standard file
  hygiene hooks, and `detect-secrets` on every commit.

---

## [1.8.0] — 2026-07-28

### New Features

**LR Finder** (`fujicv.training.LRFinder`)
- Exponential range-test (Smith 2015 / fast.ai style) to identify optimal learning rate before training.
- `range_test(loader, start_lr, end_lr, num_iter)` — runs the sweep, then auto-restores model and optimizer state.
- `suggestion()` — returns the LR at the steepest loss descent.
- `plot()` — smoothed loss-vs-LR curve with suggested LR marked.

**Stochastic Weight Averaging** (`fujicv.training.SWA`)
- Wraps `torch.optim.swa_utils.AveragedModel` with a clean API: `update()` / `finalize(loader)`.
- `get_scheduler(optimizer)` — returns a `SWALR` cosine annealing scheduler.
- `state_dict()` / `load_state_dict()` for checkpointing.

**TorchScript export** (`fujicv.export.export_torchscript`)
- `export_torchscript(model, path, example_inputs, method='trace')` — trace or script export.
- `verify_torchscript(scripted, inputs, original_model)` — numerical output check.
- `load_torchscript(path)` — reload saved `ScriptModule`.

**ONNX quantization** (`fujicv.export.quantize_onnx`)
- Post-training dynamic INT8 quantization via `onnxruntime.quantization`.
- `quantize_onnx(onnx_path, output_path, quantize_type='dynamic', per_channel=False)`.
- Reduces model size and speeds up CPU inference with no calibration data needed.

**Optuna HPO enhancements** (`fujicv.hpo.run_hpo`)
- `run_hpo` now accepts `pruner` (`'median'`, `'hyperband'`, `'percentile'`, `'successive_halving'`) and `pruner_kwargs`.
- `plot_optimization_history(study)` — scatter + best-so-far line chart.
- `plot_param_importances(study)` — horizontal bar chart of hyperparameter importances.

---

## [1.7.0] — 2026-07-22

### Bug Fixes

- **`training/sam.py`** — ASAM adaptive perturbation used `w²` instead of `|w|`.  The ASAM paper (Kwon et al., 2021) defines the perturbation as `|w| * grad / ‖|w| * grad‖`.  Fixed `torch.pow(p, 2)` → `torch.abs(p)`.
- **`data/mixup.py`** — `_rand_bbox` used `random.randint(0, W)` which is inclusive on both ends; `cx` could equal `W`, producing a degenerate zero-area patch at the right/bottom edge.  Fixed to `random.randint(0, W - 1)` and `random.randint(0, H - 1)`.
- **`engine/distillation_trainer.py`** — `_run_epoch` returned bare keys `"loss"` and `"<metric>"` instead of the `"train_loss"` / `"val_loss"` convention used by the base `Trainer`.  This silently broke `CheckpointCallback` (monitor key never matched) and `EarlyStopping`.  Fixed by prepending `"train_"` / `"val_"` prefix.
- **`inference/ensemble.py`** — `predict()` with `merge='vote'` called `_forward_all()` twice (once inside `_merge()` and again explicitly), doubling inference cost.  Fixed by caching the single `_forward_all` result and passing it to `_merge`.
- **`training/kfold.py`** — `trainer.output_dir` was overridden per-fold but `trainer._ckpt.output_dir` (set at `Trainer.__init__` time) still pointed to the original directory.  Best checkpoints were written to the wrong folder.  Fixed by also updating `trainer._ckpt.output_dir`.

### New Features

**EMA integrated into Trainer**
- `Trainer` now accepts `use_ema=True`, `ema_decay`, and `ema_warmup_steps` kwargs
- When enabled, `ModelEMA.update()` is called automatically after every optimizer step
- Validation runs use EMA shadow weights (via `average_parameters` context manager)
- EMA state is saved to `best.pt` under the `"ema_state_dict"` key for reproducibility
- 5 unit tests

**WeightedRandomSampler factory (class-imbalance-aware)**
- New `fujicv.data.sampler` module with `make_weighted_sampler` and `class_weights_from_labels`
- `make_weighted_sampler(labels)` — inverse-frequency weights, drop-in `sampler=` for DataLoader
- `class_weights_from_labels(labels)` — normalized tensor for `CrossEntropyLoss(weight=...)`
- Exported from `fujicv.data`; 10 unit tests

---

## [1.6.0] — 2026-07-22

### New Features

**Grad-CAM / Grad-CAM++ (Explainability)**
- New `fujicv.eval.gradcam` module with `GradCAM`, `GradCAMPlusPlus`, `overlay_heatmap`
- `GradCAM(model, target_layer)` — hooks any conv layer; `.generate(image)` → (H, W) heatmap in [0, 1]
- `GradCAMPlusPlus` — improved localization using second-order gradients (Chattopadhyay et al., 2018)
- `overlay_heatmap(image, heatmap, alpha)` — blends JET colormap onto original image (requires opencv)
- Accepts raw numpy images or pre-processed tensors; auto-resizes output to input dimensions
- Exported from `fujicv.eval`; 6 unit tests

**Ensemble Prediction**
- New `fujicv.inference.ensemble.EnsemblePredictor` — combine any number of models
- Four merge strategies: `mean`, `vote` (majority), `max` (element-wise), `weighted_mean`
- Supports classification, regression, and multilabel tasks
- `predict(image)`, `predict_proba(image)`, `predict_batch(loader, return_targets=True)`
- Exported from `fujicv.inference`; 11 unit tests

**SAM Optimizer (Sharpness-Aware Minimization)**
- New `fujicv.training.sam.SAM` — wraps any base optimizer (SGD, AdamW, …)
- Explicit two-step API: `first_step()` → perturb, `second_step()` → restore + update
- Adaptive SAM (ASAM) mode via `adaptive=True` — per-parameter magnitude normalization
- Compatible with gradient clipping and AMP
- Exported from `fujicv.training`; 7 unit tests

**Confusion Matrix + Per-class Metrics**
- New `fujicv.eval.confusion` module with `plot_confusion_matrix` and `per_class_metrics`
- `plot_confusion_matrix` — normalized or raw counts, custom class names, save to file
- `per_class_metrics` — returns DataFrame with precision, recall, F1, support per class
- Exported from `fujicv.eval`; 8 unit tests

**RandAugment**
- New `fujicv.data.autoaugment` module with `RandAugment` and `RandAugmentTransform`
- `RandAugment(n, magnitude)` — randomly selects N ops from a 14-operation bank (Cubuk et al., 2019)
- Ops include rotate, shear, translate, solarize, posterize, sharpness, color, brightness, contrast, equalize
- `RandAugmentTransform` — albumentations-compatible `__call__(image=...) → dict` wrapper
- `magnitude_std` for stochastic magnitude sampling; `prob` to gate per-batch
- Exported from `fujicv.data`; 10 unit tests

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
