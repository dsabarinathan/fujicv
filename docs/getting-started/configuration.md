# Configuration

FujiCV is a plain Python API — there's no hidden global config. Everything is
set through constructor arguments. This page collects the knobs you'll reach for
most often.

## Model

```python
from fujicv.models.builder import ModelBuilder

model = ModelBuilder(
    backbone_name="resnet50",     # timm / torchvision name, or a HF repo id
    backbone_source="timm",       # "timm" | "torchvision" | "hf"
    pretrained=True,
    task="classification",        # "classification" | "regression" | "multilabel"
    num_outputs=10,
    image_size=224,
    pooling="avg",                # "avg" | "max" | "gem" | "attention"
    custom_layers=[               # optional pre-head blocks
        {"type": "Linear", "out_features": 512},
        {"type": "LayerNorm"},
        {"type": "Activation", "fn": "gelu"},
        {"type": "Dropout", "p": 0.2},
    ],
).build()
```

## Trainer

```python
from fujicv.engine.trainer import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    metrics={"accuracy": Accuracy()},
    optimizer=optimizer,
    scheduler=scheduler,          # optional
    epochs=10,
    task="classification",
    output_dir="runs/exp1",
    monitor_metric="val_accuracy",
    mixed_precision=True,         # AMP on CUDA
    grad_clip=1.0,
    grad_accum_steps=1,           # effective batch = batch_size × this
    early_stopping_patience=None,
    use_ema=False,
    use_ddp=False,                # True + torchrun for multi-GPU
    compile_model=False,          # torch.compile (PyTorch 2.x)
    loggers=[],                   # WandbLogger / TensorBoardLogger / MLflowLogger
)
```

Common `monitor_metric` values are `val_loss` (mode `min`, the default) or any
metric key you registered, e.g. `val_accuracy` (mode `max`, inferred from the
name).

## Reproducibility

```python
import fujicv
fujicv.set_seed(42)   # seeds Python, NumPy, and torch (+ CUDA)
```

## Optional dependencies

Install extras only for what you use:

```bash
pip install "fujicv[wandb]"        # W&B logging
pip install "fujicv[tensorboard]"  # TensorBoard logging
pip install "fujicv[mlflow]"       # MLflow logging
pip install "fujicv[hpo]"          # Optuna hyperparameter search
pip install "fujicv[onnx]"         # ONNX export + quantization
pip install "fujicv[hf-models]"    # Hugging Face transformers backbones
pip install "fujicv[retrieval]"    # FAISS retrieval index
```
