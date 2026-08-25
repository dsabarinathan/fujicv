# Regression

Predict continuous targets from images (age, count, score, …).

```python
import torch
from fujicv.engine.trainer import Trainer
from fujicv.losses.regression import HuberLoss
from fujicv.metrics.regression import MAE, RMSE
from fujicv.models.builder import ModelBuilder

model = ModelBuilder(
    "resnet18", task="regression", num_outputs=1, image_size=224,
).build()

trainer = Trainer(
    model=model, train_loader=train_loader, val_loader=val_loader,
    loss_fn=HuberLoss(delta=1.0),
    metrics={"mae": MAE(), "rmse": RMSE()},
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
    epochs=20, task="regression", output_dir="runs/reg",
    monitor_metric="val_mae",   # lower is better → mode="min"
)
history = trainer.train()
```

## Multi-output regression

Set `num_outputs > 1` to predict a vector per image:

```python
model = ModelBuilder("resnet18", task="regression", num_outputs=5).build()
```

The head outputs shape `(B, 5)`; the `Predictor` returns a list per image.

## Loss functions

| Loss | Notes |
|---|---|
| `MSELoss` | Standard L2 |
| `HuberLoss` | Robust to outliers (L2 near 0, L1 in the tails) |
| `LogCoshLoss` | Smooth, outlier-robust |
| `QuantileLoss` | Predict a specific quantile (e.g. median) |

## Ordinal regression

For ordered categories (ratings, severity grades), use the CORAL / CORN losses:

```python
from fujicv.losses.regression import CoralLoss, CornLoss

loss_fn = CornLoss(num_classes=5)   # head num_outputs = num_classes - 1
```
