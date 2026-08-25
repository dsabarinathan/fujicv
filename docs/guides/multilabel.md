# Multi-Label Classification

Assign any number of labels per image (attributes, tags, findings).

```python
import torch
from fujicv.engine.trainer import Trainer
from fujicv.losses.multilabel import FocalBCELoss
from fujicv.metrics.multilabel import mAP, PerLabelAUROC
from fujicv.models.builder import ModelBuilder

model = ModelBuilder(
    "resnet50", task="multilabel", num_outputs=20, image_size=224,
).build()

trainer = Trainer(
    model=model, train_loader=train_loader, val_loader=val_loader,
    loss_fn=FocalBCELoss(gamma=2.0),
    metrics={"mAP": mAP()},
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
    epochs=15, task="multilabel", output_dir="runs/multilabel",
    monitor_metric="val_mAP",
)
history = trainer.train()
```

## Label format

Targets are multi-hot float vectors of length `num_outputs`. With
`CSVImageDataset`, the label column may be a space/comma-separated string of the
active label indices, or a list — it is converted to a `(num_labels,)` float
tensor automatically.

## Loss functions

| Loss | Notes |
|---|---|
| `BCEWithLogitsLoss` | Standard baseline |
| `FocalBCELoss` | Down-weights easy negatives (common with many labels) |
| `AsymmetricLoss` | State-of-the-art for long-tailed multi-label (Ben-Baruch 2021) |

```python
from fujicv.losses.multilabel import AsymmetricLoss

loss_fn = AsymmetricLoss(gamma_pos=0, gamma_neg=4, clip=0.05)
```

## Metrics

`mAP` (mean average precision) is the standard multi-label metric.
`PerLabelAUROC` and `HammingLoss` are also available.

## Thresholding at inference

The `Predictor` applies a 0.5 sigmoid threshold by default and returns the list
of predicted labels plus a mean confidence.
