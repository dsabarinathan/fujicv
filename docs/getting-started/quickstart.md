# Quickstart

This guide trains a ResNet50 on your own image dataset in under 15 lines.

## 1. Prepare your CSV

FujiCV reads image datasets from a CSV file with at least two columns:

```
path,label
/data/cat.jpg,0
/data/dog.jpg,1
```

## 2. Train

```python
from fujicv.models.builder import ModelBuilder
from fujicv.losses import get_loss
from fujicv.metrics import get_metric
from fujicv.engine.trainer import Trainer
from fujicv.data import build_splits, build_dataloaders
from fujicv.utils import set_seed
import torch.optim as optim

set_seed(42)

# Split CSV into train / val / test
train_df, val_df, test_df = build_splits({
    "csv_path": "data.csv",
    "image_col": "path",
    "label_col": "label",
    "val_size": 0.15,
    "test_size": 0.05,
})

train_loader, val_loader, test_loader = build_dataloaders(
    train_df, val_df, test_df,
    dataset_cfg={"image_col": "path", "label_col": "label", "task": "classification"},
    aug_cfg={"preset": "medium", "image_size": 224},
)

model = ModelBuilder(
    backbone_name="resnet50",
    task="classification",
    num_outputs=2,
    pretrained=True,
).build()

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=get_loss("LabelSmoothingCE", {"smoothing": 0.1}),
    metrics={"accuracy": get_metric("Accuracy"), "f1": get_metric("F1")},
    optimizer=optim.AdamW(model.parameters(), lr=3e-4),
    epochs=30,
    task="classification",
    output_dir="outputs/",
    early_stopping_patience=5,
)
history = trainer.train()
```

After training you will find:

- `outputs/best.pt` — best checkpoint (monitored on `val_loss` by default)
- `outputs/last.pt` — checkpoint from the final epoch
- `outputs/history.csv` — per-epoch metrics

## 3. Inference

```python
from fujicv.inference import Predictor

predictor = Predictor.from_checkpoint("outputs/best.pt", model=model)
label, confidence = predictor.predict("test_image.jpg")
print(f"Predicted class {label} with {confidence:.1%} confidence")
```

## Next steps

- [Classification guide](../guides/classification.md) — loss functions, metrics, augmentation presets
- [HPO with Optuna](../guides/hpo.md) — find the best hyperparameters automatically
- [Grad-CAM explainability](../guides/explainability.md) — visualize what the model sees
- [Export for production](../guides/export.md) — ONNX and TorchScript
