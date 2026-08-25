# Classification

Train a single-label image classifier end to end.

```python
import torch
from torch.utils.data import DataLoader

from fujicv.data.datasets import get_default_dataset
from fujicv.data.transforms import get_train_transforms, get_val_transforms
from fujicv.engine.trainer import Trainer
from fujicv.losses.classification import CrossEntropyLoss
from fujicv.metrics.classification import Accuracy
from fujicv.models.builder import ModelBuilder

# Data (CIFAR-10 downloads automatically)
train_ds, val_ds, class_to_idx = get_default_dataset(
    "cifar10", root="data",
    train_transform=get_train_transforms(32, level="medium"),
    val_transform=get_val_transforms(32),
)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False, num_workers=2)

# Model
model = ModelBuilder("resnet18", task="classification",
                     num_outputs=10, image_size=32).build()

# Train
trainer = Trainer(
    model=model, train_loader=train_loader, val_loader=val_loader,
    loss_fn=CrossEntropyLoss(),
    metrics={"accuracy": Accuracy()},
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4),
    epochs=10, task="classification", output_dir="runs/cifar10",
    monitor_metric="val_accuracy",
)
history = trainer.train()   # → best.pt  last.pt  history.csv  history.json
```

## Loss functions

| Loss | When to use |
|---|---|
| `CrossEntropyLoss` | Standard multiclass baseline |
| `FocalLoss` | Class imbalance / hard examples |
| `LabelSmoothingCE` | Regularization, over-confident models |

```python
from fujicv.losses.classification import FocalLoss, LabelSmoothingCE

loss_fn = FocalLoss(gamma=2.0)
loss_fn = LabelSmoothingCE(smoothing=0.1)
```

## Metrics

```python
from fujicv.metrics.classification import Accuracy, F1Score, AUROC

metrics = {"accuracy": Accuracy(), "f1": F1Score(average="macro")}
```

## Imbalanced data

Use a weighted sampler so rare classes are seen more often:

```python
from fujicv.data.sampler import make_weighted_sampler, class_weights_from_labels

sampler = make_weighted_sampler(train_labels)
train_loader = DataLoader(train_ds, batch_size=128, sampler=sampler)
```

## Fine-grained / retrieval

For fine-grained classification, swap the head for an ArcFace margin head and
use GeM pooling — see the [Kaggle Recipes](kaggle.md) guide.
