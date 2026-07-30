# Kaggle Recipes

Practical patterns for getting the most out of FujiCV in competition settings.

## Baseline in < 20 lines

```python
from fujicv.models.builder import ModelBuilder
from fujicv.losses import get_loss
from fujicv.metrics import get_metric
from fujicv.engine.trainer import Trainer
from fujicv.data import build_splits, build_dataloaders
from fujicv.utils import set_seed
import torch.optim as optim

set_seed(42)
train_df, val_df, _ = build_splits({"csv_path": "train.csv", "image_col": "path", "label_col": "label"})
train_loader, val_loader, _ = build_dataloaders(train_df, val_df, None,
    {"image_col": "path", "label_col": "label", "task": "classification"},
    {"preset": "heavy", "image_size": 224})

model = ModelBuilder("efficientnet_b3", task="classification", num_outputs=NUM_CLASSES).build()
trainer = Trainer(
    model=model, train_loader=train_loader, val_loader=val_loader,
    loss_fn=get_loss("LabelSmoothingCE", {"smoothing": 0.1}),
    metrics={"accuracy": get_metric("Accuracy")},
    optimizer=optim.AdamW(model.parameters(), lr=3e-4),
    epochs=20, task="classification", output_dir="outputs/", early_stopping_patience=3,
)
trainer.train()
```

## Stratified K-Fold OOF

```python
from fujicv.training import KFoldTrainer

kfold = KFoldTrainer(
    model_factory=lambda: ModelBuilder("resnet34", task="classification", num_outputs=5).build(),
    dataset_factory=lambda df: MyDataset(df),
    trainer_factory=lambda model, tl, vl, fold_dir: Trainer(model, tl, vl, ...),
    n_splits=5,
    stratify_col="label",
)
summary, oof_preds = kfold.fit(train_df, output_dir="kfold_outputs/")
```

## LR Finder before training

```python
from fujicv.training import LRFinder

finder = LRFinder(model, optimizer, criterion)
finder.range_test(train_loader, start_lr=1e-7, end_lr=10, num_iter=100)
best_lr = finder.suggestion()
finder.reset()   # restore weights before actual training
print(f"Suggested LR: {best_lr:.2e}")
```

## Ensemble predictions

```python
from fujicv.inference.ensemble import EnsemblePredictor
import torch

models = [load_model(f"fold_{i}/best.pt") for i in range(5)]
ensemble = EnsemblePredictor(models, merge="mean", task="classification")

# Single image
probs = ensemble.predict_proba(image_tensor)

# Full test loader
all_preds = ensemble.predict_batch(test_loader)
```

## ONNX quantization for faster submission

```python
from fujicv.export import to_onnx, quantize_onnx

to_onnx(model, "model.onnx")
quantize_onnx("model.onnx", "model_int8.onnx")   # 4× smaller, faster on CPU
```

## Optuna HPO

```python
from fujicv.hpo import run_hpo

result = run_hpo(objective, n_trials=50, pruner="hyperband")
```
