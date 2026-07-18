"""Example training script.

Usage::

    python examples/train.py --config examples/configs/classification_example.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the repo root is on the Python path when running from examples/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.optim as optim

from fujicv.data.dataloader import build_dataloaders
from fujicv.data.datasets import build_splits
from fujicv.engine.logger import WandbLogger
from fujicv.engine.trainer import Trainer
from fujicv.losses import get_loss
from fujicv.metrics import get_metric
from fujicv.models.builder import ModelBuilder
from fujicv.utils.config import load_config, save_resolved_config
from fujicv.utils.seed import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def build_optimizer(params, opt_cfg: dict):
    name = opt_cfg.get("name", "AdamW")
    lr = float(opt_cfg.get("lr", 1e-3))
    wd = float(opt_cfg.get("weight_decay", 1e-2))
    cls = getattr(optim, name)
    return cls(params, lr=lr, weight_decay=wd)


def build_scheduler(optimizer, sched_cfg: dict, epochs: int, steps_per_epoch: int = 1):
    name = sched_cfg.get("name", "CosineAnnealingLR")
    kwargs = {k: v for k, v in sched_cfg.items() if k != "name"}

    if name == "OneCycleLR":
        kwargs.setdefault("total_steps", epochs * steps_per_epoch)

    sched_cls = getattr(optim.lr_scheduler, name)
    return sched_cls(optimizer, **kwargs)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    seed = cfg.get("dataset", {}).get("random_seed", 42)
    set_seed(seed)

    ds_cfg = cfg["dataset"]
    aug_cfg = cfg.get("augmentation", {})
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    wandb_cfg = cfg.get("wandb", {})

    output_dir = Path(train_cfg.get("output_dir", "outputs"))
    save_resolved_config(cfg, output_dir)

    # ---- Data ----
    logger.info("Building dataset splits…")
    ds_cfg["output_dir"] = str(output_dir)
    train_df, val_df, test_df = build_splits(ds_cfg)
    train_loader, val_loader, _ = build_dataloaders(train_df, val_df, test_df, ds_cfg, aug_cfg)

    # ---- Model ----
    logger.info("Building model…")
    builder = ModelBuilder(
        backbone_name=model_cfg["backbone_name"],
        backbone_source=model_cfg.get("backbone_source", "timm"),
        pretrained=model_cfg.get("pretrained", True),
        custom_layers=model_cfg.get("custom_layers"),
        task=model_cfg["task"],
        num_outputs=model_cfg["num_outputs"],
        head_kwargs=model_cfg.get("head_kwargs", {}),
        image_size=model_cfg.get("image_size", 224),
    )
    model = builder.build()
    logger.info("Model built: %s", model_cfg["backbone_name"])

    # ---- Loss ----
    loss_cfg = train_cfg.get("loss", {"name": "CrossEntropyLoss"})
    loss_fn = get_loss(loss_cfg["name"], loss_cfg.get("kwargs", {}))

    # ---- Metrics ----
    task = model_cfg["task"]
    if task in ("classification", "multiclass"):
        metric_names = ["Accuracy", "F1", "AUROC"]
    elif task == "regression":
        metric_names = ["MAE", "RMSE", "R2Score"]
    else:
        metric_names = ["HammingLoss", "mAP"]
    metrics = {name: get_metric(name) for name in metric_names}

    # ---- Optimiser + Scheduler ----
    optimizer = build_optimizer(model.parameters(), train_cfg.get("optimizer", {}))
    epochs = int(train_cfg.get("epochs", 30))
    scheduler = build_scheduler(
        optimizer, train_cfg.get("scheduler", {}), epochs, len(train_loader)
    )

    # ---- W&B Logger ----
    wb_logger = WandbLogger(
        project=wandb_cfg.get("project", "fujicv"),
        entity=wandb_cfg.get("entity"),
        config=cfg,
        use_wandb=wandb_cfg.get("use_wandb", False),
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metrics=metrics,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        task=task,
        output_dir=output_dir,
        wandb_logger=wb_logger,
        mixed_precision=train_cfg.get("mixed_precision", True),
        grad_clip=train_cfg.get("grad_clip", 1.0),
        monitor_metric=train_cfg.get("monitor_metric", "val_loss"),
        early_stopping_patience=train_cfg.get("early_stopping_patience"),
    )

    history = trainer.train()
    logger.info("Training finished. Best checkpoint saved to %s/best.pt", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FujiCV training script")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()
    main(args.config)
