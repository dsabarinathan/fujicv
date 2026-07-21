"""K-Fold cross-validation training wrapper."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class KFoldTrainer:
    """Run stratified k-fold cross-validation and return per-fold metrics.

    Trains a fresh model copy for each fold, saves per-fold checkpoints, and
    aggregates metrics (mean ± std) across all folds.  Out-of-fold (OOF)
    predictions are collected so the caller can build a fold ensemble or
    compute a single evaluation score over the full dataset.

    Args:
        model_factory: Zero-argument callable that returns a fresh ``nn.Module``
            each time it is called (one per fold).
        train_df: Full training DataFrame (with *image_col* and *label_col*).
        dataset_factory: Callable ``(df, transform) -> Dataset``.
        train_transform: Albumentations pipeline for training folds.
        val_transform: Albumentations pipeline for validation folds.
        trainer_factory: Callable ``(model, train_loader, val_loader) -> Trainer``.
            Responsible for building the optimizer, loss, metrics, and Trainer.
        n_splits: Number of folds (default 5).
        stratify_col: Column to stratify on (default ``None`` — uses plain
            KFold). Pass the label column name for classification.
        output_dir: Base directory; per-fold checkpoints go into
            ``<output_dir>/fold_0/``, ``<output_dir>/fold_1/``, …
        seed: Random seed for reproducibility (default 42).
        dataloader_kwargs: Extra kwargs forwarded to both DataLoaders
            (e.g. ``num_workers=4``).

    Returns (from :meth:`run`):
        A dict with keys:

        * ``fold_histories`` — list of :class:`~fujicv.engine.trainer.History`
          objects, one per fold.
        * ``fold_metrics`` — list of per-fold best metric dicts.
        * ``summary`` — DataFrame with mean and std across folds.
        * ``oof_preds`` — numpy array of out-of-fold logits (full dataset).
        * ``oof_targets`` — numpy array of ground-truth targets.

    Example::

        from fujicv.training.kfold import KFoldTrainer
        from fujicv.models.builder import ModelBuilder
        from fujicv.engine.trainer import Trainer
        from fujicv.losses.classification import CrossEntropyLoss
        from fujicv.metrics.classification import Accuracy
        import torch.optim as optim

        def model_factory():
            return ModelBuilder("resnet18", task="classification",
                                num_outputs=3, pretrained=True).build()

        def dataset_factory(df, transform):
            return CSVImageDataset(df, img_dir, "filename", "label",
                                   "classification", transform)

        def trainer_factory(model, train_loader, val_loader):
            return Trainer(
                model=model, train_loader=train_loader, val_loader=val_loader,
                loss_fn=CrossEntropyLoss(),
                metrics={"accuracy": Accuracy()},
                optimizer=optim.AdamW(model.parameters(), lr=1e-3),
                epochs=10, task="classification",
                output_dir="runs/kfold",
            )

        kfold = KFoldTrainer(
            model_factory=model_factory,
            train_df=df,
            dataset_factory=dataset_factory,
            train_transform=get_train_transforms(224),
            val_transform=get_val_transforms(224),
            trainer_factory=trainer_factory,
            n_splits=5,
            stratify_col="label",
            output_dir="runs/kfold",
        )
        results = kfold.run()
        print(results["summary"])
    """

    def __init__(
        self,
        model_factory: Callable[[], nn.Module],
        train_df: pd.DataFrame,
        dataset_factory: Callable,
        train_transform: Any,
        val_transform: Any,
        trainer_factory: Callable,
        n_splits: int = 5,
        stratify_col: Optional[str] = None,
        output_dir: Union[str, Path] = "runs/kfold",
        seed: int = 42,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_factory    = model_factory
        self.train_df         = train_df.reset_index(drop=True)
        self.dataset_factory  = dataset_factory
        self.train_transform  = train_transform
        self.val_transform    = val_transform
        self.trainer_factory  = trainer_factory
        self.n_splits         = n_splits
        self.stratify_col     = stratify_col
        self.output_dir       = Path(output_dir)
        self.seed             = seed
        self.dl_kwargs        = dataloader_kwargs or {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute k-fold cross-validation.

        Returns:
            Dict with ``fold_histories``, ``fold_metrics``, ``summary``,
            ``oof_preds``, ``oof_targets``.
        """
        try:
            from sklearn.model_selection import KFold, StratifiedKFold
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for KFoldTrainer. "
                "Install with: pip install scikit-learn"
            ) from exc

        indices = np.arange(len(self.train_df))

        if self.stratify_col is not None:
            labels = self.train_df[self.stratify_col].values
            splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            splits = list(splitter.split(indices, labels))
        else:
            splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            splits = list(splitter.split(indices))

        fold_histories: List[Any] = []
        fold_metrics:   List[Dict[str, float]] = []
        # OOF storage — we don't know logit dim until first batch, fill lazily
        oof_preds:   Optional[np.ndarray] = None
        oof_targets: Optional[np.ndarray] = None

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            logger.info("=" * 60)
            logger.info("FOLD %d / %d", fold_idx + 1, self.n_splits)
            logger.info("  train=%d  val=%d", len(train_idx), len(val_idx))

            fold_dir = self.output_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            train_fold_df = self.train_df.iloc[train_idx].reset_index(drop=True)
            val_fold_df   = self.train_df.iloc[val_idx].reset_index(drop=True)

            train_ds = self.dataset_factory(train_fold_df, self.train_transform)
            val_ds   = self.dataset_factory(val_fold_df,   self.val_transform)

            train_loader = DataLoader(train_ds, shuffle=True,  **self.dl_kwargs)
            val_loader   = DataLoader(val_ds,   shuffle=False, **self.dl_kwargs)

            # Fresh model per fold
            model = self.model_factory()

            # Build trainer (caller sets output_dir, epochs, etc.)
            trainer = self.trainer_factory(model, train_loader, val_loader)
            # Override output_dir so each fold gets its own checkpoint
            trainer.output_dir = fold_dir
            trainer.output_dir.mkdir(parents=True, exist_ok=True)

            history = trainer.train()
            fold_histories.append(history)

            # Best metric values for this fold
            best: Dict[str, float] = {}
            for k, vals in history.metrics.items():
                if "loss" in k:
                    best[k] = float(min(vals)) if vals else float("nan")
                else:
                    best[k] = float(max(vals)) if vals else float("nan")
            fold_metrics.append(best)

            # Collect OOF predictions
            oof_logits, oof_tgts = self._collect_oof(trainer.model, val_loader, trainer.device)

            if oof_preds is None:
                # Initialise arrays now that we know the shapes
                total = len(self.train_df)
                logit_shape = (total,) + oof_logits.shape[1:] if oof_logits.ndim > 1 else (total,)
                oof_preds   = np.full(logit_shape, np.nan, dtype=np.float32)
                oof_targets = np.full(total, np.nan, dtype=np.float32)

            oof_preds[val_idx]   = oof_logits
            oof_targets[val_idx] = oof_tgts

            logger.info("Fold %d best metrics: %s", fold_idx + 1,
                        {k: f"{v:.4f}" for k, v in best.items()})

        summary = self._summarise(fold_metrics)
        logger.info("\n%s", summary.to_string())

        return {
            "fold_histories": fold_histories,
            "fold_metrics":   fold_metrics,
            "summary":        summary,
            "oof_preds":      oof_preds,
            "oof_targets":    oof_targets,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_oof(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        all_logits, all_targets = [], []
        with torch.no_grad():
            for batch in loader:
                images, targets = batch
                logits = model(images.to(device))
                all_logits.append(logits.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        return np.concatenate(all_logits), np.concatenate(all_targets)

    @staticmethod
    def _summarise(fold_metrics: List[Dict[str, float]]) -> pd.DataFrame:
        if not fold_metrics:
            return pd.DataFrame()
        df = pd.DataFrame(fold_metrics)
        summary = pd.DataFrame({
            "mean": df.mean(),
            "std":  df.std(),
            "min":  df.min(),
            "max":  df.max(),
        })
        return summary
