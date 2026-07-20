"""Dataset classes and split-building utilities."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_VALID_TASKS = {"classification", "regression", "multilabel", "multiclass"}


class CSVImageDataset(Dataset):
    """Dataset that reads images specified by a pandas DataFrame.

    The dataset pre-validates file existence at construction time, logging a
    warning for each missing file and excluding it from the dataset.

    Args:
        df: DataFrame with at least *image_col* and *label_col* columns.
        image_dir: Base directory; image paths in the CSV are resolved relative
            to this directory.  If an absolute path is already present in the
            CSV it will be used as-is.
        image_col: Column containing image filenames or paths.
        label_col: Column containing labels.  For multilabel tasks this should
            be a column of space/comma-separated strings or a list column.
        task: One of ``'classification'``, ``'regression'``, ``'multilabel'``,
            ``'multiclass'``.
        transform: An albumentations ``Compose`` pipeline (or any callable that
            accepts ``image=np.ndarray`` and returns ``{"image": tensor}``).
        class_to_idx: Optional mapping from class name → integer index.  If
            omitted for classification tasks it is built from the unique values
            in *label_col*.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: Union[str, Path],
        image_col: str,
        label_col: str,
        task: str,
        transform: Optional[Callable] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        if task not in _VALID_TASKS:
            raise ValueError(f"task must be one of {sorted(_VALID_TASKS)}, got {task!r}")

        self.image_dir = Path(image_dir)
        self.image_col = image_col
        self.label_col = label_col
        self.task = task
        self.transform = transform

        # Pre-validate file existence
        valid_rows = []
        for _, row in df.iterrows():
            img_path = self._resolve_path(row[image_col])
            if img_path.exists():
                valid_rows.append(row)
            else:
                logger.warning("Image not found, skipping: %s", img_path)

        if not valid_rows:
            warnings.warn("No valid images found in the dataset.", stacklevel=2)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        # Build class_to_idx for classification tasks
        if task in ("classification", "multiclass"):
            if class_to_idx is not None:
                self.class_to_idx = class_to_idx
            else:
                unique_labels = sorted(self.df[label_col].astype(str).unique())
                self.class_to_idx: Dict[str, int] = {cls: i for i, cls in enumerate(unique_labels)}
        else:
            self.class_to_idx = class_to_idx or {}

    def _resolve_path(self, filename: str) -> Path:
        p = Path(filename)
        if p.is_absolute():
            return p
        return self.image_dir / p

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row[self.image_col])

        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        raw_label = row[self.label_col]
        label = self._encode_label(raw_label)
        return image, label

    def _encode_label(self, raw: Any) -> Any:
        if self.task in ("classification", "multiclass"):
            return torch.tensor(self.class_to_idx[str(raw)], dtype=torch.long)
        elif self.task == "regression":
            return torch.tensor(float(raw), dtype=torch.float32)
        elif self.task == "multilabel":
            # Support comma/space-separated strings or lists
            if isinstance(raw, str):
                raw = [x.strip() for x in raw.replace(",", " ").split()]
            # Expect a list of 0/1 or float values
            return torch.tensor([float(v) for v in raw], dtype=torch.float32)
        else:
            raise ValueError(f"Unknown task: {self.task}")


def build_splits(
    dataset_cfg: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build train / val / test DataFrames from a dataset configuration dict.

    If the CSV contains a ``split_col`` column its values (``'train'``,
    ``'val'`` / ``'valid'``, ``'test'``) are used directly.  Otherwise a
    stratified split (classification) or random split (regression / multilabel)
    is performed.

    The resulting split assignment is saved to
    ``<output_dir>/split_assignment.csv`` when *output_dir* is provided.

    Expected config keys:
    * ``csv_path`` — path to the CSV file.
    * ``label_col`` — label column name.
    * ``task`` — task type.
    * ``split_col`` — (optional) pre-existing split column name.
    * ``val_fraction`` — fraction for validation (default 0.15).
    * ``test_fraction`` — fraction for test (default 0.15).
    * ``random_seed`` — (optional) integer seed.
    * ``output_dir`` — (optional) directory to write split_assignment.csv.

    Returns:
        Tuple of ``(train_df, val_df, test_df)`` DataFrames.
    """
    csv_path = Path(dataset_cfg["csv_path"])
    df = pd.read_csv(csv_path)

    split_col = dataset_cfg.get("split_col")
    label_col = dataset_cfg["label_col"]
    task = dataset_cfg["task"]
    val_frac = float(dataset_cfg.get("val_fraction", 0.15))
    test_frac = float(dataset_cfg.get("test_fraction", 0.15))
    seed = int(dataset_cfg.get("random_seed", 42))
    output_dir = dataset_cfg.get("output_dir")

    if split_col and split_col in df.columns:
        train_df = df[df[split_col].isin(["train"])].copy()
        val_df = df[df[split_col].isin(["val", "valid"])].copy()
        test_df = df[df[split_col].isin(["test"])].copy()
    else:
        if task in ("classification", "multiclass"):
            try:
                from sklearn.model_selection import train_test_split

                idx_trainval, idx_test = train_test_split(
                    np.arange(len(df)),
                    test_size=test_frac,
                    stratify=df[label_col],
                    random_state=seed,
                )
                val_relative = val_frac / (1.0 - test_frac)
                idx_train, idx_val = train_test_split(
                    idx_trainval,
                    test_size=val_relative,
                    stratify=df[label_col].iloc[idx_trainval],
                    random_state=seed,
                )
            except Exception:
                # Fall back to random split if stratification fails
                logger.warning("Stratified split failed; falling back to random split.")
                idx_all = np.random.default_rng(seed).permutation(len(df))
                n_test = int(len(df) * test_frac)
                n_val = int(len(df) * val_frac)
                idx_test = idx_all[:n_test]
                idx_val = idx_all[n_test : n_test + n_val]
                idx_train = idx_all[n_test + n_val :]
        else:
            rng = np.random.default_rng(seed)
            idx_all = rng.permutation(len(df))
            n_test = int(len(df) * test_frac)
            n_val = int(len(df) * val_frac)
            idx_test = idx_all[:n_test]
            idx_val = idx_all[n_test : n_test + n_val]
            idx_train = idx_all[n_test + n_val :]

        train_df = df.iloc[idx_train].copy()
        val_df = df.iloc[idx_val].copy()
        test_df = df.iloc[idx_test].copy()

        # Save split assignment
        df = df.copy()
        df["_split"] = "train"
        df.loc[df.index[idx_val], "_split"] = "val"
        df.loc[df.index[idx_test], "_split"] = "test"
        if output_dir:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            df.to_csv(out / "split_assignment.csv", index=False)

    logger.info(
        "Split sizes — train: %d  val: %d  test: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Built-in default datasets
# ---------------------------------------------------------------------------

class CIFAR10Dataset(Dataset):
    """CIFAR-10 wrapped as a FujiCV-compatible dataset (auto-downloads).

    Args:
        root: Directory to store/cache the dataset.
        split: ``'train'`` or ``'val'`` (uses the CIFAR-10 test split for val).
        transform: Albumentations ``Compose`` transform applied to each image.
    """

    CLASSES = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

    def __init__(
        self,
        root: str | Path = "data/cifar10",
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        from torchvision.datasets import CIFAR10 as _CIFAR10

        train = split == "train"
        self._ds = _CIFAR10(root=str(root), train=train, download=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        img, label = self._ds[idx]
        img = np.array(img)  # PIL → numpy for albumentations
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, label


class MNISTDataset(Dataset):
    """MNIST digits 0-9 wrapped as a FujiCV-compatible dataset (auto-downloads).

    Only ~11 MB. Images are converted to 3-channel RGB so any backbone works.

    Args:
        root: Directory to store/cache the dataset.
        split: ``'train'`` or ``'val'`` (uses the MNIST test split for val).
        transform: Albumentations ``Compose`` transform applied to each image.
    """

    CLASSES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

    def __init__(
        self,
        root: str | Path = "data/mnist",
        split: str = "train",
        transform: Optional[Callable] = None,
    ) -> None:
        from torchvision.datasets import MNIST as _MNIST

        train = split == "train"
        self._ds = _MNIST(root=str(root), train=train, download=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self._ds)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        img, label = self._ds[idx]
        # MNIST is grayscale — convert to RGB so backbones expecting 3 channels work
        img = np.array(img.convert("RGB"))
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, label


def get_default_dataset(
    name: str = "mnist",
    root: str | Path = "data",
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
) -> Tuple[Dataset, Dataset, Dict[str, int]]:
    """Return a built-in default dataset ready for training.

    Args:
        name: Dataset name — ``'mnist'`` (default, 11 MB) or ``'cifar10'`` (170 MB).
        root: Root directory for download/cache.
        train_transform: Albumentations transform for the training split.
        val_transform: Albumentations transform for the validation split.

    Returns:
        ``(train_dataset, val_dataset, class_to_idx)``
    """
    if name.lower() == "mnist":
        train_ds = MNISTDataset(root=Path(root) / "mnist", split="train", transform=train_transform)
        val_ds = MNISTDataset(root=Path(root) / "mnist", split="val", transform=val_transform)
        return train_ds, val_ds, MNISTDataset.class_to_idx
    if name.lower() == "cifar10":
        train_ds = CIFAR10Dataset(root=Path(root) / "cifar10", split="train", transform=train_transform)
        val_ds = CIFAR10Dataset(root=Path(root) / "cifar10", split="val", transform=val_transform)
        return train_ds, val_ds, CIFAR10Dataset.class_to_idx
    raise ValueError(f"Unknown default dataset '{name}'. Available: ['mnist', 'cifar10']")
