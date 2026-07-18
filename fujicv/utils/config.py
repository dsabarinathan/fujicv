"""YAML configuration loading and validation utilities."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file and return it as a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        Dictionary representation of the YAML contents.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if cfg is None:
        cfg = {}
    return cfg


def load_dataset_config(path: str | Path) -> Dict[str, Any]:
    """Load and validate the dataset block of a config file.

    Expected keys inside ``dataset``:
    * ``image_dir`` — directory containing images (must exist)
    * ``csv_path`` — CSV with at least ``image_col`` and ``label_col`` columns
    * ``image_col`` — column name for image file paths
    * ``label_col`` — column name for labels
    * ``task`` — one of classification / regression / multilabel / multiclass

    Args:
        path: Path to the YAML file.

    Returns:
        The ``dataset`` sub-dict from the config.

    Raises:
        KeyError: If required keys are missing.
        ValueError: If values are invalid.
    """
    cfg = load_config(path)

    if "dataset" not in cfg:
        raise KeyError(
            "Config is missing the required top-level 'dataset' block. "
            f"Keys found: {list(cfg.keys())}"
        )

    ds = cfg["dataset"]

    required_keys = ["csv_path", "image_col", "label_col", "task"]
    missing = [k for k in required_keys if k not in ds]
    if missing:
        raise KeyError(
            f"dataset block is missing required keys: {missing}. "
            f"Keys present: {list(ds.keys())}"
        )

    valid_tasks = {"classification", "regression", "multilabel", "multiclass"}
    if ds["task"] not in valid_tasks:
        raise ValueError(
            f"dataset.task must be one of {sorted(valid_tasks)}, got: {ds['task']!r}"
        )

    csv_path = Path(ds["csv_path"])
    if not csv_path.exists():
        raise FileNotFoundError(f"dataset.csv_path does not exist: {csv_path}")

    if "image_dir" in ds:
        image_dir = Path(ds["image_dir"])
        if not image_dir.exists():
            raise FileNotFoundError(f"dataset.image_dir does not exist: {image_dir}")

    return ds


def _get_git_hash() -> Optional[str]:
    """Return the current HEAD git commit hash, or None if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def save_resolved_config(
    config: Dict[str, Any],
    output_dir: str | Path,
    filename: str = "resolved_config.yaml",
) -> Path:
    """Save a resolved configuration dict alongside package version and git hash.

    Args:
        config: The resolved configuration dictionary.
        output_dir: Directory where the file will be written (created if absent).
        filename: Output filename (default: ``resolved_config.yaml``).

    Returns:
        Path to the written file.
    """
    import fujicv

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "fujicv_version": fujicv.__version__,
        "git_hash": _get_git_hash(),
    }

    payload = {"_meta": meta, **config}

    out_path = output_dir / filename
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.dump(payload, fh, default_flow_style=False, allow_unicode=True)

    return out_path
