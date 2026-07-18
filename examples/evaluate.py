"""Example evaluation script.

Usage::

    python examples/evaluate.py \\
        --config examples/configs/classification_example.yaml \\
        --checkpoint outputs/classification/best.pt \\
        --with-attention-maps
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from fujicv.data.dataloader import build_dataloaders
from fujicv.data.datasets import build_splits
from fujicv.eval.attention_map import generate_attention_grid
from fujicv.eval.curves import plot_pr_curve, plot_roc_curve
from fujicv.eval.plots import plot_loss_curves
from fujicv.eval.report import classification_report
from fujicv.eval.tsne import extract_embeddings, plot_tsne
from fujicv.models.builder import ModelBuilder
from fujicv.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main(config_path: str, checkpoint_path: str, with_attention_maps: bool) -> None:
    cfg = load_config(config_path)
    ds_cfg = cfg["dataset"]
    aug_cfg = cfg.get("augmentation", {})
    model_cfg = cfg["model"]
    train_cfg = cfg.get("training", {})

    output_dir = Path(train_cfg.get("output_dir", "outputs")) / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data ----
    train_df, val_df, test_df = build_splits(ds_cfg)
    _, val_loader, test_loader = build_dataloaders(train_df, val_df, test_df, ds_cfg, aug_cfg)
    eval_loader = test_loader if test_loader is not None else val_loader

    # ---- Model ----
    builder = ModelBuilder(
        backbone_name=model_cfg["backbone_name"],
        backbone_source=model_cfg.get("backbone_source", "timm"),
        pretrained=False,
        custom_layers=model_cfg.get("custom_layers"),
        task=model_cfg["task"],
        num_outputs=model_cfg["num_outputs"],
        head_kwargs=model_cfg.get("head_kwargs", {}),
        image_size=model_cfg.get("image_size", 224),
    )
    model = builder.build()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    class_to_idx: dict = ckpt.get("class_to_idx", {})
    class_names = [k for k, _ in sorted(class_to_idx.items(), key=lambda x: x[1])] or None
    model.to(device).eval()

    # ---- Collect predictions ----
    logger.info("Running inference on eval set…")
    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.numpy())

    logits_arr = np.concatenate(all_logits, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    task = model_cfg["task"]

    # ---- Classification reports ----
    if task in ("classification", "multiclass"):
        text, fig_cm = classification_report(labels_arr, logits_arr, class_names)
        logger.info("\n%s", text)
        fig_cm.savefig(output_dir / "confusion_matrix.png", dpi=150)

        probs = torch.softmax(torch.tensor(logits_arr), dim=-1).numpy()
        fig_roc = plot_roc_curve(labels_arr, probs, class_names)
        fig_roc.savefig(output_dir / "roc_curve.png", dpi=150)

        fig_pr = plot_pr_curve(labels_arr, probs, class_names)
        fig_pr.savefig(output_dir / "pr_curve.png", dpi=150)

        # t-SNE
        logger.info("Extracting embeddings for t-SNE…")
        embeddings, emb_labels = extract_embeddings(model, eval_loader, device)
        fig_tsne = plot_tsne(embeddings, emb_labels, class_names)
        fig_tsne.savefig(output_dir / "tsne.png", dpi=150)

        # Attention maps
        if with_attention_maps:
            arch_family = getattr(model, "arch_family", "cnn")
            logger.info("Generating attention maps (arch_family=%s)…", arch_family)
            fig_attn = generate_attention_grid(
                model, eval_loader, arch_family=arch_family,
                class_names=class_names, device=device,
            )
            fig_attn.savefig(output_dir / "attention_maps.png", dpi=150)

    logger.info("Evaluation outputs saved to %s", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FujiCV evaluation script")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument(
        "--with-attention-maps",
        action="store_true",
        default=False,
        help="Generate Grad-CAM / attention rollout visualisations",
    )
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.with_attention_maps)
