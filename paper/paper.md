---
title: 'FujiCV: A Lightweight, Batteries-Included Library for Image Classification, Regression, and Retrieval in PyTorch'
tags:
  - Python
  - PyTorch
  - computer vision
  - image classification
  - deep learning
  - transfer learning
authors:
  - name: Sabarinathan
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 30 August 2026
bibliography: paper.bib
---

# Summary

`FujiCV` is an open-source Python library that turns a trained image model into a
few lines of declarative code while retaining the full flexibility of PyTorch
[@paszke2019pytorch]. It targets the gap between writing a raw PyTorch training
loop — which requires manual mixed-precision, checkpointing, early stopping,
metric tracking, and distributed handling — and adopting a heavier framework.
`FujiCV` provides a single `ModelBuilder` → `Trainer` → evaluate/export pipeline
that spans **classification, regression, multi-label, and metric-learning /
retrieval** tasks on top of the `timm` [@rw2019timm], `torchvision`, and Hugging
Face `transformers` model zoos.

Beyond a training loop, `FujiCV` ships modern techniques out of the box —
Exponential Moving Average, Stochastic Weight Averaging [@izmailov2018swa],
Sharpness-Aware Minimization [@foret2021sam], Model Soups [@wortsman2022soups],
Mixup [@zhang2018mixup] and CutMix [@yun2019cutmix], layer-wise learning-rate
decay, gradient accumulation, `DistributedDataParallel` training, and
`torch.compile`. It also covers the parts of a project that usually require
separate tooling: explainability via Grad-CAM [@selvaraju2017gradcam],
hyperparameter search via Optuna [@akiba2019optuna], image retrieval with
ArcFace [@deng2019arcface] / CosFace [@wang2018cosface] heads and GeM pooling
[@radenovic2018gem], and deployment via ONNX [@onnx], TorchScript, and
post-training quantization.

# Statement of need

Practitioners, students, and researchers who fine-tune image models repeatedly
re-implement the same scaffolding: data loaders, augmentation, an AMP training
loop, checkpoint selection, metrics, and an export path. Doing this correctly is
error-prone — subtle bugs in distributed metric aggregation, checkpoint race
conditions, or mismatched normalization statistics silently degrade results.
Existing options force a trade-off: raw PyTorch offers full control at the cost
of substantial boilerplate; larger frameworks such as PyTorch Lightning
[@falcon2019lightning] and fastai [@howard2020fastai] reduce boilerplate but
introduce their own abstractions and learning curves, and are oriented mainly
toward classification.

`FujiCV` occupies a deliberate middle ground: a small, config-light API that (i)
reduces the code needed for a complete, production-grade pipeline, (ii) adds no
meaningful throughput or memory overhead over an equivalent hand-written loop,
and (iii) unifies classification, regression, multi-label, and retrieval — plus
explainability and deployment — under one interface. Correctness is treated as a
first-class concern: the library ships an extensive automated test suite and
continuous integration, and captures a resolved configuration (with package and
git versions) for every run to support reproducibility.

The library is designed for rapid prototyping, teaching, competitive machine
learning (e.g. Kaggle), and applied research where the goal is results rather
than framework engineering.

# State of the field

`FujiCV` complements rather than replaces existing tools. It builds on `timm`
[@rw2019timm] and `torchvision` for backbones, `albumentations`
[@buslaev2020albumentations] for augmentation, and `scikit-learn`
[@pedregosa2011scikit] for metrics, and interoperates with the Hugging Face
ecosystem — including modern self-supervised and vision-language encoders such as
DINOv2 [@oquab2024dinov2], CLIP [@radford2021clip], and SigLIP [@zhai2023siglip],
whose vision towers can be used directly as classification backbones. Relative to
PyTorch Lightning and fastai, `FujiCV` emphasizes a smaller surface area, broader
task coverage (regression and retrieval as first-class citizens), and an
integrated deployment/quantization path.

# Acknowledgements

We thank the maintainers of PyTorch, `timm`, `albumentations`, and the wider
open-source computer-vision community, whose work `FujiCV` builds upon.

# References
