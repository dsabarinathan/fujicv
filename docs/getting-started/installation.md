# Installation

## Prerequisites

FujiCV requires **Python 3.9+** and **PyTorch 2.0+**.

Install PyTorch first (choose the right CUDA build for your system):

```bash
# CPU only
pip install torch torchvision

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

See [pytorch.org/get-started](https://pytorch.org/get-started/locally/) for all options.

## Install FujiCV

```bash
# Core package
pip install fujicv

# With optional extras
pip install "fujicv[wandb]"       # W&B logging
pip install "fujicv[onnx]"        # ONNX export + INT8 quantization
pip install "fujicv[hpo]"         # Optuna hyperparameter search
pip install "fujicv[wandb,onnx,hpo]"   # all extras

# Development (includes pytest, ruff, mypy)
pip install "fujicv[dev]"
```

## Verify installation

```python
import fujicv
print(fujicv.__version__)

from fujicv.models.builder import ModelBuilder
model = ModelBuilder("resnet18", task="classification", num_outputs=10, pretrained=False).build()
print("ModelBuilder OK")
```
