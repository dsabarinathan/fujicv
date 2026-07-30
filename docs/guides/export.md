# Export & Deployment

## ONNX export

```bash
pip install "fujicv[onnx]"
```

```python
from fujicv.export import to_onnx, verify_onnx

to_onnx(model, "model.onnx", input_size=(1, 3, 224, 224))
verify_onnx(model, "model.onnx")   # numeric check vs PyTorch
```

## ONNX INT8 quantization

Reduce model size ~4× and speed up CPU inference — no calibration data required:

```python
from fujicv.export import quantize_onnx

quantize_onnx("model.onnx", "model_int8.onnx")
```

## TorchScript export

```python
import torch
from fujicv.export import export_torchscript, verify_torchscript, load_torchscript

example = torch.randn(1, 3, 224, 224)
scripted = export_torchscript(model, "model.pt", example, method="trace")
verify_torchscript(scripted, example, original_model=model)

# Load later
model_loaded = load_torchscript("model.pt", map_location="cpu")
```

## Comparison

| Format | Runtime | Quantization | Cross-language |
|---|---|---|---|
| **PyTorch** (`.pt`) | PyTorch only | Manual | No |
| **ONNX** | ONNXRuntime, TensorRT, … | Yes (INT8) | Yes |
| **TorchScript** | LibTorch (C++) | Manual | Yes (C++) |
