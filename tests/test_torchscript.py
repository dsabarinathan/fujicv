"""Tests for TorchScript export utilities."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn


def _model():
    return nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 5))


def _inputs():
    return torch.randn(2, 3, 8, 8)


def test_export_torchscript_creates_file():
    from fujicv.export.torchscript import export_torchscript
    model = _model().eval()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "model.pt"
        export_torchscript(model, out, _inputs())
        assert out.exists()


def test_export_torchscript_returns_script_module():
    from fujicv.export.torchscript import export_torchscript
    model = _model().eval()
    with tempfile.TemporaryDirectory() as tmp:
        out      = Path(tmp) / "model.pt"
        scripted = export_torchscript(model, out, _inputs())
        assert isinstance(scripted, torch.jit.ScriptModule)


def test_export_torchscript_invalid_method_raises():
    from fujicv.export.torchscript import export_torchscript
    model = _model().eval()
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="method must be"):
            export_torchscript(model, Path(tmp) / "m.pt", _inputs(), method="bad")


def test_verify_torchscript_passes():
    from fujicv.export.torchscript import export_torchscript, verify_torchscript
    model = _model().eval()
    with tempfile.TemporaryDirectory() as tmp:
        out      = Path(tmp) / "model.pt"
        scripted = export_torchscript(model, out, _inputs())
        assert verify_torchscript(scripted, _inputs(), original_model=model)


def test_verify_torchscript_no_original():
    from fujicv.export.torchscript import export_torchscript, verify_torchscript
    model = _model().eval()
    with tempfile.TemporaryDirectory() as tmp:
        out      = Path(tmp) / "model.pt"
        scripted = export_torchscript(model, out, _inputs())
        # With no original model to compare, should still return True
        assert verify_torchscript(scripted, _inputs())


def test_load_torchscript_roundtrip():
    from fujicv.export.torchscript import export_torchscript, load_torchscript
    model = _model().eval()
    x     = _inputs()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "model.pt"
        export_torchscript(model, out, x)
        loaded = load_torchscript(str(out), map_location="cpu")
        with torch.no_grad():
            orig_out   = model(x)
            loaded_out = loaded(x)
        assert torch.allclose(orig_out, loaded_out, atol=1e-4)


def test_export_creates_parent_dirs():
    from fujicv.export.torchscript import export_torchscript
    model = _model().eval()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "subdir" / "deep" / "model.pt"
        export_torchscript(model, out, _inputs())
        assert out.exists()
