"""E5 — Deployment: post-training quantization size & CPU latency.

Measures FP32 vs. dynamic-INT8 vs. static-INT8 (FX) model size and CPU inference
latency using FujiCV's export.quantization utilities.

Example:
    python bench_deployment.py --backbone resnet18 --img 224 --iters 50
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from common import save_result, set_seed


def cpu_latency(model, example, iters: int, warmup: int = 5) -> dict:
    model.eval()
    times = []
    with torch.no_grad():
        for i in range(iters + warmup):
            t0 = time.perf_counter()
            model(example)
            dt = (time.perf_counter() - t0) * 1000.0  # ms
            if i >= warmup:
                times.append(dt)
    return {"mean_ms": statistics.mean(times), "std_ms": statistics.pstdev(times)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    set_seed(0)
    from fujicv.export.quantization import measure_model_size, quantize_dynamic, quantize_static
    from fujicv.models.builder import ModelBuilder

    model = ModelBuilder(args.backbone, backbone_source="timm", pretrained=False,
                         task="classification", num_outputs=10, image_size=args.img).build().cpu()
    example = torch.randn(args.batch_size, 3, args.img, args.img)

    result = {"config": vars(args), "variants": {}}

    result["variants"]["fp32"] = {
        "size_mb": measure_model_size(model),
        "latency": cpu_latency(model, example, args.iters),
    }

    qd = quantize_dynamic(model)
    result["variants"]["dynamic_int8"] = {
        "size_mb": measure_model_size(qd),
        "latency": cpu_latency(qd, example, args.iters),
    }

    try:
        calib = [torch.randn(args.batch_size, 3, args.img, args.img) for _ in range(8)]
        qs = quantize_static(model, calib, num_calibration_batches=8)
        result["variants"]["static_int8"] = {
            "size_mb": measure_model_size(qs),
            "latency": cpu_latency(qs, example, args.iters),
        }
    except Exception as exc:  # backend may be unavailable
        result["variants"]["static_int8"] = {"error": str(exc)}

    save_result("e5_deployment", result)
    for name, v in result["variants"].items():
        if "size_mb" in v:
            print(f"  {name:14} {v['size_mb']:6.2f} MB   {v['latency']['mean_ms']:6.2f} ms")
        else:
            print(f"  {name:14} skipped ({v.get('error','')[:60]})")


if __name__ == "__main__":
    main()
