"""Generate LaTeX tables from the benchmark JSON results in ``results/``.

Reads whichever ``e*.json`` files exist and prints ready-to-paste LaTeX table
bodies (matching the labels in ../main.tex). Missing results are skipped.

Example:
    python make_tables.py            # print all available tables
    python make_tables.py > tables.tex
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def _load(name: str):
    p = RESULTS / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def table_loc():
    d = _load("e1_loc")
    if not d:
        return None
    rows = sorted(d["counts"].items(), key=lambda kv: -kv[1]["loc"])
    body = "\n".join(
        f"{lbl} & {v['loc']} & {v['relative']:.2f}$\\times$ \\\\" for lbl, v in rows
    )
    return f"% E1 lines of code\n{body}"


def table_throughput():
    d = _load("e2_throughput")
    if not d:
        return None
    r, f = d["raw_pytorch"], d["fujicv"]
    over = d["fujicv_throughput_overhead_frac"] * 100

    def cell(x):
        return f"{x['mean']:.1f}$\\pm${x['std']:.1f}"

    body = (
        f"Single-GPU & Raw PyTorch & {cell(r['throughput_img_s'])} & {cell(r['peak_mem_mb'])} & --- \\\\\n"
        f"Single-GPU & FujiCV & {cell(f['throughput_img_s'])} & {cell(f['peak_mem_mb'])} & {over:+.1f}\\% \\\\"
    )
    return f"% E2 throughput/memory\n{body}"


def table_accuracy():
    lines = []
    for ds in ("cifar10", "mnist"):
        d = _load(f"e3_accuracy_{ds}")
        if d:
            acc = d["best_val_accuracy"] * 100
            lines.append(f"{ds} ({d['config']['backbone']}) & --- & {acc:.2f} & --- \\\\")
    return "% E3 accuracy\n" + "\n".join(lines) if lines else None


def table_ablation():
    for ds in ("cifar10", "mnist"):
        d = _load(f"e4_ablation_{ds}")
        if not d:
            continue
        base = d.get("baseline")
        lines = []
        for name, acc in d["results"].items():
            delta = "---" if (base is None or name == "baseline") else f"{(acc - base) * 100:+.2f}"
            lines.append(f"{name} & {acc * 100:.2f} & {delta} \\\\")
        return "% E4 ablation\n" + "\n".join(lines)
    return None


def table_deployment():
    d = _load("e5_deployment")
    if not d:
        return None
    lines = []
    labels = {"fp32": "FP32", "dynamic_int8": "Dynamic INT8", "static_int8": "Static INT8"}
    for key, label in labels.items():
        v = d["variants"].get(key)
        if v and "size_mb" in v:
            lines.append(f"{label} & {v['size_mb']:.2f} & {v['latency']['mean_ms']:.2f} & --- \\\\")
    return "% E5 deployment\n" + "\n".join(lines) if lines else None


def main():
    builders = [
        ("Table 3 (E1 LoC)", table_loc),
        ("Table 4 (E2 throughput)", table_throughput),
        ("Table 5 (E3 accuracy)", table_accuracy),
        ("Table 6 (E4 ablation)", table_ablation),
        ("Table 7 (E5 deployment)", table_deployment),
    ]
    any_found = False
    for title, fn in builders:
        out = fn()
        if out:
            any_found = True
            print(f"\n%% ===== {title} =====")
            print(out)
    if not any_found:
        print("No results found in results/. Run the bench_*.py scripts first.")


if __name__ == "__main__":
    main()
