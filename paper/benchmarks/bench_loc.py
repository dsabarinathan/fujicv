"""E1 — Lines-of-code comparison for an equivalent training pipeline.

Counts non-comment, non-blank, non-docstring lines of one or more training
scripts and writes a JSON result. Point it at the FujiCV script and each baseline
(raw PyTorch, Lightning, fastai) that implement the SAME pipeline.

Example:
    python bench_loc.py \
        --script "Raw PyTorch=../../examples/baseline_pytorch.py" \
        --script "FujiCV=../../examples/train_cifar10.py"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from common import save_result


def count_loc(path: str | Path) -> int:
    """Count code lines, excluding blank lines, comments, and triple-quoted docstrings."""
    n = 0
    in_doc = False
    quote = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if in_doc:
            if quote in s:
                in_doc = False
            continue
        if not s or s.startswith("#"):
            continue
        # Opening of a docstring/triple-quoted block on its own line.
        for q in ('"""', "'''"):
            if s.startswith(q):
                if s.count(q) == 1:  # block continues on later lines
                    in_doc, quote = True, q
                break
        else:
            n += 1
            continue
        if not in_doc:  # single-line triple-quoted string
            continue
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", action="append", default=[],
                    help='Repeatable "Label=path/to/script.py".')
    args = ap.parse_args()
    if not args.script:
        ap.error("provide at least one --script Label=path")

    counts: Dict[str, int] = {}
    for spec in args.script:
        label, _, path = spec.partition("=")
        counts[label.strip()] = count_loc(path.strip())

    baseline = max(counts.values())
    table = {
        lbl: {"loc": loc, "relative": round(loc / baseline, 3)}
        for lbl, loc in counts.items()
    }
    save_result("e1_loc", {"counts": table})
    for lbl, d in sorted(table.items(), key=lambda kv: -kv[1]["loc"]):
        print(f"  {lbl:20} {d['loc']:4d} LoC  ({d['relative']:.2f}x)")


if __name__ == "__main__":
    main()
