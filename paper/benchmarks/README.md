# FujiCV Paper Benchmarks

Reproducible harness for the experiments in the paper (E1–E6). Every script
writes a JSON file into `results/`, and `make_tables.py` turns those into the
LaTeX tables used in `../main.tex`.

## Setup

```bash
pip install "fujicv[dev,onnx]"   # torch, timm, albumentations, etc.
cd paper/benchmarks
```

Each script captures the software/hardware environment (torch version, GPU, git
hash, seeds) into its result file automatically — see `common.py`.

## Experiments

| ID | Script | Produces | Needs |
|----|--------|----------|-------|
| E1 | `bench_loc.py` | `results/e1_loc.json` | the compared training scripts |
| E2 | `bench_throughput.py` | `results/e2_throughput.json` | GPU (uses synthetic data) |
| E3 | `bench_accuracy.py` | `results/e3_accuracy_*.json` | dataset auto-download |
| E4 | `bench_ablation.py` | `results/e4_ablation_*.json` | dataset auto-download |
| E5 | `bench_deployment.py` | `results/e5_deployment.json` | CPU only |

### E1 — Lines of code

```bash
python bench_loc.py \
  --script "Raw PyTorch=/path/to/baseline_pytorch.py" \
  --script "PyTorch Lightning=/path/to/baseline_lightning.py" \
  --script "FujiCV=../../examples/train_cifar10.py"
```

### E2 — Throughput & peak memory (mean ± std over N runs, with warmup)

```bash
python bench_throughput.py --runs 3 --epochs 1 --batch-size 256
```

> For the paper, replace the synthetic loader in `build_loaders` with your real
> dataset (e.g. CelebA) so the numbers are dataset-representative. Run the
> single-GPU comparison **back to back** and report mean ± std — throughput is
> sensitive to machine contention.

### E3 — Accuracy parity

```bash
python bench_accuracy.py --dataset cifar10 --backbone resnet18 --epochs 10
```

### E4 — Built-in technique ablation

```bash
python bench_ablation.py --dataset cifar10 --epochs 15
```

### E5 — Deployment / quantization (CPU)

```bash
python bench_deployment.py --backbone resnet18 --img 224 --iters 50
```

## Generate the LaTeX tables

```bash
python make_tables.py > tables.tex
```

Paste the emitted table bodies into the corresponding `\begin{tabular}` blocks in
`../main.tex` (labels match).

## Notes on rigor

- Fix seeds (done by `common.set_seed`) and report hardware/versions (captured).
- Discard a warmup run before timing (E2 does this).
- Report mean ± std over ≥3 runs for any timing number.
- E6 (retrieval) uses a dedicated benchmark dataset (CUB-200 / SOP); build
  embeddings with `fujicv.retrieval.Embedder` and score with
  `fujicv.retrieval.evaluate_retrieval`.
