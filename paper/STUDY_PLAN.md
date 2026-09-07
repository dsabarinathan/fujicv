# FujiCV — Study Plan for SIVP Q2

**Target journal:** Signal, Image and Video Processing (Springer), Q2  
**Current paper skeleton:** `main.tex` + `paper.md`  
**Benchmark harness:** `paper/benchmarks/` — every experiment writes JSON to `results/`

---

## S0 — Environment setup (do this first)

### 1. Clone the repo

```bash
git clone https://github.com/dsabarinathan/fujicv.git
cd fujicv
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA

Install the wheel that matches your driver. The paper hardware is an RTX 2070 with CUDA 11.7:

```bash
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 \
    --index-url https://download.pytorch.org/whl/cu117
```

For CUDA 12.x or CPU-only, substitute the appropriate index URL from [pytorch.org](https://pytorch.org/get-started/locally/).

### 4. Install FujiCV in editable mode

Install the package plus the extras needed for the benchmarks (ONNX export, dev/test tools):

```bash
pip install -e ".[dev,onnx,hpo,retrieval]"
```

What each extra adds:

| Extra | Packages | Needed for |
|-------|----------|-----------|
| `dev` | pytest, ruff, mypy, detect-secrets | running tests, CI |
| `onnx` | onnxruntime, onnx | S7 deployment benchmarks |
| `hpo` | optuna | hyperparameter search |
| `retrieval` | faiss-cpu | S8 CUB-200 retrieval |

Optional extras (install if you have accounts set up):

```bash
pip install -e ".[wandb,tensorboard]"   # experiment logging
```

### 5. Verify the install

```bash
python - <<'EOF'
import torch, fujicv
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("fujicv:", fujicv.__version__)

# smoke test: build a model and run a forward pass
from fujicv.models.builder import ModelBuilder
model = ModelBuilder("resnet18", task="classification", num_outputs=10, image_size=32).build()
import torch
x = torch.zeros(2, 3, 32, 32)
out = model(x)
print("forward pass shape:", out.shape)   # expect torch.Size([2, 10])
print("Setup OK")
EOF
```

Expected output:
```
torch: 2.0.1+cu117
cuda available: True
GPU: NVIDIA GeForce RTX 2070
fujicv: 1.16.0
forward pass shape: torch.Size([2, 10])
Setup OK
```

### 6. Run the test suite (optional but recommended)

```bash
pytest tests/ -q
```

All tests should pass before running any benchmark.

---

## Status overview

| ID | Study | Status | Priority |
|----|-------|--------|----------|
| S1 / E1 | Boilerplate reduction (LOC) | **Done** | — |
| S2 / E2 | Throughput & memory overhead (single-GPU) | **Done** | — |
| S3 / E3 | Accuracy parity — CIFAR-10 scratch | **Done** | — |
| S3b | Accuracy parity — pretrained, Flowers102 | **TODO** | High |
| S3c | Accuracy parity — multiple backbones | **TODO** | Medium |
| S4 / E4 | Built-in technique ablation | **TODO** | High |
| S5 | Regression task study (UTKFace age) | **TODO** | High |
| S6 | Multi-label classification (PASCAL VOC 2007) | **TODO** | High |
| S7 / E5 | Deployment — fix static INT8 via ONNX path | **Partial** | Medium |
| S8 / E6 | Retrieval — ArcFace + GeM on CUB-200 | **TODO** | Medium |
| S9 | Grad-CAM / explainability (qualitative) | **TODO** | Medium |
| S10 / E2-DDP | DDP throughput (E2 DDP rows are TBD) | **TODO** | Low |

---

## Detailed study descriptions

---

### S1 / E1 — Boilerplate reduction ✅ Done

**What:** Non-comment, non-blank LOC for an identical CIFAR-10 pipeline (AMP, cosine schedule, accuracy, best-checkpoint, early stopping) across Raw PyTorch, PyTorch Lightning, and FujiCV.

**Result:** Raw 94 → Lightning 69 → FujiCV **61** (35% fewer lines than raw PyTorch).

**Paper section:** §E1, Table 2.  
**Script:** `benchmarks/bench_loc.py`

---

### S2 / E2 — Throughput & memory overhead ✅ Done

**What:** Training throughput (img/s) and peak GPU memory, FujiCV vs Raw PyTorch (both do: train + validate + track accuracy + checkpoint), ResNet-18, 64×64, AMP, batch 256, 3 runs.

**Result:** FujiCV 330 img/s vs Raw 344 img/s, **+4.1% overhead**, identical memory (436.5 MB).

**Paper section:** §E2, Table 3.  
**Script:** `benchmarks/bench_throughput.py`

---

### S3 / E3 — Accuracy parity CIFAR-10 ✅ Done

**What:** Identical from-scratch recipe (ResNet-18 torchvision, 5 epochs, AdamW, light aug), FujiCV vs hand-written loop.

**Result:** FujiCV 72.9% vs Raw 71.9%, Δ = **+1.0 pt** (within variance).

**Paper section:** §E3, Table 4.  
**Script:** `benchmarks/bench_accuracy.py`

---

### S3b — Accuracy parity with pretrained backbone (Flowers102) ⬜ TODO

**Why:** CIFAR-10 from-scratch at 5 epochs is deliberately minimal. SIVP reviewers will ask for a more realistic setting. Pretrained fine-tuning is the dominant real-world use case and gives cleaner parity evidence (less noise from random init).

**Task:** Fine-tune ResNet-18 (ImageNet pretrained, `torchvision`) on Oxford Flowers-102 for 15 epochs. Compare FujiCV vs a matched hand-written loop. Report top-1 val accuracy.

**Dataset:** `torchvision.datasets.Flowers102` — auto-downloads (~330 MB). 102 classes, 8,189 images.

**Config:**
```
backbone: resnet18 (torchvision, pretrained=True)
image_size: 224
epochs: 15
batch_size: 64
optimizer: AdamW, lr=1e-4 (head), lr=1e-5 (backbone)
scheduler: cosine warmup
aug_level: medium
```

**Expected output:** `results/e3b_accuracy_flowers102.json`

```bash
python benchmarks/bench_accuracy.py \
  --dataset flowers102 --backbone resnet18 \
  --source torchvision --pretrained \
  --epochs 15 --batch-size 64 --aug-level medium
```

**Paper contribution:** Additional row in Table 4. Proves parity in the more common pretrained fine-tuning regime.

---

### S3c — Accuracy parity across backbone families ⬜ TODO

**Why:** SIVP expects the claim "the abstraction does not cost accuracy" to hold beyond one architecture.

**Task:** Run the same Flowers102 pretrained recipe on:
- `convnext_tiny` (timm, CNN transformer hybrid)
- `vit_small_patch16_224` (timm, pure ViT)

Compare FujiCV vs matched hand-written loops for each.

**Expected output:** `results/e3c_accuracy_convnext_flowers102.json`, `results/e3c_accuracy_vit_flowers102.json`

**Paper contribution:** Additional rows in Table 4, showing CNN and ViT both stay within variance.

---

### S4 / E4 — Built-in technique ablation ⬜ TODO

**Why:** This is explicitly TBD in the paper (`main.tex`, §E4). It is the strongest quantitative argument for FujiCV's value — showing that built-in techniques give "free" accuracy gains vs the baseline.

**Task:** Start from the Flowers102 pretrained ResNet-18 baseline (S3b). Cumulatively enable each technique and measure val accuracy:

| Step | Config change | Expected Δ |
|------|--------------|------------|
| Baseline | AdamW, cosine, no extras | — |
| + aug=heavy | `aug_level: heavy` | +0.5–2 pt |
| + Mixup/CutMix | `mixup_alpha=0.2, cutmix_alpha=1.0` | +0.5–1.5 pt |
| + EMA | `ema_decay=0.9998` | +0.3–1 pt |
| + SWA | `swa_start=0.75` (last 25% of epochs) | +0.3–1 pt |
| + Model soup | Uniform soup over last 3 checkpoints | +0.2–0.5 pt |

Each configuration: 1 run, same seed (42). Report val accuracy and cumulative Δ.

**Expected output:** `results/e4_ablation_flowers102.json`

```bash
python benchmarks/bench_ablation.py \
  --dataset flowers102 --backbone resnet18 \
  --source torchvision --pretrained --epochs 20
```

**Paper contribution:** Table 5 (§E4) — currently all TBD cells. This unlocks the ablation section.

---

### S5 — Regression task study ⬜ TODO

**Why:** FujiCV claims first-class regression support, but the paper has zero regression evidence. SIVP reviewers will notice. Without this, regression is just a feature claim.

**Task:** Age estimation on UTKFace (or IMDB-WIKI subset if UTKFace is unavailable). Compare:
1. FujiCV regression pipeline (MSE/MAE loss, R² metric)
2. Equivalent hand-written raw PyTorch loop

**Dataset:** UTKFace — ~23,000 face images with age labels (0–116). Download from [Kaggle](https://www.kaggle.com/datasets/jangedoo/utkface-new) or use a public mirror.

**Config:**
```
backbone: resnet18 (torchvision, pretrained)
task: regression
loss: mae (L1)
metrics: MAE, RMSE, R²
image_size: 128
epochs: 20
batch_size: 64
```

**What to report:**
- FujiCV vs Raw PyTorch: MAE, RMSE, R² on the test set
- Code lines needed for regression (additional LOC row in Table 2)

**Expected output:** `results/e_regression_utk.json`

**Paper contribution:** New §E7 (Regression task) + additional row in Table 2 (LOC). This is a key SIVP differentiator — regression on image data is squarely in-scope for the journal.

---

### S6 — Multi-label classification (PASCAL VOC 2007) ⬜ TODO

**Why:** Multi-label is another first-class claim with no supporting experiment.

**Task:** Multi-label classification on PASCAL VOC 2007 (20 classes, ~10,000 images). Compare FujiCV (BCE + ASL loss, mAP metric) vs hand-written baseline.

**Dataset:** `torchvision.datasets.VOCDetection` or `VOCSegmentation` — auto-downloadable. Parse bounding box annotations into multi-hot labels.

**Config:**
```
backbone: convnext_tiny (timm, pretrained)
task: multilabel
loss: asymmetric_loss (ASL)
metrics: mAP, per-class AUROC
image_size: 224
epochs: 20
batch_size: 32
```

**What to report:**
- FujiCV vs Raw: mAP on test set
- Code lines for multi-label (LOC Table row)

**Expected output:** `results/e_multilabel_voc2007.json`

**Paper contribution:** New §E8 (Multi-label task). Completes the "three tasks" promise.

---

### S7 / E5 — Deployment: fix static INT8 via ONNX path ⬜ Partial

**Current state:**
- FP32: 42.73 MB, 150 ms latency
- Dynamic INT8: 42.72 MB, 185 ms (no benefit on CNN — expected)
- Static INT8 (FX): **failed** — FujiCV assembled model not FX-traceable

**What to do:**
1. Export FujiCV model to ONNX first (`fujicv.export.to_onnx`)
2. Apply ONNX INT8 quantization via `onnxruntime.quantization`
3. Measure latency and size on ONNX Runtime (CPU)
4. Add TorchScript export as a lightweight alternative

**Config:** ResNet-18, 224×224, batch=1, CPU, 50 inference iters, 3 runs.

**Expected output:** `results/e5_deployment_onnx_int8.json` with columns: FP32, ONNX FP32, ONNX INT8

**Paper contribution:** Complete Table 6 (§E5). The honest comparison (dynamic INT8 ≈ no benefit on CNN; ONNX INT8 gives real size/speed reduction) is scientifically valuable and already framed correctly in §E5.

```bash
python benchmarks/bench_deployment.py \
  --backbone resnet18 --img 224 --iters 50 --mode onnx_int8
```

---

### S8 / E6 — Retrieval: ArcFace + GeM on CUB-200 ⬜ TODO

**Why:** Table 6 (§E6 retrieval) is fully TBD. Retrieval is a differentiator vs Lightning/fastai.

**Task:** Metric learning on CUB-200-2011 (200 bird species, 11,788 images). Standard split: first 100 classes for training, last 100 for retrieval evaluation.

**Config:**
```
backbone: resnet50 (timm, pretrained)
head: ArcFace (margin=0.5, scale=64)
pooling: GeM (p=3)
image_size: 224
epochs: 25
batch_size: 32
```

**What to report:**
- Recall@1, Recall@5, mAP@10 on the retrieval split
- Compare with vanilla cosine similarity baseline (no ArcFace)

**Expected output:** `results/e6_retrieval_cub200.json`

**Paper contribution:** Table 7 (§E6 retrieval). Adds the retrieval evidence that is currently all TBD.

---

### S9 — Grad-CAM / explainability (qualitative) ⬜ TODO

**Why:** Explainability is listed as a feature (§Features table) but has no paper evidence.

**Task:** Generate Grad-CAM and Grad-CAM++ visualization grids for:
- A CNN backbone (ResNet-18 on Flowers102): correct vs misclassified examples
- A ViT backbone (ViT-Small on Flowers102): attention rollout map

**What to produce:**
- 2 publication-quality figures (PDF): `figures/gradcam_resnet18_flowers102.pdf`, `figures/attention_rollout_vit_flowers102.pdf`
- Each figure: 4 correct + 4 misclassified examples, image + heatmap overlay + label

**Script:**
```bash
python examples/evaluate.py \
  --config runs/flowers102_resnet18/resolved_config.yaml \
  --checkpoint runs/flowers102_resnet18/best.pt \
  --with-attention-maps --n-correct 4 --n-wrong 4
```

**Paper contribution:** Figure 2 in §Features or a new §E9. Visual evidence of explainability working correctly on both CNN and ViT.

---

### S10 / E2-DDP — Distributed training throughput ⬜ TODO

**Why:** E2 DDP rows in Table 3 are currently TBD. DDP is listed as a key correctness contribution.

**What:** Only feasible with 2 GPUs. On single RTX 2070 hardware, this requires either:
- Running on a cloud instance (Colab Pro / RunPod with 2× A100/V100), OR
- Reporting DDP correctness (all-gathered metrics, rank-guarded checkpoints) via unit tests instead of throughput numbers, and removing the TBD rows

**Recommendation:** If a cloud run is not feasible, replace the TBD DDP rows with a footnote: *"DDP correctness is validated by automated tests (CI); throughput scaling on multi-GPU hardware is reserved for future work."* Remove the TBD rows from the table to avoid reviewer questions.

**Priority:** Low — do not block paper submission on this.

---

## Execution order

Run studies in this order to minimize data re-download and reuse trained models:

```
Week 1 (unblock paper):
  S3b  →  Flowers102 pretrained accuracy parity
  S4   →  Ablation on Flowers102 (builds on S3b checkpoint)
  S3c  →  ConvNeXt + ViT accuracy parity (same dataset)

Week 2 (broaden task coverage):
  S5   →  UTKFace regression
  S6   →  PASCAL VOC multi-label
  S7   →  Deployment ONNX INT8 fix

Week 3 (complete remaining):
  S8   →  CUB-200 retrieval
  S9   →  Grad-CAM figures (uses S3b checkpoint)
  S10  →  DDP or remove TBD rows (decision point)
```

---

## Paper → study mapping

| Paper section | Needs |
|--------------|-------|
| §E1 Table 2 | S1 ✅ + add regression/multilabel LOC rows (S5, S6) |
| §E2 Table 3 | S2 ✅ + DDP rows (S10 or remove TBD) |
| §E3 Table 4 | S3 ✅ + S3b (Flowers102 pretrained) + S3c (ConvNeXt, ViT) |
| §E4 Table 5 | S4 (ablation — all TBD) |
| §E5 Table 6 | S7 (ONNX INT8 fix) |
| §E6 Table 7 | S8 (CUB-200 retrieval) |
| §E7 (new) | S5 (regression) |
| §E8 (new) | S6 (multi-label) |
| Figure 2 | S9 (Grad-CAM / attention rollout) |

---

## SIVP Q2 acceptance checklist

Before submission, verify:

- [ ] All TBD cells in tables are filled with real numbers
- [ ] At least 3 tasks demonstrated with data: classification ✅, regression (S5), multi-label (S6)
- [ ] Accuracy claim covers ≥ 2 datasets and ≥ 2 backbone families (S3b, S3c)
- [ ] Ablation quantifies benefit of built-in techniques (S4)
- [ ] Deployment path complete: ONNX FP32 + ONNX INT8 (S7)
- [ ] Retrieval results present (S8) or retrieval removed from claims
- [ ] At least one qualitative figure (Grad-CAM, S9) for visual evidence
- [ ] main.tex compiled clean (no overfull hbox, no missing refs)
- [ ] All benchmark JSON files committed to `paper/benchmarks/results/`
- [ ] Author ORCID updated (currently `0000-0000-0000-0000`)
- [ ] `paper.bib` contains all cited works with DOIs
- [ ] Switched `\documentclass` to `svjour3` with `\journalname{Signal, Image and Video Processing}`
