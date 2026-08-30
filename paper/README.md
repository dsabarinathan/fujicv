# FujiCV — Paper Materials

Everything needed to write and reproduce the FujiCV paper.

```
paper/
├── paper.md            # JOSS submission (short; submit as-is after edits)
├── main.tex            # Full manuscript skeleton (SIVP / SoftwareX / arXiv)
├── paper.bib           # Shared bibliography (used by both paper.md and main.tex)
├── benchmarks/         # Reproducible experiment harness (E1–E6) + LaTeX table gen
│   ├── common.py       #   env capture, seeding, timing, JSON I/O
│   ├── bench_*.py      #   one script per experiment
│   ├── make_tables.py  #   results/*.json -> LaTeX tables
│   └── results/        #   JSON outputs (created on first run)
└── README.md           # this file
```

## Two submission tracks (do both)

1. **JOSS** — `paper.md` is already written and close to submittable. JOSS reviews
   the *software* (tests, docs, API), so the heavy lifting is already done. Edit
   the author ORCID/affiliation, then submit at <https://joss.theoj.org>. Gives a
   citable DOI quickly.
2. **SIVP / SoftwareX / arXiv** — `main.tex` is the full manuscript. Fill the
   `TBD` cells from the benchmark results, write the prose sections (stubs are
   marked with `%` comments), and add a pipeline figure.

## Steps to a finished paper

1. **Freeze the release.** Tag the version the paper describes (e.g. `v1.16.0`)
   and mint a **Zenodo DOI** (enable the GitHub↔Zenodo integration once, then
   publish a release). Cite that DOI in both papers.
2. **Run the experiments.** See `benchmarks/README.md`. Each writes JSON to
   `benchmarks/results/`.
3. **Generate tables.** `cd benchmarks && python make_tables.py > tables.tex`,
   then paste into `main.tex`.
4. **Write the prose.** Fill Sections 1–4, 7–9 of `main.tex` (Intro, Related
   Work, Design, Features, Software Quality, Threats to Validity, Conclusion).
5. **Make the figure.** A single pipeline diagram
   (ModelBuilder → Trainer → eval/export) goes a long way; add training curves
   from `fujicv.eval.plot_loss_curves`.
6. **Compile.** `pdflatex main && bibtex main && pdflatex main && pdflatex main`.
   For SIVP switch the class to `svjour3`; for SoftwareX use `elsarticle`.
7. **Post arXiv preprint**, then submit to the journal.

## Claims and how each is supported

| Claim | Evidence | Experiment |
|---|---|---|
| Less boilerplate than raw PyTorch / heavy frameworks | LoC table | E1 |
| Negligible throughput/memory overhead | timing + peak-mem table | E2 |
| No accuracy loss from the abstraction | parity table | E3 |
| Built-ins add real value | ablation table | E4 |
| Deployment-ready (edge) | size/latency table | E5 |
| Breadth: retrieval as a first-class task | Recall@K / mAP@K | E6 |
| Correct & reproducible | 375 tests, CI, resolved-config capture | §Software Quality |

## Honesty checklist (pre-empt reviewers)

- Frame the contribution as **standardized, tested integration**, not new
  algorithms — many components wrap `timm` / `albumentations` / Hugging Face.
- Report throughput as **mean ± std over ≥3 runs after warmup**; a single-GPU
  run where the wrapper looks *faster* than raw PyTorch is noise — control for it.
- State that DDP is validated on a single node (2 GPUs); note multi-node is
  future work.
- LoC is an imperfect effort proxy; state the counting rule.
