# Multi-domain decentralized LoRA paper

**Why Multi-Domain Decentralized LoRA Falls Short of Pooled Training** is the
standalone report of the original research. This directory includes its own
LaTeX source, bibliography, result-table input, and every figure it uses;
compilation does not read from another manuscript directory.

The corrected nine-arm, three-seed CIFAR-100 study finds that the complete
adaptive-rank, domain-weighted protocol fails to preserve pooled accuracy:
6.90 ± 2.03% versus 57.23 ± 0.59%. The paper retains the completed ownership,
refactorization, projection-retention, and optimizer-order diagnostics;
the exact synchronized-gradient positive control; the unsuccessful
retained-base residual screen; and the later partial-gradient validation study.
The latter narrows the gap while retaining full rank-16 model and optimizer
state. Neither result establishes the original complete-model resource claim
or a privacy guarantee. Supporting rank, domain-allocation, and discovery
studies keep their distinct protocols and qualifications.

Raw comparisons and verification records are archived in
[`../../docs/artifacts/integrated-corrected/`](../../docs/artifacts/integrated-corrected/)
and [`../../docs/artifacts/methodology-exploration/`](../../docs/artifacts/methodology-exploration/).
Figure sources and endpoint qualifications are listed in
[`figures/README.md`](figures/README.md).

Build and render on the designated **gpu003** machine, from this directory:

```bash
pdflatex -halt-on-error -interaction=nonstopmode main.tex
bibtex main
pdflatex -halt-on-error -interaction=nonstopmode main.tex
pdflatex -halt-on-error -interaction=nonstopmode main.tex
mkdir -p rendered
pdftoppm -r 90 -png main.pdf rendered/main
```

Use the installed TeX Live 2026 toolchain at
`~/ahlora-tools/texlive-202609/.TinyTeX/bin/x86_64-linux`.
Every page of the final PDF requires visual review after substantive changes.
