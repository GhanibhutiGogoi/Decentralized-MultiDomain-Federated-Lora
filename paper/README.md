# AH-LoRA paper

The completed 38-page PDF was compiled and rendered on gpu003 on 2026-09-18.
Every page passed visual review, and the final build has no layout or reference
warnings. The [build record](../docs/artifacts/quantity-skew/paper-build/README.md)
binds the PDF to its source, figures and reviewed page renders.

`main.tex` reports two completed studies, with mathematical definitions integrated
into the methodology. The new quantity-skew study compares seven decentralized
RoBERTa-base/SST-2 methods at five matched seeds, plus a separate equal-size
Dec-LoRA reconstruction anchor. All 36 full-budget runs completed; separate
inference audits reproduce all 72 best/final checkpoint prediction vectors.

Adaptive rank with sample-size weighting reaches **94.24 ± 0.25%** best validation
accuracy versus **94.54 ± 0.24%** for reconstructed Dec-LoRA16. The paired difference
is −0.298 percentage points, with an unadjusted 95% interval of [−0.755, +0.159].
This does not establish parity. Final accuracy is 93.90% versus 94.45%, with a
paired difference interval of [−0.919, −0.182] points. Training traffic is 31.81%
lower than rank 16, but **8.92% higher** than feasible Dec-LoRA4, which reaches
94.50 ± 0.33% best validation accuracy. The proposal therefore demonstrates a
working protocol and a measured tradeoff, not a win or superior feasible
accuracy/traffic tradeoff. The sample-weighting ablation shows no established
accuracy benefit.

This is an independent reconstruction of the REALM 2025 Dec-LoRA method on a
new quantity partition, not a direct numerical reproduction or a win over its
printed score. The study uses the official labeled validation set, best-round
selection, and a single-process peer simulation. It includes no pooled
RoBERTa comparator, hidden-test evaluation, independent-client memory
measurement, real multi-machine deployment, or privacy guarantee. Static sample
weights are distinct from the historical dynamic domain weights. The paper
acknowledges DeCAF's related effective-product aggregation.

The historical nine-arm, three-seed pooled-versus-peer experiment and its completed
failure investigation are preserved. The original adaptive-rank, domain-weighted
state-averaging protocol fails to match pooled CIFAR-100 accuracy. Controls examine
ownership, SVD, retained information, and optimizer ordering; exact synchronized
gradients reproduce the full-data pooled prediction path with full-rank state.

The completed validation-only studies are also included. All eight retained-base
residual recipes fail their one-seed screen. Shared-model partial gradients give
55.03 +/- 1.01% validation accuracy under adaptive/domain allocation versus
57.18 +/- 0.71% for conventional pooled LoRA, but retain full rank-16 factors,
forward computation and Adam state. These results do not establish the original
whole-model resource claim. The masked uniform control matches all three final
pooled prediction vectors and 89/90 intermediate correct counts; the earlier
exact full-data gradient control matches predictions at all 90 evaluations.

The new study's prospective protocol, full per-run records, source snapshots,
checkpoint audit bindings, final statistics, and figure data/provenance are in
[`../docs/artifacts/quantity-skew/`](../docs/artifacts/quantity-skew/). Its
independent final artifact audit verifies 36 runs, 108 current checkpoint files,
and 720 saved round prediction records. That artifact audit does not itself
rerun inference; the earlier separate checkpoint evaluators did so. Raw wire
buffers and per-run initial-state hashes were not retained, as disclosed in the
paper. Large checkpoint binaries remain remotely archived and hash-addressed
in the repository records.

Historical comparison data, transport ledgers, and launch sources are in
`../docs/artifacts/integrated-corrected/`. Follow-up plans, failed candidates,
checkpoint verifications, paired statistics and source snapshots are in
`../docs/artifacts/methodology-exploration/`. The validation candidates use only
a fixed 45,000/5,000 split of the original training data; their official test
remains unopened. The new study does not overwrite that negative result or
establish the original pooled-parity and complete-model resource claim.

Compile with:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The verified build on gpu003 uses the installed TeX Live 2026 distribution:

```bash
cd paper
export PATH="$HOME/ahlora-tools/texlive-202609/.TinyTeX/bin/x86_64-linux:$PATH"
pdflatex -halt-on-error -interaction=nonstopmode main.tex
bibtex main
pdflatex -halt-on-error -interaction=nonstopmode main.tex
pdflatex -halt-on-error -interaction=nonstopmode main.tex
pdftoppm -r 90 -png main.pdf rendered/main
```

Create the rendering output directory first. The manuscript uses scalable Latin
Modern fonts. All scientific execution and PDF
compilation/rendering for this delivery use gpu003. Inspect every final rendered
page after a substantive manuscript change.

The generated `main.pdf` is the rendered paper artifact. Numerical source data
are in `../docs/artifacts/`, with `claude_handoff.json` as the index.
The figures included by the manuscript and their source contexts are documented
in [`figures/README.md`](figures/README.md).
