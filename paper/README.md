# AH-LoRA paper

`main.tex` reports the corrected nine-arm, three-seed pooled-versus-peer experiment
and the completed investigation of its failure, with mathematical definitions
integrated into the methodology. It preserves the negative result for the
original adaptive-rank, domain-weighted state-averaging protocol. Subsequent
controls examine ownership, SVD, retained information and optimizer ordering;
exact synchronized gradients reproduce the full-data pooled prediction path.

The completed validation-only studies are also included. All eight retained-base
residual recipes fail their one-seed screen. Shared-model partial gradients give
55.03 +/- 1.01% validation accuracy under adaptive/domain allocation versus
57.18 +/- 0.71% for conventional pooled LoRA, but retain full rank-16 factors,
forward computation and Adam state. These results do not establish the original
whole-model resource claim. The masked uniform control matches all three final
pooled prediction vectors and 89/90 intermediate correct counts; the earlier
exact full-data gradient control matches predictions at all 90 evaluations.

Primary comparison data, transport ledgers and launch sources are in
`../docs/artifacts/integrated-corrected/`. Follow-up plans, failed candidates,
checkpoint verifications, paired statistics and source snapshots are in
`../docs/artifacts/methodology-exploration/`. The validation candidates use only
a fixed 45,000/5,000 split of the original training data; their official test
remains unopened. No privacy guarantee is claimed. The new published-method
quantity-skew benchmark is a separate study and is not presented as a result in
this historical paper.

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
