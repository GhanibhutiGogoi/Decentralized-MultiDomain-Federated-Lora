# AH-LoRA paper

`main.tex` reports the corrected nine-arm, three-seed pooled-versus-peer experiment,
with the mathematical definitions integrated into the methodology. It states
the negative accuracy-preservation result and separates supporting historical
experiments from the invalid, superseded pilot. Source data, peer-transport
ledgers, resource costs, validation, and the exact launch source archive are in
`../docs/artifacts/integrated-corrected/`. No privacy guarantee is claimed.

Compile with:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The generated `main.pdf` is the rendered paper artifact. Numerical source data
are in `../docs/artifacts/`, with `claude_handoff.json` as the index.
The figures included by the manuscript and their source contexts are documented
in [`figures/README.md`](figures/README.md).
