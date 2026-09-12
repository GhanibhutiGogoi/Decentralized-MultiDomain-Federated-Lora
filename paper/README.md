# AH-LoRA paper

`main.tex` is a completed, evidence-scoped technical paper for the current
repository milestone. It reports the remote V100 benchmark, the heterogeneous
rank and error-feedback comparisons, offline signature validation, and the
real-data Project 2 calibration. It explicitly labels oracle-domain inputs,
simulated communication counts, weak signature discovery, and unresolved
adaptive-rank calibration.

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
