# Two standalone research papers

The experiments are presented as **two separate papers**. Each folder contains
its own manuscript, bibliography, figures and rendered PDF and can be built
independently. Mathematical definitions are integrated into each methodology.

| Paper | Experiment and conclusion | Files |
|---|---|---|
| Original multi-domain study | CIFAR-100, frozen ResNet-18 features, head-only LoRA; pooled versus decentralized adaptive/domain training and controlled failure analysis. The original accuracy-preservation claim is not met. | [PDF](multidomain/main.pdf), [LaTeX](multidomain/main.tex), [guide](multidomain/README.md) |
| New quantity-skew study | RoBERTa-base/SST-2, seven decentralized methods across five matched seeds, plus one separate reconstruction anchor. Adaptive/sample reduces traffic versus rank 16, but does not establish parity; rank 4 is cheaper and has higher mean accuracy. | [PDF](quantity-skew/main.pdf), [LaTeX](quantity-skew/main.tex), [guide](quantity-skew/README.md) |

These are independent experimental reports. The original paper uses the final
CIFAR-100 test endpoint and retains its distinct training-validation diagnostics.
The new paper uses best official SST-2 labeled-validation accuracy as primary,
final-round accuracy as secondary, and has no pooled transformer comparator.
Results, privacy limitations, protocols and evidence are not pooled between them.

The previous combined 38-page version is preserved in
[`archive/combined-20260918/`](archive/combined-20260918/). It is a historical
archive, not the current manuscript. The original shared figure collection is
also retained for provenance; each current paper uses its own local assets.

## Build and inspect

All compilation, rendering and scientific validation use **gpu003**. From a
checkout with the installed TeX Live 2026 distribution:

```bash
bash scripts/build_research_papers.sh
```

For one paper, run from its own directory:

```bash
export PATH="$HOME/ahlora-tools/texlive-202609/.TinyTeX/bin/x86_64-linux:$PATH"
pdflatex -halt-on-error -interaction=nonstopmode main.tex
bibtex main
pdflatex -halt-on-error -interaction=nonstopmode main.tex
pdflatex -halt-on-error -interaction=nonstopmode main.tex
mkdir -p rendered
pdftoppm -r 100 -png main.pdf rendered/main
```

Build records and page-review coverage are preserved under
[`../docs/artifacts/paper-split-20260919/`](../docs/artifacts/paper-split-20260919/).
The data and existing experiment audits remain unchanged. Current artifact
explainers should link the paper matching the experiment they describe.
