#!/usr/bin/env bash
# Build standalone manuscripts on the designated experiment machine only.
set -euo pipefail
if [[ "$(hostname -s)" != gpu003 ]]; then
  echo 'Build and render these papers on gpu003.' >&2
  exit 1
fi
paper_task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tex_bin="$HOME/ahlora-tools/texlive-202609/.TinyTeX/bin/x86_64-linux"
export PATH="$tex_bin:$PATH"
for paper_study in multidomain quantity-skew; do
  (
    cd "$paper_task_root/paper/$paper_study"
    mkdir -p build rendered
    pdflatex -halt-on-error -interaction=nonstopmode main.tex > build/pass1.log
    bibtex main > build/bibtex.log
    pdflatex -halt-on-error -interaction=nonstopmode main.tex > build/pass2.log
    pdflatex -halt-on-error -interaction=nonstopmode main.tex > build/pass3.log
    pdftoppm -r 100 -png main.pdf rendered/main
    pdftotext -layout main.pdf build/main.txt
    pdfinfo main.pdf > build/pdfinfo.txt
    sha256sum main.pdf > build/main.pdf.sha256
    printf '%s\n' "Built paper/$paper_study/main.pdf"
  )
done
