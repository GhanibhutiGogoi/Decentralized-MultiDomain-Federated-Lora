# Completed paper build and review

`paper/main.pdf` was compiled and rendered on gpu003 using TeX Live 2026 and
Poppler. The final manuscript has **38 pages**. All pages were visually reviewed,
with independent-agent coverage of scientific claims and 16 rendered pages.
There are no final LaTeX layout/reference warnings or unresolved-reference
markers. `build-record.json` records every included figure/source hash, the PDF
hash, rendered-page hashes, exact unchanged-page comparisons, and review coverage.

Final PDF SHA256:
`8a0816eadf36888c930bd448c6ebe3f938cf314d503809f87429a83f0f3cbd5b`

The manuscript integrates the completed five-seed transformer comparison while
preserving earlier negative results. Equations are part of the methodology.
The four new quantity-study figures were checked against all 35 matched runs;
font changes preserved numerical statistics byte for byte. Two historical
appendix figures were regenerated for readability from retained data, with
separate scripts, exact numeric sidecars and provenance in `historical-figure-qa/`.
The original historical figure files are retained.

The final log and bibliography-build log are preserved beside this record.
The full remote build and rendered pages remain under
`~/ahlora-quantity-20260917/paper-final-20260918/paper/` on gpu003. Rendered PNGs
are also available locally under `/tmp/dmfl-paper-final-20260918/rendered-final/`;
only the final PDF, scientific figure files and compact provenance are committed.

To rebuild, use the commands in `paper/README.md` on gpu003. After source or
figure changes, rerender and inspect the affected pages; the hashes in this
record identify this delivery, not later builds.
