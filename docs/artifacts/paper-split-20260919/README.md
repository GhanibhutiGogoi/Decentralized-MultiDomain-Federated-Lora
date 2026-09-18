# Two independent research papers — 2026-09-19

The user requested separate papers for the original experiment and the new research experiment. Both standalone manuscripts are complete. This change separates the scientific presentation; it does not rerun experiments, alter recorded data, or change either study's conclusion.

| Study | Current deliverable | Scope |
|---|---|---|
| Original multi-domain study | [30-page PDF](../../../paper/multidomain/main.pdf), [source](../../../paper/multidomain/main.tex), 14 figures | Corrected nine-arm CIFAR-100 comparison, completed failure analysis, and separately qualified supporting component studies. The tested adaptive/domain method does not preserve pooled accuracy. |
| New quantity-skew study | [14-page PDF](../../../paper/quantity-skew/main.pdf), [source](../../../paper/quantity-skew/main.tex), 4 figures | RoBERTa/SST-2, seven decentralized methods at five seeds and a separate reconstruction anchor. Reduced traffic versus rank 16 does not establish parity; feasible rank 4 has higher mean accuracy and lower traffic. |

Each folder has its own methods and equations, bibliography, figures, and conclusion. Neither manuscript imports the other study's results or requires the other folder to compile. The new study uses static sample-size weights, not the original dynamic domain allocation. Its best labeled-validation endpoint is separate from the original official-test endpoint. Neither establishes privacy guarantees.

## Verification and provenance

- [Scientific review](SCIENTIFIC_REVIEW.md): an independent agent checked both manuscripts against source and evidence. A second reviewer checked every reported new-study table, contrast, interval, and resource figure against the completed five-seed artifacts.
- [Visual review](VISUAL_REVIEW.md): every page passed after rendering on gpu003. The final original-paper rebuild exactly matches all 30 previously inspected page images.
- [Build record](build-record.json): source/PDF/asset hashes, all 44 rendered-page hashes, page counts, confined manuscript dependencies, archive checks, and figure provenance comparisons.
- [Delivery verification](delivery-verification.json): all 33 delivered source, PDF, and figure files match the remote build identities; both current machine-readable handoff records have matching paths, page counts, and PDF hashes.
- The `multidomain/` and `quantity-skew/` subfolders contain final compiler passes, bibliography logs, PDF metadata and hashes. Both final logs have zero layout/reference warnings and no unresolved references.
- [Audit source](audit_split.py): document-validation checks executed on gpu003. It verifies 22 historical archive files, all 26 copied scientific figure assets, and both independent source sets. It does not recompute experimental statistics or rerun inference.

The scripts and checks use the designated SSH machine, gpu003. Rebuild with `bash scripts/build_research_papers.sh`, then run `python docs/artifacts/paper-split-20260919/audit_split.py` there. Rebuilding changes PDF metadata; new hashes should be recorded and any changed pages reviewed before replacing delivered files.

The previous 38-page combined manuscript is preserved in [the archive](../../../paper/archive/combined-20260918/). Its PDF remains byte-identical to the 2026-09-18 delivery. Existing hashed research manifests and append-only logs are not rewritten; old paths in those records describe their dated versions. The root paper index and both Claude handoffs now point to the matching standalone manuscript.
