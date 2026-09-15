# Paper figures

These figures are the authoritative plots used by `../main.tex`.

The corrected comparison uses `integrated_accuracy_curves`, `integrated_final_comparison`, `integrated_factorial`, `integrated_communication`, and `integrated_rank_trajectories`, each in PDF and PNG. They are generated on gpu003 by `scripts/summarize_integrated.py` from all 27 raw records in `docs/artifacts/integrated-corrected/`. `summary.json` records input, script, and figure hashes. Error bars are sample standard deviations over seeds 42--44; the communication plot includes control traffic and one final assembly. The following figures retain their historical protocols.

- `p3_uniform_curves.png` and `p3_uniform_domains.png` come from the completed
  remote report for context `ded87b4e7797` (rank 16, 50 rounds, seeds 42--44).
- `p3_heterogeneous_curves.png` and `p3_heterogeneous_domains.png` come from
  context `b729ad24b6b0` (ranks 4/12/32, 50 rounds, seeds 42--44), including
  the delta-W, factor-zero-padding, and feedback arms where applicable.
- `p3_signature_ari.png` is generated from
  `docs/artifacts/p3-signatures/summary.csv`, aggregating mean and sample SD by
  training stage, method, and signature.
- `p2_lambda_validation.png` is generated from the regenerated
  `docs/artifacts/p2-exp2-real-seed42/evaluation_metrics.csv` global rows.

The underlying aggregate CSV/JSON files and manifests are retained under
`docs/artifacts/` so the plotted values can be audited.
