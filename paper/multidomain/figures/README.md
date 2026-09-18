# Figures for the standalone multi-domain study

Only figures actually included by `../main.tex` are copied here. All were
generated on gpu003 from archived scientific records; no values were inferred
from images. The files are local to this paper so it can be compiled alone.
Artifact paths below are relative to the repository root.

- `integrated_final_comparison`, `integrated_accuracy_curves`,
  `integrated_factorial`, `integrated_communication`, and
  `integrated_rank_trajectories` report all 27 corrected runs in
  `docs/artifacts/integrated-corrected/`. The generating script is
  `scripts/summarize_integrated.py`. Error bars are sample SD across seeds
  42–44; communication includes control traffic and one final peer assembly.
- `exploration_optimization` reports seven paired three-seed controls using
  the original 50,000 training and 10,000 official-test examples.
- `exploration_rank_retention` reports the no-training projection/mixing
  stress test from completed pooled checkpoints, including initial projection.
- `exploration_gradient_equivalence` reports the full-data synchronized-gradient
  control, with matching predictions at all 90 epoch evaluations.
- `exploration_residual` preserves all eleven seed-42 validation endpoints,
  including eight unsuccessful residual recipes. It has no between-seed
  uncertainty bars.
- `exploration_masked` reports the four paired three-seed partial-gradient
  and pooled arms. Error bars are sample SD. These smaller gradient-coordinate
  budgets do not reduce full rank-16 model or optimizer storage.

The five exploration figures are backed by
`docs/artifacts/methodology-exploration/`. The residual and masked studies use
the same 45,000/5,000 fitting/validation split of the original training data;
their official test remains unopened. Their uniform driver matches all three
final pooled prediction vectors and 89/90 intermediate correct counts,
distinct from the full-data synchronized control above.

The appendix keeps four separately qualified component figures:

- `p3_uniform_curves.png`: historical uniform rank-16, 50-round protocol
  comparison, seeds 42–44, source context `ded87b4e7797`.
- `p3_heterogeneous_curves_readable.pdf`: historical ranks 4/12/32 and
  50-round comparison, source context `b729ad24b6b0`, redrawn from exact
  archived records with a separate readable legend.
- `p1_adaptive_final_accuracy_readable.pdf`: exact five-task final endpoints
  after five rounds at seed 42 from
  `docs/artifacts/p1-adaptive-rank/federated_lora_summary.csv`.
  Intermediate round arrays were not archived in that CSV and are not
  reconstructed. The fixed rank-32 comparator exceeds the tested ceilings.
- `p3_adaptive_discovery.png`: coordinator-visible online-discovery results;
  group recovery is not evidence for a neighborhood-local private protocol.
