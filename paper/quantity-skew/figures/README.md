# Standalone quantity-study figures

These are unchanged copies of the final gpu003-generated quantity-study figures, included locally so the manuscript has no dependency on another paper folder. Each stem has PDF, PNG, and exact plotted-data JSON files:

| Stem | Manuscript content |
|---|---|
| `quantity-skew-accuracy` | Best and final official-validation accuracy for all seven methods, individual seeds, means, and sample SD |
| `quantity-skew-paired-effects` | Paired comparator and factorial effects with unadjusted Student-t 95% intervals |
| `quantity-skew-trajectories` | Validation learning curves and persistent training ranks over 20 rounds |
| `quantity-skew-tradeoff` | Validation accuracy versus complete serialized training traffic, including classifier and metadata |

The data are the 35 matched quantity runs at seeds 42–46. The single equal-size anchor and seven smoke runs are excluded from these five-seed figures. Rank-16 references exceeding weaker peers' caps are marked. Best-round validation selection is not hidden-test evaluation, and an interval crossing zero does not establish parity.

Scientific generation and statistics were executed on gpu003 by [`scripts/plot_quantity_paper.py`](../../../scripts/plot_quantity_paper.py). The complete input/output hashes, source identity, CSVs, and statistics are archived in [`docs/artifacts/quantity-skew/final-figures/`](../../../docs/artifacts/quantity-skew/final-figures/). No numbers or graphical data were recomputed for this manuscript split.
