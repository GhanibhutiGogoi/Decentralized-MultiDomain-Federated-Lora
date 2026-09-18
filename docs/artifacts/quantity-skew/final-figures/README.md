# Final quantity-skew figures

Generated on gpu003 from 35 complete quantity-extension runs: seven methods × seeds 42–46. The seven smoke runs and single equal-size anchor are excluded from all figures. All 700 round accuracies were independently recomputed from predictions and labeled validation targets; 70 best/final checkpoint audit bindings and artifact hashes were rechecked.

Four figures: accuracy (primary best / secondary final), paired effects (unadjusted 95% t intervals), accuracy/serialized-training-traffic tradeoff plus classifier/adapter decomposition, and learning/rank trajectories. Each PDF/PNG pair has a matching `.data.json`; raw derived numbers are in CSV and statistics.json. Exact source and artifact hashes are in provenance.json.

All outcomes are official labeled SST-2 validation, not hidden-test generalization. Best accuracy selects the best of 20 rounds. Intervals are exploratory seed-paired t intervals with n = 5, df = 4; they are unadjusted for multiple contrasts and do not establish parity. Dec-LoRA is our independent paper-based reimplementation, not author code. Uniform rank-16 references exceed weaker client caps. Rank 4 is feasible and must remain visible: our method uses 8.92% more training bytes than rank 4 and has lower mean accuracy.

Training traffic includes actual serialized LoRA factors, classifier tensors, and metadata; setup and evaluation/final assembly are excluded from this axis and separately logged. The classifier traffic is substantial and shared across arms. Rank trajectories measure persistent training rank, not total or peak client memory; adaptive probes run at the original capacity ceiling. This is a single-process explicit peer-payload simulation without formal privacy protection.

Reproduce with `scripts/plot_quantity_paper.py --summary <campaign-summary/summary.json> --output <final-figures>` on gpu003.
