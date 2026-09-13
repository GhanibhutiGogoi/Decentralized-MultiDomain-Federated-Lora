# P3 automatic discovery artifacts

This directory contains the completed, non-smoke run of `experiments/05_signature_validation.py --methods adaptive` on the documented GPU host.

Required files:

- `manifest.json`: command, source commit, environment, data/cache hashes, seeds, stages and protocol parameters.
- `adaptive_summary.csv`: one row per seed and discovery stage, including inferred `n_clusters`, confidence, ARI/NMI (scoring only), personalized/consensus accuracy, and payload counts.
- `summary.csv`: the experiment's signature-comparison table.
- `seed{seed}_adaptive.json`: complete per-round records and discovery snapshots.
- `discovery_metrics.png`: discovery confidence, scoring ARI, and inferred cluster count by stage.

The protocol observes adapter signatures through the runner's stateful mixer. If all client states are available in one process, describe this as a centralized signature-observation view in the manifest; do not call it fully neighborhood-local discovery without a corresponding implementation.
