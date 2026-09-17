# Delivery checkpoint, 2026-09-17

The completed optimization, rank-retention, and synchronized-gradient diagnostics
are preserved here with their scripts, data, and independent checks. All
scientific execution used gpu003. The original integrated benchmark and its
negative result are retained unchanged.

The validation-only residual and partial-gradient experiments are complete and
independently summarized. Direct CPU fp32 reconstruction on gpu003 reproduces
every saved final correct count for 25 unique validation models. The archive
includes all failures, per-round records, checkpoints, stdout logs, paired
statistics, source/data hashes and a verified 47-file launch/dependency snapshot.
The original test tensor was not opened during these candidate runs or their
independent verification.

The paper now integrates the residual and masked results, their mathematical
definitions and all five exploration figure pairs. The updated 29-page PDF was
built with TeX Live 2026 on gpu003; its final log has no LaTeX warnings or
overfull/underfull boxes. Every page was rendered with Poppler at 95 dpi and
visually inspected. Direct PDF text checks confirm the new results and the
removed insertion markers. The completed original study is retained: all residual recipes fail;
adaptive partial gradients narrow the validation gap but retain full rank-16
model and optimizer state, so the original whole-model capacity claim remains
unsupported. No privacy guarantee is claimed.

The new user-directed research question is a matched comparison against a
published decentralized method using unequal quantities of one dataset per
client. It will receive a separate plan, source snapshot, and results rather
than replacing or relabeling these historical experiments.
