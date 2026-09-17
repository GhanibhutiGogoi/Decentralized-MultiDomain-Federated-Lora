# Independent source review

The separate `comparison_audit` agent reviewed both experiment drivers
read-only after the numerical runs began. This review is distinct from the
separate saved-checkpoint forward evaluator and its remote execution log.

For `diagnose_gradient_equivalence.py`, the reviewer found no training or
allreduce correctness issue for the supplied ring, counts, and settings:

- Local gradients are materialized copies before the next `zero_grad`.
- Aggregation weights are the current minibatch sample counts.
- A single shared Adam step occurs after aggregation, preserving the intended
  common optimizer semantics.
- The same-state float64 oracle separates gradient arithmetic identity from
  float32 trajectory roundoff.

The review emphasized the globally coordinated batch schedule, excluded
setup/schedule costs, collapsed optimizer replicas, and runtime encompassing
both arms plus proofs. These limitations are explicit in the README. The
initial driver-only provenance was supplemented with a 30-file dependency
archive captured from the untouched isolated snapshot; capture timing is
explicit. The generic driver does not reject duplicate seed arguments, but
the saved launch uses the unique seeds 42, 43, and 44.

For `diagnose_adam_aggregation.py`, the reviewer found no correctness issue:

- Every batch and arm deep-copies the unchanged initialization and starts
  fresh Adam moments.
- Local effective updates are weighted by actual minibatch ownership counts.
- There is no factor averaging, SVD, or test-data access.
- Initial B is zero, so the post-step effective delta is also its change
  from initialization, which justifies the reported update comparisons.
- Ownership is reconstructed with the original partition function and checked
  against saved owner hashes.
- Concatenated raw-gradient copies are taken before optimizer updates.

The one-step optimizer-order result is an arithmetic mechanism, not an
attribution of the complete multi-round accuracy gap.
