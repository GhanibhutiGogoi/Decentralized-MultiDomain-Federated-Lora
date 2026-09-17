# Quantity-skew decentralized LoRA comparison

This is the new RoBERTa/SST-2 study against an independent reimplementation of published Dec-LoRA. It is separate from the completed CIFAR-100 disjoint-domain diagnostics in `../methodology-exploration/`.

The prospective protocol and all reconstruction assumptions are in [the protocol](../../research/2026-09-17-quantity-skew-protocol.md); [the primary-source literature audit](../../research/2026-09-17-paper-baseline-shortlist.md) explains paper selection. Model/dataset revisions, run source snapshots, exact client memberships, actual serialized graph transfers, adaptive-rank probes, predictions, checkpoints and metrics are recorded by the runner.

## Current evidence

- Remote numerical/software checks: 119 tests passed on gpu003.
- Seven matched real-model smoke runs: completed, with four rounds, two local steps and a 640-example training subset. All seven best/final checkpoint pairs reproduced exactly using the independent evaluator.
- Smoke results establish execution and auditability only. They do not establish scientific efficacy, equivalence or a win over Dec-LoRA.
- Full-data comparison: launched under the prospective protocol. Completed full-budget, replicated results must replace this status before any scientific success claim.

The `smoke/` folders preserve the actual smoke source snapshots and records. They precede final hardening/provenance additions in source commit `8ae7c2e`; full-budget runs archive their own source and must not be pooled with smoke results. Small saved models remain on gpu003; checkpoint SHA256 values and verification records identify them. Source assets are public, pinned Hugging Face RoBERTa/GLUE revisions; the large foundation-model weights are not committed.

The training runtime is a single-process peer-transport simulation on gpu003. It is not a multi-machine deployment and provides no cryptographic privacy guarantee. The final adapter is assembled at a preselected rank16-capable peer; weaker peers do not install it. Whole-process peak GPU memory is not a per-client memory measurement.

## Reproduce and interpret

Run commands and comparisons must use the recorded config and source snapshot. The main quantity experiment fixes 211 local updates of exactly32 examples per peer per round, for20 rounds. Sample-size weighting is static `n_i/N`. The equal-size, one-local-epoch Dec-LoRA anchor is a separate reconstruction setting. The seven quantity arms include uniform rank16 and rank4 factor-gossip baselines, an unconstrained product-space control, and the fixed/adaptive by uniform/sample-weighted factorial.

The primary endpoint follows the paper's best labeled-validation accuracy; final accuracy and resource costs are secondary. Canonical GLUE counts and several unspecified paper hyperparameters are disclosed assumptions. Our changed-partition scores cannot be compared directly with the publication's printed93.81%. No hidden test result or performance-parity conclusion is implied.
