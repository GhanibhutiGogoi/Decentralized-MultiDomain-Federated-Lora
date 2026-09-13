# Automatic Discovery and Adaptive-Rank Completion Protocol

This note defines the completion experiment for the two open claims in AH-LoRA.
It is deliberately operational: every quantity below must be logged by the
runner, and every comparison uses the same partitions, initialisation and seed.

## Automatic domain discovery

After each local training phase, each client emits a compact signature of its
scaled effective update, `Delta W = (alpha / r) B A`. The shipped online
discovery path uses a fixed seeded random projection of the flattened update;
the projection dimension is 64. Row-norm signatures remain an offline
diagnostic in Experiment 05 and are not part of the online payload.
Only signatures are exchanged; dense `Delta W` matrices are never transmitted
for discovery. The receiver maintains an EMA (`beta=0.9`) per client, computes
cosine affinities on the EMA signatures, symmetrises them, and applies
Sinkhorn plus a self-weight floor to obtain a symmetric doubly-stochastic soft
mixing matrix.

The cluster count is unknown to the protocol. At discovery rounds, evaluate
average-linkage partitions for `k=2..min(8,N-1)` and select the largest
silhouette score. If every score is non-positive, use `k=1` and retain soft
affinity mixing. The current completion implementation deliberately deploys
the soft matrix only; hard two-tier gating and neighborhood-local observation
are follow-up work.

The full-data experiment compares local-only, flat MH, automatic soft, automatic
hard, and oracle hierarchy. It uses CIFAR-100 with five superclass domains,
three clients per domain, the frozen ResNet-18 feature cache, 50 rounds, and
seeds 42/43/44. Report personalized and consensus accuracy, worst-domain
accuracy, fairness gap, participation, cumulative factor/signature floats,
rounds to target, inferred `k`, silhouette, assignment stability, ARI/NMI
(labels are consulted only for scoring), false-peer rate, and the spectral gap
of each realised mixing matrix. The automatic protocol is considered useful
if soft discovery reaches ARI >= 0.40 by round 10 and its final personalized
accuracy is within two percentage points of the oracle at no more than 15%
additional payload. Otherwise report the soft matrix as the robust fallback
and retain the negative discovery result as a limitation.

## Adaptive rank calibration

Before each local update, a client probes LoRA gradients and computes singular
values of the gradient matrices. For candidate rank `r`, define retained
energy `q_r = sum_{j<=r} sigma_j^2 / sum_j sigma_j^2`. The target is the
smallest hardware-valid candidate with `q_r >= tau` (`tau=0.95` by default).
The target is passed through an EMA of the demand signal; rank increases use a
10% margin and one-round patience, while decreases require a 15% margin and
two consecutive probes. An optional residual ratio multiplies demand by
`1 + 0.5 * clip(residual_ratio,0,1)` so truncation error can trigger an
increase. After per-client targets are proposed, a deterministic greedy budget
projection enforces the configured total rank by selecting the largest
retained-energy gain per additional factor float, while never violating a
client's hardware ceiling.

The calibration battery runs fixed ranks, the legacy gamma rule, energy targets
`tau in {0.90,0.95,0.99}`, and an oracle per-client rank sweep on all five real
Project-1 tasks. Use five rounds for the smoke run and the documented full
battery where available, with seeds 42/43/44. Log stable rank, singular-energy
curves, selected rank, rank changes, tail mass, train/validation accuracy,
FLOPs, factor floats and budget residual. A policy is accepted only if it
stays within one percentage point of the oracle accuracy while saving at least
25% FLOPs, produces more than one rank on at least two tasks, and has zero
budget violations. If these conditions fail, the project should state that the
controller is stable and responsive but not calibrated to the current tasks.

## Reproducibility

All runs execute on the documented GPU host with the cached real datasets.
Store command lines, package versions, source hashes, split hashes, and seed
values in `docs/artifacts/`. Never commit credentials or raw datasets.
