# Independent readiness audit: quantity-skew decentralized LoRA

Audit date: 2026-09-17. Audited implementation commit: `8ae7c2e`.
Reviewer: a separate agent from the experiment operator. This review read the
protocol, implementation, archived verification records, and historical paper;
it did not rerun numerical tests, training, or evaluation. Scientific execution
and verification records identify the authorized remote host, `gpu003`.

**Decision:** no implementation or scientific-reporting blocker was found for
continuing the declared full-budget pilot. This is a readiness decision, not a
finding that the proposed method works, matches another method, or provides
privacy. Replicated full-budget results and their checkpoint audits remain
required before an efficacy conclusion.

## Scope and evidence

The review covers the [prospective protocol](2026-09-17-quantity-skew-protocol.md),
[paper/source shortlist](2026-09-17-paper-baseline-shortlist.md),
`experiments/quantity_skew_benchmark.py`, `experiments/verify_quantity_checkpoint.py`,
and the quantity partition, peer transport, compact merge, and RoBERTa LoRA
helpers under `project-3-hierarchical-gossip/`. It also reviews the completed
historical paper and the delivery claims of
[PR 46](https://github.com/GhanibhutiGogoi/Decentralized-MultiDomain-Federated-Lora/pull/46).

The archived remote regression log reports **119 passing tests**. All seven
smoke directories contain an independent audit with `status: passed`, host
`gpu003`, and exact prediction matches for both best and final checkpoints:
14 saved-checkpoint evaluations in total. These records were inspected, not
independently executed again by this reviewer. The smoke source snapshot
precedes the final hardening/provenance additions in `8ae7c2e`; its evidence is
not silently presented as a full experiment using the later snapshot.

The separate evaluator reads canonical parquet assets, validates source and
asset hashes, reconstructs source-row mappings and shard membership, checks
quantity-only class quotas and logged metrics, then predicts directly from
saved best/final states. It does not import the training runner or its metric
implementation. It shares the Hugging Face architecture and LoRA state-install
helper, so it is independent at the data/metric/driver level rather than an
entirely independent model implementation.

## Protocol-to-implementation findings

| Question | Audit finding |
|---|---|
| Is the comparator identifiable? | Yes. REALM 2025 Dec-LoRA is the primary paper, with its publication version distinguished from later arXiv revisions. The baseline is explicitly an independent reconstruction because no author implementation was verified. |
| Is this the paper's exact numerical experiment? | No, and the protocol says so. Canonical SST-2 split counts differ from the paper's table; optimizer, alpha, target modules, head handling, length, and local-step interpretation require disclosed choices. The quantity partition is a new controlled extension. |
| Does the partition isolate quantities rather than domains? | Yes. One SST-2 task is split into disjoint, exhaustive shards with exact integer client/class margins and proportional class quotas. Quantity and rank-capacity assignments use separate recorded permutations. |
| Are local optimization budgets comparable? | Yes for the seven quantity arms: 211 full batches of 32 examples per peer per round, for 20 rounds. Batches stream only from that peer's shuffled shard across epoch boundaries. Smaller shards are revisited more often. Adaptive probes are additional work and logged separately. |
| Are weights and updates mathematically consistent? | Yes within the declared protocol. Sample-weighted Metropolis mixing preserves `n_i/N`; uniform mixing preserves uniform weights. Effective-update merging includes alpha/r scaling, and classifier parameters use the same mixing weights. Finite local Adam and truncation are not claimed to equal pooled optimization. |
| Are adaptive-rank choices evaluation-free? | Yes. The demand probe and quality feedback use a fixed local training batch. No validation labels choose ranks. The probe expands to the capability ceiling, which limits memory-saving claims. |
| Are graph contributions actually represented as transfers? | For model contributions, yes: payloads are serialized, hashed, decoded, and routed on declared neighbor edges. Training is synchronous and uses the previous peer-state collection. Setup count bytes have the separate accounting qualification below. |
| Is assembly compatible with weak clients' caps? | Yes under the documented endpoint. A preselected capacity-16 peer gathers records along a spanning tree and holds the assembled rank-16 adapter. No assembled model is broadcast to weaker peers. Rank-4 factor gossip deploys rank 4. |
| Does evaluation change the training trajectory? | No. Evaluation assemblies are separate from the peer states used by the next training round. The last evaluated gather is counted once as the final delivery. |
| Are resource measurements scoped honestly? | Generally yes. The complete classifier is communicated; gather-root storage and merge workspace are separated from whole-process CUDA peaks. The process shares a backbone and retains multiple simulated peer states. These are not isolated-device client peaks or real network timing. |
| Is the endpoint fixed? | Yes. Best official labeled-validation accuracy is primary, final accuracy secondary. First occurrence wins ties. This follows the paper's selection endpoint but is not an unseen-test result. |
| Is the experiment reproducible from provenance? | Implementation, evaluator, model/data hashes, memberships, graph, random seeds, configuration and source manifest are recorded. Pinning the prose protocol alongside the launch records needs the small addition below. |

The seven-arm design supplies an unconstrained rank-16 literature comparator,
a feasible uniform rank-4 factor-gossip comparator, an unconstrained
product-space control, and a fixed/adaptive by uniform/sample-weighted
factorial. It can answer the registered comparative questions. It cannot
establish that product-space averaging is novel: that mechanism is related to
existing DeCAF work. Static dataset counts give static sample-size weights,
not dynamic domain allocation. Changes in stationary weights also change
mixing speed, so the weighting contrast does not identify an objective-only
effect.

## Nonblocking reporting corrections

These issues do not change training, saved predictions, or the declared
pilot. They should be reflected in the final artifact and paper without
modifying a running source snapshot.

1. **Count setup is modeled.** The runner records 20 directed count messages
   at 16 bytes each, for 320 bytes. Those records are not passed through the
   tensor payload encoder/decoder. Describe them as a fixed-width modeled
   setup allowance; reserve “actual serialized bytes” for model-transfer
   payloads whose wire lengths are measured.
2. **Production communication starts after provisioning.** Its sum includes
   count setup, training messages, and one final assembly. It excludes
   distributing the common base model/initial state, establishing topology,
   and network framing. The transport ledger already discloses exclusions;
   headline costs should retain this scope rather than imply total deployment
   traffic.
3. **The RoBERTa head is not a single linear layer.** The implementation
   averages the complete trainable classifier parameters, including the dense
   projection and output layer. Replace the protocol phrase “linear
   classification head” with “classifier parameters.” The implementation and
   its parameter accounting are consistent.
4. **Pin the prose protocol alongside code.** `archive_source` includes the
   runner, verifier and imported implementation sources, but does not include
   the Markdown protocol. Preserve the protocol's launch commit/hash or a
   copy with the run manifest, and identify any later clarifications as such.
   Do not relabel a later document as the exact pre-run document.

## Historical study and PR 46

No material unsupported conclusion was found in the reviewed historical
paper or PR description. The completed diagnostics retain the original
negative result and distinguish several narrower findings:

- Exact synchronized, uniform-rank peer gradients reproduce the pooled
  training reference under common full-rank state and common optimizer
  assumptions. This does not solve heterogeneous complete-model rank limits.
- Repeated heterogeneous projection destroys predictive information in a
  no-training oracle-retention intervention. This is a concrete mechanism in
  that setting, not proof that every decentralized LoRA protocol fails.
- All eight retained-base residual candidates fail their seed-42 validation
  screen. Severe finite loss growth is distinguished from NaNs or a theorem
  of divergence.
- Partial-gradient variants narrow the validation gap while retaining the
  full rank-16 model and Adam state. The adaptive variant underperforms the
  corresponding fixed variant; the paper does not claim accuracy parity,
  complete-model memory savings, or privacy from those results.
- The 89/90 intermediate correct-count agreement of the later masked driver
  is distinguished from the exact 90/90 prediction agreement of the earlier
  full-data synchronized-gradient experiment.

The historical delivery record reports a rebuilt, inspected 29-page paper,
470 remote diagnostic regression tests, and independent reconstruction of
25 unique saved residual/masked validation models. Those are completed
historical-study claims, not evidence that the new transformer comparison is
complete. PR 46 was open with no submitted reviews at the time of this audit.

## Remaining gates for the new study

Complete and retain the registered seed-42 full-budget screen and equal-size
anchor, disclose runtime failures or code changes, then finish the fixed
paired seed set under an unchanged or explicitly versioned protocol. Verify
best/final checkpoints independently and check cross-arm partitions, sample
streams, budgets, source identities, capacity/rank traces and byte ledgers.
Report all declared arms and both accuracy/resource endpoints, with paired
variation and uncertainty. Preserve the original negative study separately.

The current evidence supports executing and auditing this comparison. It
does not yet support “beats Dec-LoRA,” “matches conventional training,” a
noninferiority claim, or a hospital privacy claim. A complete negative result
would still answer the prospective research question.
