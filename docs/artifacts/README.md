# Claude artifact handoff

Start with [`claude_handoff.json`](claude_handoff.json). It records the intended completion scope, which evidence is current, where to find the measured data, and which claims the data can support. [`progress.jsonl`](progress.jsonl) is an append-only work log; each line is one JSON object. New experiments should be added to the handoff only after their files exist.

The deliverable is a reproducible CIFAR-100 benchmark of decentralized LoRA with known domain groups. The repository now also contains a measured coordinator-visible online-discovery extension, a measured adaptive-rank controller, and a conservative domain-weighting iteration. Discovery is supported by full-data evidence. The original adaptive-rank controller failed its accuracy-preservation gate against a fixed rank-32 reference (historical, preserved below); the revised controller reaches final-accuracy parity with a feasible capability-matched baseline on one Fashion-MNIST run while saving 10.0% FLOPs, and broader five-task parity is unproven. Project 2's regenerated lambda estimates remain exploratory, and the conservative domain-weighting sweep improves contribution ranking modestly without an end-to-end no-regression result. See [Scientific conclusion](#scientific-conclusion) for what the evidence does and does not support.

## Reading a benchmark run

The new entry point is `project-3-hierarchical-gossip/experiments/04_protocol_benchmark.py`. Each output directory contains:

| File | Use |
|---|---|
| `manifest.json` | Configuration, completion status, code and environment provenance, feature-cache protocol, evaluation definitions. Read this first. |
| `results.jsonl` | One JSON object per completed method/seed run, including every round. This is the primary chart source. |
| `summary.csv` | One final-round row per method/seed for comparison tables. Keep individual seeds visible. |
| Per-run JSON files | Individual runs, including initial-state and data-split hashes. |

These filenames describe the new driver's output contract; the handoff's `result_sets` array is the authoritative list of output directories that actually exist. The completed artifacts currently include the P3 aggregate report, P3 signature report, and real-data P2 Experiment 1/2 outputs.

Use complete runs with matched configuration for comparisons. Preserve failed or partial runs in the work log, but do not present them as completed evidence. A one-seed or short run is a smoke test. The planned benchmark uses seeds 42, 43 and 44; report seed mean and sample standard deviation without calling three seeds statistical proof. Do not select whichever round or seed looks best. The final round is the primary endpoint; a best-round value must be labeled separately.

## Metric dictionary

Accuracy values are fractions in `[0, 1]`. Multiply by 100 to display percentages. Differences between accuracies become **percentage points**, not percent improvements.

| Metric | Meaning and display rule |
|---|---|
| `personalized_accuracy` | Unweighted mean of each client's accuracy on its own test shard, after communication. Each client counts equally. |
| `personalized_sample_weighted_accuracy` | The same local evaluations weighted by test-shard size. Keep it distinct from the client mean. |
| `consensus_accuracy` | One uniformly merged adapter, refactorized to the configured reference rank, evaluated on the union of client test shards. With all five domains and no test cap, this is the full CIFAR-100 test set. This is a different prediction task from personalization. |
| `per_domain_accuracy` | Sample-weighted personalized accuracy within each known domain. Domain labels describe the data partition; they are not predicted clusters. |
| `per_client_accuracy` | Personalized accuracy for each client ID. Join IDs to the recorded partition and rank assignments. |
| `accuracy_gap` | Best domain accuracy minus worst domain accuracy. Lower is more even; low gap alone does not mean good accuracy. |
| `worst_domain_accuracy` | Minimum domain accuracy. Higher is better. |
| `consensus_distance` | Mean squared Frobenius distance of clients' effective updates from their mean, summed over LoRA layers. Zero means equal updates; it does not imply useful predictions. |
| `mean_tail_mass` / `max_tail_mass` | Relative squared reconstruction error from rank truncation. Dimensionless. This is not the convergence analysis's absolute epsilon squared. |
| `mean_residual_energy` | Mean absolute squared Frobenius reconstruction error, summed over layers. This is the observed per-round counterpart of the analysis's epsilon squared; a finite run does not establish a uniform bound. |
| `effective_messages` / `effective_floats` | Directed nonzero off-diagonal entries in the applied mixing matrix and their factor payload. A float is a scalar, not a byte. Multiply by the recorded dtype size only if converting to bytes. |
| `operational_messages` | Transmissions under the recorded protocol's execution schedule. The two-tier schedule has intra, bridge and intra stages, so this may differ from effective-matrix support. |
| Operational payload fields | Use the manifest's exact payload convention. A dense exact intermediate and a transmitted low-rank factor payload are different accounting models. Never silently combine them into a single bandwidth claim. |

## Suggested interactive explanation

1. Show 15 clients in five labeled domains and a shared frozen ResNet-18 feature extractor. The trainable part is a 100 by 512 classification-head update, represented by low-rank factors.
2. Animate a local training step, expand `(alpha / rank) B A` into a common-shaped update, apply the selected mixing matrix, and refactorize to each client's rank. Show the discarded residual and the optional error-feedback memory.
3. Let readers choose local-only, centralized FedAvg, flat Metropolis-Hastings or oracle two-tier gossip. Explain that centralized FedAvg is a reference baseline and the oracle arm receives the true domain labels.
4. Plot personalized and consensus accuracy side by side, with seed traces and a final-round comparison. Show per-domain results and communication accounting on separate panels.
5. Compare heterogeneous ranks under delta-W merging and factor zero-padding. Present this as a measured comparison, including losses or ties; zero-padding does not fix factor gauge ambiguity.
6. End with the measured outcome and remaining scope. Do not animate automatic domain discovery unless a separate discovery result supports it.

Historical experiments 01–03 used two-round smoke protocols and a different training path. Project 2's existing output manifests mark their data as stale. They can illustrate the project's development, but must not be mixed into current benchmark curves. Authentication details, usernames and machine access instructions do not belong in a shared artifact.

Experiment 05 adds a separate offline signature comparison; see [`EXPERIMENT05.md`](../../project-3-hierarchical-gossip/EXPERIMENT05.md). Its `summary.csv` has one row per seed, training mode, stage and signature, with ARI/NMI against the known domains. Keep this separate from Experiment 04's final-round method table. Negative ARI is valid. A high score suggests that the representation contains domain information; it does not show that a distributed discovery algorithm was run or that predicted groups improve accuracy.

## Current measured evidence

The P3 aggregate report is [`p3-completion-report/aggregate.json`](p3-completion-report/aggregate.json), with chart-ready CSVs beside it. The homogeneous rank-16 context (three seeds, 50 rounds) reached the following final-round means; accuracy values are fractions and the `±` value is sample SD:

| Arm | Personalized accuracy | Consensus accuracy | Worst-domain accuracy | Effective floats | Operational floats |
|---|---:|---:|---:|---:|---:|
| Local ΔW | 0.5793 ± 0.0055 | 0.1423 ± 0.0127 | 0.5058 | 0 | 0 |
| FedAvg ΔW | 0.2320 ± 0.0029 | 0.2320 ± 0.0029 | 0.1377 | 102,816,000 | 45,744,000 |
| Flat MH ΔW | 0.4230 ± 0.0139 | 0.1514 ± 0.0510 | 0.2760 | 14,688,000 | 14,688,000 |
| Oracle hierarchy ΔW | 0.6898 ± 0.0095 | 0.1025 ± 0.0062 | 0.6048 | 32,313,600 | 40,288,000 |

The heterogeneous rank/merge context is a separate comparison. Its notable final personalized/consensus means are: Local ΔW `0.5360 / 0.1450`; FedAvg ΔW `0.0574 / 0.0573`; flat MH ΔW `0.2845 / 0.0683`; oracle hierarchy ΔW `0.4407 / 0.1147`; flat MH factor zero-pad `0.2063 / 0.0775`; and flat MH ΔW with error feedback `0.2663 / 0.1062`. These values describe this fixed rank cycle and do not validate adaptive rank selection. Factor zero-padding is an intentionally naive baseline and its merge error is measured separately.

The historical P3 signature report is [`p3-signatures/gate_g1.json`](p3-signatures/gate_g1.json); its offline local/MH screen remains below the 0.4 soft-candidate threshold. The completed online discovery run is [`p3-adaptive-discovery/`](p3-adaptive-discovery/): the EMA/silhouette mixer inferred K=5 with scoring ARI 1.0 at stage 10 for all three seeds, with stage-20 ARI 1.0/0.918/1.0. This is coordinator-visible full-state observation, not a neighborhood-local protocol.

The regenerated P2 artifacts are [`p2-exp1-real-seed42`](p2-exp1-real-seed42) and [`p2-exp2-real-seed42`](p2-exp2-real-seed42). All five datasets are marked real in `dataset_manifest.json`. Form A uses gamma 2.44458 and has global Spearman 0.368, pairwise ranking accuracy 0.641 and permutation p 0.162. Form B uses gamma 5 and has global Spearman 0.369, pairwise ranking accuracy 0.630 and permutation p 0.207. Form B's selected ridge alpha is 1000. These seed-42, five-round results show weak and task-dependent ranking behavior; they do not establish a universal preferred form or cross-task generalization.

The follow-up conservative domain-weighting sweep is [`p2-domain-weighting`](p2-domain-weighting) (`README.md`, `sweep_summary.json`). Bounded domain-aware factors are computed from standardized domain signals with default blend strength 0.10, clipped to [0.85, 1.15], and normalized under the base sample-times-quality weights. On the 75 recorded real-data client-round observations at seed 42 the mean metrics move from quality-only to conservative-domain as follows: Spearman correlation 0.152174 to 0.195652; pairwise accuracy 0.520000 to 0.546667; weighted contribution 0.575943 to 0.577445. These are model-free contribution-ranking improvements against the measured leave-one-client-out target. They are not an end-to-end training-accuracy result and carry no privacy claim.

### Adaptive-rank controller: historical gate failure and the revised controller

Two adaptive-rank results exist and must not be conflated.

**Historical (preserved, superseded as the operational comparison).** The original shipped controller (`gamma = 0.5`, no warm-up) was run on five real tasks for five rounds at seed 42 against a **fixed rank-32 reference**; see [`p1-adaptive-rank/summary.csv`](p1-adaptive-rank/summary.csv) and `manifest.json`. It selected rank 2 for every client in every round, saved 93.75% FLOPs, made zero capability violations, and failed the preregistered accuracy gate on CIFAR, Fashion and Tabular. Rank 32 exceeds every client's capability maximum (4, 8, 16), so that comparison is a compute reference, not a feasible hardware-matched baseline.

**Revised controller (current).** The controller now has a two-round warm-up at each client's feasible maximum rank, a half-capability minimum rank, and a relative quality-drop safeguard that restores the client to its feasible maximum. It is compared with a **capability-matched fixed baseline** at ranks [4, 8, 16]. On gpu003 with cached Fashion-MNIST data, 5 rounds, seed 42 ([`p1-adaptive-rank/adaptive_vs_matched_fashion.json`](p1-adaptive-rank/adaptive_vs_matched_fashion.json)):

| Quantity | Matched fixed [4, 8, 16] | Revised adaptive |
|---|---:|---:|
| Final accuracy | 82.85% | 82.85% |
| Total FLOPs | 26.64144 B | 23.977296 B |
| FLOP reduction | — | 10.0% |
| Rank history | constant | [4,8,16], [4,8,16], [4,8,16], [2,6,12], [4,6,12] |

The adaptive run dipped at round 4 (70.24% against 83.08%) and recovered by round 5. This is parity with a feasible baseline on one task and one seed; it is not five-task parity, and it does not revisit the fixed rank-32 reference.

## Scientific conclusion

Stated once, so that no summary drifts past the evidence:

- The project's original end-to-end goal (broadly accuracy-preserving adaptive heterogeneous LoRA with a validated decentralized and privacy benefit) is **not yet achieved**.
- Adaptive rank now reaches parity with a feasible capability-matched baseline on **one** Fashion-MNIST experiment (seed 42, five rounds, 10.0% FLOP saving). Broader five-task parity is **unproven**; the historical five-task battery against the infeasible fixed rank-32 reference failed its gate and is preserved as such.
- Conservative domain weighting improves model-free contribution ranking **modestly** (Spearman 0.152 to 0.196, pairwise accuracy 0.520 to 0.547). An end-to-end no-regression result is **unproven**.
- **No privacy guarantee has been demonstrated.** Low-rank adapters and decentralized exchange alone do not establish privacy. Any privacy claim requires separate differential-privacy, membership-inference, reconstruction, or secure-aggregation experiments, none of which has been run.
- The decentralized benchmark results (oracle hierarchy 68.98 ± 0.95% personalized at uniform rank 16 versus 42.30 ± 1.39% flat gossip and 23.20 ± 0.29% centralized FedAvg) are three-seed finite-run measurements under one frozen-feature protocol, with the hierarchy receiving true domain labels. The online-discovery result is coordinator-visible, not neighborhood-local.

### Why isolated gains do not transfer to the full pipeline

The component studies and the P3 decentralized benchmark are different experiments. In P3, each round applies

`ΔW_{i,t+1} = C_{r_{i,t}}(Σ_j W_{ij,t} ΔW^{local}_{j,t})`,

where `W_t` is the gossip matrix and `C_r` is local rank truncation. The current P3 runner uses fixed rank 16 (or the fixed `4/12/32` cycle); it does not invoke the Project 1 adaptive controller. The one-task Fashion-MNIST parity result therefore cannot be read as an adaptive-rank P3 result.

The model conventions differ as well: Project 1/P2 Experiment 1 use the unscaled LoRA update `B_i A_i`, while P3 uses `(α/r_i) B_i A_i` and undoes that scaling during SVD refactorization. Both aggregation paths match their own forward pass; this audit found no missing-scale error in either path. The parameterization and resulting local optimization still differ across studies.

Project 2 evaluates a leave-one-client-out target, `y_i = Δaccuracy_{-i}`, using normalized contribution weights `n_i q_i λ_i`. Its modest Spearman and pairwise improvements are model-free ranking metrics over 75 rows. The current P2 Experiment 1/P3 training path does not pass the conservative `λ_i` factors into the decentralized aggregation loop, so those factors cannot change P3 accuracy. Even after integration, a better scalar contribution ranking is not equivalent to improving the vector of personalized client objectives or the separate consensus objective.

For an end-to-end claim, all factors must be matched: data split, seed, rounds, rank budget, initialization, graph, aggregation operator, and evaluation target. A controlled P3 ablation is now complete. On heterogeneous `(4,12,32)` ranks over 30 rounds and three seeds, the tested loss-adaptive rank policy reached 20.30 ± 2.51% personalized accuracy versus 24.16 ± 1.04% for fixed MH; bounded domain reweighting produced no measurable change. This negative result is specific to the tested policy and protocol.

The [`protocol_composition_audit.json`](protocol_composition_audit.json) artifact records the source hashes and code-location evidence. The composed run is in [`p3-e2e-ablation/`](p3-e2e-ablation/).

## Validation and the required test machine

The adaptive-rank and domain-weighting regression tests pass on gpu003: **27 passed**. Earlier remote counts on the same host: Project 3 full suite 378 passed, Project 2 127 passed with 46 subtests (PR #42), adaptive-rank controller suite 20 passed, discovery/runner/protocol suite 30 passed.

gpu003 (the documented SSH GPU machine, Tesla V100S, Python 3.10.12, torch 2.3.0+cu121) remains the **required** test and experiment machine. Local runs on a development laptop do not replace SSH validation and must not be reported as if they did. Access details are not part of this bundle.

## Pending completion extensions

The automatic-discovery and adaptive-rank drivers are implemented and their measured outputs are tracked separately from the completed P3 battery:

- [`p3-adaptive-discovery/`](p3-adaptive-discovery/) records online signature discovery, realised mixing matrices, and the ARI/NMI scoring view.
- [`p1-adaptive-rank/`](p1-adaptive-rank/) records controller diagnostics, rank histories, budget checks, and fixed-rank/oracle comparisons.

The P3 discovery directory contains the completed manifest, per-seed records, aggregate table, and graph. The P1 directory contains the completed five-task summary, rank histories, diagnostics, manifest, and figures of the **original** controller, whose preregistered gate failed because every client stayed at rank 2 and accuracy dropped on three tasks against the fixed rank-32 reference (historical, preserved). It also contains `adaptive_vs_matched_fashion.json`, the revised controller's capability-matched Fashion-MNIST comparison described above.
