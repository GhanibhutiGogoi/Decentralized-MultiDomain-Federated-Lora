# Claude artifact handoff

Start with [`claude_handoff.json`](claude_handoff.json). It records the intended completion scope, which evidence is current, where to find the measured data, and which claims the data can support. [`progress.jsonl`](progress.jsonl) is an append-only work log; each line is one JSON object. New experiments should be added to the handoff only after their files exist.

The deliverable is a reproducible CIFAR-100 benchmark of decentralized LoRA with known domain groups. The benchmark is not evidence that clients can discover their domains, that adaptive ranks are calibrated, or that Project 2's old lambda estimates are valid. Those are separate extensions. A paper is optional.

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

The P3 signature report is [`p3-signatures/gate_g1.json`](p3-signatures/gate_g1.json). Stage-10 mean ARI was 0.127 (local spectral), 0.067 (local row norms), 0.274 (local inverse ΔW L2), −0.060 (MH spectral), 0.161 (MH row norms), and 0.110 (MH inverse ΔW L2). All are below the 0.4 soft-candidate screen, so this run does not support automatic domain discovery for the fc-only adapters.

The regenerated P2 artifacts are [`p2-exp1-real-seed42`](p2-exp1-real-seed42) and [`p2-exp2-real-seed42`](p2-exp2-real-seed42). All five datasets are marked real in `dataset_manifest.json`. Form A uses gamma 2.44458 and has global Spearman 0.368, pairwise ranking accuracy 0.641 and permutation p 0.162. Form B uses gamma 5 and has global Spearman 0.369, pairwise ranking accuracy 0.630 and permutation p 0.207. Form B's selected ridge alpha is 1000. These seed-42, five-round results show weak and task-dependent ranking behavior; they do not establish a universal preferred form or cross-task generalization.
