# Corrected pooled-versus-peer LoRA experiment

This directory holds the corrected experiment that compares a final, deployable rank-16 peer assembly with conventional rank-16 LoRA trained directly on pooled data. The objective is to preserve accuracy, not to improve training accuracy. Consult each raw record and `manifest.json` for completion status; the presence of this README alone does not establish a completed experiment.

The benchmark is limited to **CIFAR-100 classification using frozen ResNet-18 features and a LoRA classification head**. It simulates the peer message flow on gpu003 in one process. It is not a multi-host network deployment, an end-to-end trainable backbone experiment, a language-model benchmark, or a hospital-data evaluation.

## Comparison contract

- The default experiment uses seeds 42, 43 and 44, 30 training rounds, all 50,000 training examples and the full 10,000-example test set. The canonical raw manifest records the actual command configuration.
- Every arm shares the same LoRA alpha, initial effective adapter and frozen head, feature-cache identity, per-seed partitions, optimizer hyperparameters, and training sample exposures. The pooled arm trains on the exact union of the peer training partitions.
- Equal sample exposures do **not** imply equal optimizer steps. Fifteen partitioned clients have separate minibatches and optimizer states. The conventional pooled arm preserves Adam moments across epochs; `pooled_reset` diagnoses resetting Adam each round, as the peer arms do. Probe passes are additional work and are counted separately.
- The primary endpoint is **full-test accuracy of the final rank-16 model**, compared within each seed against `pooled`. Personalized accuracy, where recorded, is secondary and is not a substitute for this endpoint.
- The four factorial arms use the same graph and sample-times-quality base weights: `fixed_quality`, `fixed_domain`, `adaptive_quality`, `adaptive_domain`. Fixed capability ceilings are 4/8/16 repeated across 15 peers. Adaptive arms execute the canonical P1 controller; domain arms execute the canonical P2 conservative policy using local training changes and training histograms.
- Other controls are uniform-rank centralized `fedavg16`, uniform-rank neighbor `mh16`, and capability-limited sample-weighted `mh_sample`. Pooled reference rank 16 is a conventional comparator; it exceeds the smallest peers' resource ceilings.

## What is decentralized

Training factors traverse the declared peer graph. A reversible, row-stochastic Metropolis kernel targets the stated sample/quality/domain weights; it is generally neither symmetric nor doubly stochastic. The domain factors therefore change the actual mixer. The earlier diagonal-scale-plus-Sinkhorn implementation erased these weights and cannot be used as evidence about their effect.

Quality and domain-control metadata are delivered by a graph-restricted tree gather and broadcast. Each peer derives the allocation from its received view. Final deployment uses a neighbor-only tree reduction of weighted effective updates to a predetermined capable peer, followed by one rank-16 refactorization. This is a simulation of explicit message operations, not a distributed socket benchmark. Globally normalized weights and allgather have real communication costs even without a separate permanent server.

**No privacy guarantee is established.** Raw examples stay local in the simulated protocol, but domain-control exchange reveals training class histograms and dense local adapter changes to participating peers. LoRA, decentralization, and local data retention do not by themselves prevent information leakage. There is no differential privacy, secure aggregation, or empirical attack-resistance claim here.

## Resource accounting

Deployment payload is the sum of training factor exchange, all weight-control traffic, and **one final assembly**. Factor values use four bytes; the control-plane ledger models numeric values and source identifiers at eight bytes. The final tree reduction records its actual dense tensor precision and scalar masses. Setup, packet framing, transport headers, and later model dissemination are excluded.

Intermediate assemblies are needed only to draw evaluation curves. Their bytes are recorded separately as `evaluation_only_assembly_bytes`; they are not silently charged as required training communication. Rank-sample products and training factor bytes are useful workload proxies, not measured total FLOPs. Gradient and quality probes, dense control payloads, SVD work, and measured timings remain visible. Peak peer dense memory describes the assembly accumulator and receive buffer; it excludes SVD and diagnostic workspace and is not the simulation process's peak memory.

## Reproduction and outputs

Run training on gpu003 from `project-3-hierarchical-gossip/`, using a fresh output directory:

```sh
CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=~/pystubs:$PWD \
  ~/ahlora-venv/bin/python -m experiments.integrated_benchmark \
  --data-dir ~/ahlora-data --feature-cache ~/ahlora-data/features-v2 \
  --output ~/ahlora-runs/integrated-reproduction \
  --seeds 42 43 44 --rounds 30 --alpha 32
```

All nine arms are selected by default. Final adapter checkpoints are retained at the remote run directory recorded in the manifest; their hashes and independently reproduced predictions are recorded in `independent_checkpoint_verification.json`. Binary checkpoints are excluded by the repository's model-artifact ignore rule. Raw per-round records, exact partitions, source archive, and plots are tracked here.

Run the summarizer on gpu003 after the complete raw record set is present:

```sh
~/ahlora-venv/bin/python scripts/summarize_integrated.py \
  --input docs/artifacts/integrated-corrected \
  --figures paper/multidomain/figures \
  --latex paper/multidomain/integrated_results.tex
```

The script refuses incomplete runs, missing seed/arm pairs, mismatched alpha/initialization/data identity/exposures, inconsistent full-test counts, and inconsistent communication totals. It does not run new training. All figures and tables derive from the raw per-round records, with file and script SHA-256 hashes saved in `summary.json`.

| Output | Purpose |
|---|---|
| `seed*_*.json` | Raw per-round metrics, ranks, weights, matrices, transport ledger, policies and completion status |
| `manifest.json`, `splits_seed*.json` | Configuration, code/data provenance and exact partitions |
| `source_snapshot.tar.gz`, `source_snapshot_manifest.json` | Exact launch sources, canonical policy dependencies and remotely verified hashes |
| `summary.json`, `summary.csv` | Aggregate final results and communication/timing statistics |
| `per_seed.csv` | Unaggregated endpoints and matched differences |
| `factorial_per_seed.csv` | Conditional rank/domain effects, average main effects and interaction |
| `adaptive_resource_pairs.csv` | Adaptive-versus-fixed factor, payload and rank-sample-product changes |
| `curves.csv`, `rank_trajectories.csv` | Plot-ready full-test curves and every peer's rank trajectory |
| `RESULTS.md` | Generated numerical tables with interpretation limits |
| `paper/multidomain/integrated_results.tex` | Generated paper tables |
| `paper/multidomain/figures/integrated_*.pdf`, `.png` | Final comparison, accuracy curves, factorial effects, communication components and rank trajectories |

All reported error bars use **sample standard deviation across seeds** (`ddof=1`). Accuracy differences are computed within matching seeds before aggregation. The factorial interaction is `(adaptive_domain - fixed_domain) - (adaptive_quality - fixed_quality)`. Three seeds and no prespecified equivalence margin support a descriptive comparison; they do not establish formal equivalence. A negative result should be reported directly after checking these completed corrected runs, without substituting invalid historical weighting experiments.

## Exact source snapshot

`source_snapshot.tar.gz` preserves the **35 P3 Python files named by the launch manifest plus four canonical P1/P2 source dependencies**. Every file hash, the combined P3 source hash, and every archived member were verified on gpu003 against the recorded manifest. The accompanying snapshot manifest lists the paths and SHA-256 hashes. The archive contains only these source files, without credentials or datasets.

After the experiment process started, the final repository disabled invalid historical weighted/adaptive composition entry points in `protocol_benchmark.py`; the running process had already loaded the launch version. The integrated driver and its `FeatureClient` behavior did not change. The exact launch `protocol_benchmark.py` was reconstructed from commit `8677d1a` plus the unchanged launched `FeatureClient` block, then accepted **only after its full-file SHA-256 matched the launch manifest**. This preserves the original experiment faithfully while the final repository prevents accidental reuse of invalid legacy methods. The result validator and plotting script were added after launch and are consequently outside that launch archive.

The source snapshot was captured while the experiment was running; this timestamp distinction is recorded in `source_snapshot_manifest.json`. It says nothing about the final run status, which belongs to the raw experiment manifest. To reproduce the launched code, extract the archive into a separate checkout with the recorded dependencies and feature cache, and use the command configuration from `manifest.json`.
