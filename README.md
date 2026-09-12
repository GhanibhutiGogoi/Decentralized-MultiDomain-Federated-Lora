# AH-LoRA

A working testbed for heterogeneous LoRA adapters in multi-domain federated learning. Clients share a frozen backbone, train low-rank classification-head updates, and aggregate those updates centrally or through peer-to-peer gossip. The completion target is a reproducible measured system; a paper is optional.

The immediate benchmark uses real CIFAR-100, ImageNet-pretrained ResNet-18, and 15 clients across five known domains. It compares local-only training, client-uniform centralized FedAvg, flat Metropolis-Hastings gossip, and a hierarchy supplied with the true domain groups. Merging expands each scaled update `(alpha / rank) B A`, averages in a common matrix space, then refactorizes by SVD at each client's rank.

## Current status

As of 12 September 2026, the new P3 benchmark, feature-cache checks and signature validation are implemented. The SSH GPU machine passed **379 P3 tests** and **100 P2 tests plus 46 subtests**. The full-data P3 battery (39 completed method/seed runs across uniform and heterogeneous contexts) and signature validation completed, as did the all-five-real-data P2 regeneration.

| Folder | Purpose | Remaining scope |
|---|---|---|
| `project-1-adaptive-rank` | Select client adapter ranks | Rank-policy calibration remains unresolved |
| `project-2-domain-aware-allocation` | Fit domain-aware aggregation weights | Real-data Experiment 1/2 complete; calibration remains exploratory |
| `project-3-hierarchical-gossip` | Mix heterogeneous adapters without a central server | Benchmark and offline discovery artifacts complete; automatic discovery remains open |

The homogeneous schedule uses rank 16; the heterogeneous schedule repeats ranks 4, 12 and 32. Both have total rank 240 over 15 clients. Seeds are 42, 43 and 44, alpha is 32, and consensus evaluation uses reference rank 16. Fixed heterogeneous ranks do not establish that an adaptive rank policy works. Known-domain hierarchy does not establish automatic domain discovery.

## Reproduce and inspect

Use the documented SSH GPU machine for tests and experiments. Each project is self-contained and has its own dependency file. The [Experiment 04 guide](project-3-hierarchical-gossip/EXPERIMENT04.md) gives the exact benchmark commands, evaluation rules, payload accounting and cache protocol. The [Experiment 05 guide](project-3-hierarchical-gossip/EXPERIMENT05.md) compares spectral and direction-aware adapter signatures at stages 2, 5, 10 and 20; labels are used for scoring only.

The feature cache contains all 50,000 official CIFAR-100 training images and 10,000 test images, encoded by the frozen backbone in evaluation mode. Every run records source and data hashes, configuration, environment, per-round accuracy, per-domain fairness, communication and compression diagnostics. Personalized accuracy and full-test consensus accuracy are separate outputs. Network costs are simulated payload counts; no distributed network speedup is claimed.

Start with the [machine-readable Claude handoff](docs/artifacts/claude_handoff.json), [metric and artifact guide](docs/artifacts/README.md), and [append-only progress log](docs/artifacts/progress.jsonl). They distinguish completed evidence from smoke tests and historical results. [PR #42](https://github.com/GhanibhutiGogoi/Decentralized-MultiDomain-Federated-Lora/pull/42) contains the P2 projection and real-data runtime fixes; the P3 completion PR contains the benchmark, signatures and evidence bundle.

The old P3 experiments 01–03 preserve their historical two-round protocol and should not be mixed into new benchmark curves. Existing P2 lambda tables are stale until the real-data regeneration completes. The convergence analysis is in [the research note](docs/research/2026-09-03-project3-convergence-analysis.md), with its assumptions and separate error-feedback/default bounds stated explicitly.

## Results table

The aggregate report keeps uniform and heterogeneous protocols separate and reports final-round means with sample standard deviations over seeds 42, 43 and 44.

| Comparison context | Arms | Seeds | Final personalized accuracy | Final consensus accuracy | Communication/compression | Status |
|---|---|---:|---:|---:|---|---|
| Uniform rank 16, ΔW | Local / FedAvg / MH / oracle | 42, 43, 44 | 57.93±0.55% / 23.20±0.29% / 42.30±1.39% / 68.98±0.95% | 14.23±1.27% / 23.20±0.29% / 15.14±5.10% / 10.25±0.62% | report in `docs/artifacts/p3-completion-report` | complete |
| Heterogeneous ranks 4/12/32, ΔW | Local / FedAvg / MH / oracle | 42, 43, 44 | 53.60±0.62% / 5.74±0.71% / 28.45±2.42% / 44.07±0.70% | 14.50±0.80% / 5.73±0.59% / 6.83±0.45% / 11.47±1.13% | report in `docs/artifacts/p3-completion-report` | complete |
| Heterogeneous ranks 4/12/32, factor zero-pad | FedAvg / MH / oracle | 42, 43, 44 | 4.71±0.22% / 20.63±4.40% / 31.24±1.27% | 5.35±0.37% / 7.75±1.64% / 8.13±1.10% | factor payload comparison in report | complete |
| Heterogeneous ranks 4/12/32, ΔW + feedback | MH / oracle | 42, 43, 44 | 26.63±1.68% / 41.28±0.71% | 10.62±1.33% / 9.45±1.40% | feedback residuals in report | complete |
| Signature validation stages 2/5/10/20 | Local, MH; spectral/row-norm/inverse-L2 | 42, 43, 44 | n/a | n/a | stage-10 mean ARI ≤0.274; see `docs/artifacts/p3-signatures` | complete; discovery gate remains open |
