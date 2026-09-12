# Project 3: Hierarchical Gossip Aggregation for Decentralized Federated Learning

## Overview

This project implements **hierarchical gossip aggregation** for decentralized federated learning. Clients train LoRA adapters on a shared frozen ResNet-18 backbone and exchange updates through a configurable mixing matrix. The current hierarchy accepts known domain groups as inputs; discovering those groups from learned adapters remains a separate experiment.

## Key Ideas

1. **Delta-W merging**: Expand each scaled adapter `(alpha / rank) B A`, mix in a common matrix space, and refactorize by truncated SVD to each client's rank.
2. **Mass-preserving gossip**: Symmetric doubly stochastic Metropolis-Hastings, affinity and two-tier mixing.
3. **Oracle hierarchy**: Frequent communication within known domains and periodic bridges between them. Domain groups and transfer weights are inputs, not automatically learned outputs.
4. **Measured compression**: Optional error feedback, absolute residual energy, relative tail mass and consensus distance.

## Project Structure

```
project-3-hierarchical-gossip/
├── configs/             # Experiment configurations
├── src/
│   ├── data/            # Dataset loading and domain splitting
│   ├── models/          # ResNet-18 + LoRA model
│   ├── federated/       # Delta-W merge, mixing, hierarchy, runner and legacy baselines
│   ├── clustering/      # SVD-based domain clustering
│   └── utils/           # Metrics and visualization
├── experiments/         # Runnable experiment scripts
├── notebooks/           # Exploration notebooks
└── literature/          # Reading list and paper notes
```

## Recorded baseline scripts

Experiments 01–03 preserve the original implementation and its two-round smoke-test lineage. They are useful historical references, but do not exercise the new runner or establish current benchmark performance.

```bash
pip install -r requirements.txt
python experiments/01_fedavg_baseline.py
python experiments/02_gossip_baseline.py
python experiments/03_clustering_validation.py
```

## Current Status

- [x] CIFAR-100 data pipeline, frozen-backbone LoRA model and recorded baselines.
- [x] Gauge-invariant merging with heterogeneous-rank SVD projection.
- [x] Metropolis-Hastings, affinity and two-tier mixing.
- [x] Decentralized runner with error feedback and compression diagnostics.
- [x] Convergence analysis and numerical tests of its stated assumptions.
- [ ] Reproducible real-data benchmark with personalized and consensus evaluation.
- [ ] Three-seed baseline and heterogeneous-rank experiment battery.
- [ ] Automatic domain discovery; the original singular-value signature did not recover the known domains.

The immediate completion target is a measured CIFAR-100 system with explicit oracle-domain assumptions. NLP experiments, rank-policy calibration and a paper are extensions, not prerequisites.

The new entry points and output contracts are documented in [Experiment 04](EXPERIMENT04.md) (protocol benchmark) and [Experiment 05](EXPERIMENT05.md) (offline signature validation). Their implementation and their measured completion status are tracked separately.

See [`docs/artifacts/claude_handoff.json`](../docs/artifacts/claude_handoff.json) for current work status, actual result paths and pull requests, and the [artifact reading guide](../docs/artifacts/README.md) for metric definitions. Run tests and experiments on the project's documented SSH GPU machine. Machine access details are intentionally excluded from the shareable handoff.
