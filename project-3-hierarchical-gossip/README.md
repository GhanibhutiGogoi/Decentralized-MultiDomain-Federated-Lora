# Project 3: Hierarchical Gossip Aggregation for Decentralized Federated Learning

## Overview

This project implements a **hierarchical gossip aggregation algorithm** for decentralized federated learning that clusters clients by domain similarity and performs domain-aware peer-to-peer communication.

## Key Ideas

1. **Domain-Aware Clustering**: Cluster clients into domain groups based on LoRA matrix similarity (SVD features)
2. **Hierarchical Gossip**: Within-domain gossip (frequent, high-weight) + between-domain gossip (learned transfer weights)
3. **Convergence Guarantees**: Theoretical analysis for decentralized settings

## Project Structure

```
project-3-hierarchical-gossip/
├── configs/             # Experiment configurations
├── src/
│   ├── data/            # Dataset loading and domain splitting
│   ├── models/          # ResNet-18 + LoRA model
│   ├── federated/       # Client, FedAvg, Gossip implementations
│   ├── clustering/      # SVD-based domain clustering
│   └── utils/           # Metrics and visualization
├── experiments/         # Runnable experiment scripts
├── notebooks/           # Exploration notebooks
└── literature/          # Reading list and paper notes
```

## Quick Start

```bash
pip install -r requirements.txt
python experiments/01_fedavg_baseline.py
python experiments/02_gossip_baseline.py
python experiments/03_clustering_validation.py
```

## Current Status

- [x] Week 1-2: Foundation (data pipeline, model, FedAvg baseline, basic gossip, clustering)
- [ ] Week 3-4: Hierarchical gossip with domain-aware topology
- [ ] Week 5-6: Cross-domain transfer weights
- [ ] Week 7-8: NLP experiments with LLMs
