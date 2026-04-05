# Project 2: Domain-Aware LoRA Rank Allocation

**Part of the AH-LoRA (Adaptive Heterogeneous LoRA) research framework.**

## Research Question

> Given a federated client with data distribution D and resource constraints R, what is the optimal LoRA rank r* that maximizes accuracy?

## Key Innovation

A **domain complexity metric** that automatically determines optimal LoRA rank for each client based on 5 data-driven sub-metrics, eliminating the need for manual rank tuning in heterogeneous federated learning.

## Quick Start

```bash
cd project-2-domain-aware-allocation

# Install dependencies
pip install -r requirements.txt

# Run experiments in order:
python experiments/01_data_and_complexity.py   # ~5 min
python experiments/02_oracle_rank_search.py    # ~2 hours
python experiments/03_correlation_analysis.py  # seconds
python experiments/04_allocation_comparison.py # ~30 min
```

## Project Structure

```
project-2-domain-aware-allocation/
├── configs/default_config.yaml     # All hyperparameters
├── src/
│   ├── data/cifar100_domains.py    # CIFAR-100 → 5 domains, 15 clients
│   ├── models/lora_resnet.py       # ResNet-18 + variable-rank LoRA
│   ├── complexity/
│   │   └── domain_complexity.py    # Core: 5 sub-metrics → complexity score
│   ├── allocation/
│   │   └── rank_allocator.py       # Uniform / Random / Domain-Aware / Oracle
│   ├── federated/
│   │   ├── client.py               # Local training client
│   │   └── hetero_fedavg.py        # FedAvg with heterogeneous ranks
│   └── utils/
│       ├── metrics.py              # Tracking & fairness metrics
│       └── visualization.py        # All plots
├── experiments/
│   ├── 01_data_and_complexity.py   # Compute complexity scores
│   ├── 02_oracle_rank_search.py    # Brute-force optimal rank
│   ├── 03_correlation_analysis.py  # Validate hypothesis
│   └── 04_allocation_comparison.py # Strategy comparison
└── results/                        # Auto-created outputs
```

## Week 1-2 Milestones

- **Milestone 1**: Prove domain complexity metric predicts optimal rank (Experiments 01-03)
- **Milestone 2**: Show domain-aware allocation beats uniform/random in FedAvg (Experiment 04)

See [WEEK1-2_GUIDE.md](WEEK1-2_GUIDE.md) for detailed explanation of all code.
