# Week 1-2 Guide: Domain-Aware LoRA Rank Allocation

## Overview

This guide explains the Week 1-2 codebase for Project 2 of the AH-LoRA framework. The goal is to validate a fundamental hypothesis:

> **A domain complexity metric computed from a client's local data can predict the optimal LoRA rank for that client in federated learning.**

If true, this enables automatic rank allocation — no manual tuning needed.

---

## The Research Hypothesis

### Why This Matters

In heterogeneous federated learning, clients have different data domains and computational resources. Standard approaches assign a uniform LoRA rank to all clients (e.g., rank 16 for everyone). This is suboptimal because:

- A client with simple, low-diversity data wastes parameters with rank 64
- A client with complex, diverse data is constrained by rank 4
- The optimal rank depends on the intrinsic complexity of the client's domain

### What We Want to Show

1. **Different clients genuinely need different ranks** (Oracle search, Experiment 02)
2. **We can predict the optimal rank without exhaustive search** (Complexity metric, Experiment 01)
3. **Using predicted ranks improves federated learning** (Allocation comparison, Experiment 04)

---

## Architecture

### Data Pipeline: `src/data/cifar100_domains.py`

Splits CIFAR-100 into 5 domains using the superclass hierarchy:

| Domain | Superclasses | Example Classes |
|--------|-------------|-----------------|
| 0: Aquatic & Nature | aquatic mammals, fish, flowers, food | dolphin, trout, tulip, plate |
| 1: Household & Insects | fruit/veg, household devices, furniture, insects | apple, lamp, bed, butterfly |
| 2: Large Animals & Outdoor | large carnivores, outdoor things, scenes, herbivores | bear, bridge, forest, elephant |
| 3: Medium Animals & People | medium mammals, invertebrates, people, reptiles | fox, spider, boy, snake |
| 4: Small Animals & Vehicles | small mammals, trees, vehicles 1 & 2 | hamster, oak, bus, pickup |

Each domain has 20 classes and 3 clients. Data is split using Dirichlet distribution (alpha=0.5) for non-IID partitioning.

**Key addition vs. P3:** The `train_indices` field is stored per client, and `get_client_labels()` provides numpy label arrays for complexity computation.

### LoRA Model: `src/models/lora_resnet.py`

ResNet-18 with LoRA on the final FC layer. Supports **variable ranks** — each client can use a different rank from {4, 8, 16, 32, 64}.

**Key additions for heterogeneous aggregation:**

- `merge_lora_to_delta_w(lora_state)`: Computes the effective weight update `delta_W = B @ A`. This is **rank-independent** — the output is always (out_features x in_features) regardless of the rank used to produce it.

- `decompose_delta_w(delta_w_dict, target_rank)`: Uses SVD to decompose `delta_W` back into A, B matrices at any desired rank. This enables converting between ranks.

- `reset_lora_params(model)`: Re-initializes LoRA to start fresh (for oracle search).

### Domain Complexity: `src/complexity/domain_complexity.py`

**This is the core research contribution.**

The `DomainComplexityAnalyzer` computes a composite complexity score from 5 sub-metrics:

#### Sub-metric 1: Label Entropy (weight: 0.3)

```
H(Y) = -Σ p(y) log₂ p(y), normalized by log₂(num_classes)
```

Measures the diversity of the client's class distribution. A client with 20 equally-represented classes has entropy = 1.0 (most complex). A client dominated by 2 classes has low entropy.

**Intuition:** More diverse class distributions require more model capacity (higher rank) to capture all class boundaries.

#### Sub-metric 2: Feature Diversity (weight: 0.2)

Mean pairwise cosine distance between the client's data representations in the pretrained model's feature space (512-dim avgpool features).

**Intuition:** If a client's data is tightly clustered in feature space, a low-rank adaptation suffices. If the data is spread across diverse representations, higher rank is needed.

#### Sub-metric 3: Intrinsic Dimensionality (weight: 0.2)

Fraction of PCA components needed to explain 95% variance, normalized by feature dimension.

**Intuition:** Data that lives on a low-dimensional manifold (few PCA components needed) can be adapted with a low-rank LoRA. Data with high intrinsic dimensionality needs more rank.

#### Sub-metric 4: Task Difficulty (weight: 0.2)

1 - accuracy of a linear probe trained on frozen pretrained features, scaled by log(num_classes).

**Intuition:** If the pretrained model already works well on this domain (high linear probe accuracy), less adaptation is needed (lower rank). If the domain is far from ImageNet (low probe accuracy), more adaptation is needed.

#### Sub-metric 5: Data Imbalance (weight: 0.1)

Gini coefficient of the class distribution. High Gini = heavily imbalanced.

**Intuition:** Imbalanced data creates a harder optimization landscape, potentially requiring more model capacity. Gets a multiplicative penalty if Gini > 0.8.

#### Composite Score

```
score = 0.3 * entropy + 0.2 * diversity + 0.2 * intrinsic_dim + 0.2 * difficulty + 0.1 * imbalance
```

The weights are configurable in `configs/default_config.yaml` for ablation studies.

### Rank Allocator: `src/allocation/rank_allocator.py`

Four strategies:

| Strategy | Formula | Use Case |
|----------|---------|----------|
| **Uniform** | All clients get rank r | Baseline |
| **Random** | Random from {4,8,16,32,64} | Baseline |
| **Domain-Aware** | `snap(16 × (1 + 3 × complexity))` | Our method |
| **Oracle** | Best rank from grid search | Upper bound |

Domain-aware allocation maps:
- complexity ≈ 0.0 → rank 16
- complexity ≈ 0.3 → rank 32
- complexity ≈ 0.5 → rank 32-64
- complexity ≈ 1.0 → rank 64

### Heterogeneous FedAvg: `src/federated/hetero_fedavg.py`

Standard FedAvg requires all clients to have the same LoRA rank. Our `HeteroFedAvgServer` solves this by aggregating in **delta_W space**:

1. Each client trains locally and sends LoRA params (A, B at their rank)
2. Convert each client's params to `delta_W = B @ A` (always same shape)
3. Weighted average of `delta_W` matrices
4. SVD-decompose the average back to each client's assigned rank
5. Send rank-specific LoRA params back

This is more principled than zero-padding because:
- The effective weight update is rank-independent
- SVD preserves the most important directions
- No artificial zeros introduced

---

## Experiments

### Experiment 01: Data & Complexity Analysis

**What it does:** Creates the federated data splits and computes complexity scores for all 15 clients.

**What to look for:**
- Do clients in the same domain have similar complexity scores?
- Which sub-metrics vary most across domains?
- Is the complexity score range wide enough to differentiate clients?

**Output files:**
- `results/complexity_scores.json` — raw data
- `results/complexity_breakdown.png` — stacked bar chart of sub-metrics
- `results/data_distribution.png` — sample counts per client

### Experiment 02: Oracle Rank Search

**What it does:** For each of 15 clients, trains LoRA independently at ranks {4, 8, 16, 32, 64} for 20 epochs. Records test accuracy at each rank.

**What to look for:**
- Do different clients genuinely have different optimal ranks?
- Is there a clear relationship between domain and optimal rank?
- Is the accuracy difference between best and worst rank significant?

**Resumability:** If the run is interrupted, it saves partial results. Just restart — it picks up where it left off.

**Output files:**
- `results/oracle_results.json` — accuracy grid
- `results/oracle_heatmap.png` — visual heatmap

### Experiment 03: Correlation Analysis

**What it does:** Tests the fundamental hypothesis by computing Spearman correlation between complexity score and oracle-optimal rank.

**Success criteria:** Spearman ρ > 0.6 with p < 0.05

**What to look for:**
- R² value: fraction of rank variance explained by complexity
- Per-submetric correlations: which sub-metric is most predictive?
- If correlation is weak: which clients are outliers?

**Output files:**
- `results/complexity_vs_rank.png` — **THE key figure for the paper**
- `results/submetric_correlations.png` — per-submetric analysis
- `results/correlation_analysis.json` — all statistics

### Experiment 04: Allocation Comparison

**What it does:** Runs 50 rounds of HeteroFedAvg with 3 strategies and compares.

**What to look for:**
- Domain-aware should have higher final accuracy than uniform
- Domain-aware should have lower fairness gap (accuracy variance across domains)
- Random should be worst (arbitrary ranks, some clients over/under-provisioned)

**Output files:**
- `results/allocation_convergence.png` — 3 convergence curves
- `results/allocation_per_domain.png` — per-domain accuracy bars
- `results/allocation_comparison.json` — full results

---

## Troubleshooting

### Weak correlation (ρ < 0.5)

1. Check if all clients have the same oracle rank — if so, the dataset isn't diverse enough. Try lower `dirichlet_alpha` (more non-IID)
2. Check per-submetric correlations — maybe some sub-metrics hurt. Try setting their weight to 0
3. The 20-epoch oracle might be too short — try 40 epochs

### Domain-aware doesn't beat uniform

1. Check if the complexity scores are all similar — if so, domain-aware degenerates to near-uniform
2. The base_rank and scaling factor in the allocation formula may need tuning
3. Try a wider range of candidate ranks (add rank 2 and rank 128)

### Memory issues

- ResNet-18 is small (~45MB). If running on CPU, reduce `n_feature_samples` to 200
- Oracle search processes clients sequentially — no memory accumulation
- Reduce `batch_size` in config if needed

---

## Next Steps (Week 3-4)

1. **Ablation study** on complexity metric weights
2. **More datasets**: DomainNet, Office-Home
3. **Resource-aware allocation**: incorporate GPU memory constraints
4. **Integration with Project 3**: domain-aware allocation + gossip aggregation
