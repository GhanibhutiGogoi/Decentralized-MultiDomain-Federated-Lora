# Week 1-2: Foundation & Low-Hanging Fruit

## What We Built and Why

### The Research Goal (Reminder)
You're building a **Hierarchical Gossip Aggregation** protocol for decentralized federated learning. The key insight: instead of treating all clients equally in gossip, cluster them by domain and gossip hierarchically (within-domain first, then between-domain with learned weights).

But before building the novel algorithm, we need **baselines to beat** and **infrastructure to build on**. That's what Week 1-2 is about.

---

## What Each File Does (Plain English)

### 1. Data Pipeline — `src/data/cifar100_domains.py`

**What it does:**
Takes CIFAR-100 (100 classes of tiny images) and creates a realistic federated learning scenario:

- **5 domains** of 20 classes each, grouped by CIFAR-100's natural superclasses:
  - Domain 0: Aquatic & Nature (fish, flowers, aquatic mammals, food)
  - Domain 1: Household & Insects (furniture, electrical devices, fruit/veg, insects)
  - Domain 2: Large Animals & Outdoor (carnivores, outdoor scenes, herbivores)
  - Domain 3: Medium Animals & People (mammals, invertebrates, people, reptiles)
  - Domain 4: Small Animals & Vehicles (small mammals, trees, vehicles)

- **15 clients** (3 per domain), each getting a non-IID (non-uniform) slice of their domain's data via Dirichlet distribution (alpha=0.5)

**Why this matters for the research:**
This simulates the real-world scenario from your proposal — e.g., 3 cardiology hospitals each with slightly different patient populations (non-IID), but all in the "cardiology domain." The 5 domains simulate the multi-domain setting (cardiology vs oncology vs neurology).

**Why CIFAR-100 and not LLMs:**
CIFAR-100 + ResNet-18 trains in minutes, not hours. You debug algorithms fast, get results fast, iterate fast. Once the algorithm works, you scale to LLMs later (Month 3-4).

---

### 2. LoRA Model — `src/models/lora_resnet.py`

**What it does:**
Takes a pretrained ResNet-18 (11.7M parameters), freezes everything, and adds a LoRA adapter to the final fully-connected layer.

- **LoRA rank 16**: Only ~16K trainable parameters out of 11.7M (~0.14%)
- The LoRA adapter is: `output = frozen_FC(x) + (x @ A^T @ B^T) * scaling`
- A is (rank × input_dim), B is (output_dim × rank)
- A initialized with Kaiming, B initialized with zeros (so LoRA starts as no-op)

**Why custom LoRA instead of using PEFT library:**
PEFT's LoRA targets transformer attention layers. ResNet has convolutions and FC layers. Writing a custom `LoRALinear` wrapper is cleaner and gives us full control over extracting/setting LoRA matrices — which is critical for the gossip aggregation.

**Key functions you'll use later:**
- `get_lora_state(model)` → extracts A and B matrices as CPU tensors
- `set_lora_state(model, state)` → loads A and B matrices into model
- These are the "messages" that clients exchange during gossip

---

### 3. Federated Client — `src/federated/client.py`

**What it does:**
Each client is one "organization" in the federated network. It:
- Holds its own private training data (never shared)
- Holds its own copy of the model
- Can `train()` locally for E epochs on its private data
- Can `evaluate()` on its test set
- Can `get_lora_state()` / `set_lora_state()` to exchange LoRA params

**The key insight:** Only LoRA parameters (A, B matrices) are exchanged between clients. The frozen ResNet base weights never move. This is what makes communication efficient.

---

### 4. FedAvg Baseline — `src/federated/fedavg.py`

**What it does:**
Implements classic Federated Averaging with a central server. Each round:
1. Server broadcasts global LoRA params to all 15 clients
2. Each client trains locally for 5 epochs
3. Each client sends updated LoRA params back to server
4. Server computes weighted average (weighted by number of samples)
5. Repeat

**Why this is a "low-hanging fruit":**
- This is the **upper bound** for your gossip approach. A central server has perfect information and can aggregate optimally.
- Your gossip protocol should approach FedAvg's accuracy while being fully decentralized.
- Getting this number early tells you what's achievable.

**What to expect:** ~40-55% accuracy on CIFAR-100 after 50 rounds with LoRA rank 16 on ResNet-18. This is reasonable — full fine-tuning gets ~75%, but we're only training 0.14% of parameters.

---

### 5. Basic Gossip Protocol — `src/federated/gossip.py`

**What it does:**
Implements decentralized gossip WITHOUT domain awareness. Each round:
1. All clients train locally
2. Each client picks a random neighbor from the topology graph
3. Client and neighbor average their LoRA params (50/50 uniform weight)
4. Repeat

Supports three topologies:
- **Ring**: Each client connected to 2 neighbors (minimal connectivity)
- **Fully connected**: Everyone talks to everyone (maximum but unrealistic)
- **Random regular**: Each client has exactly 4 random neighbors

**Why this is a "low-hanging fruit":**
- This is the **decentralized baseline to beat**. It's domain-unaware — a cardiology client might average with a neurology client, diluting both.
- Your hierarchical gossip should outperform this by being domain-smart.
- The gap between FedAvg and this baseline is the "room for improvement" your paper claims.

**What to expect:** Lower accuracy than FedAvg, slower convergence, and particularly poor per-domain performance because uniform averaging hurts domain-specific knowledge.

---

### 6. Domain Clustering — `src/clustering/domain_clustering.py`

**What it does:**
After clients train locally for a few rounds, their LoRA matrices diverge based on their data domain. This module:
1. Extracts SVD features from each client's LoRA A and B matrices (singular values as fingerprint)
2. Also computes the product BA (the effective LoRA update) and its SVD
3. Normalizes features
4. Runs Agglomerative Clustering (Ward linkage) to group clients into 5 clusters
5. Evaluates against true domain labels using ARI (Adjusted Rand Index) and NMI (Normalized Mutual Information)

**Why this is the most important "low-hanging fruit":**
- **This is the foundation of your entire paper.** If LoRA matrices can't distinguish domains, hierarchical gossip has no basis.
- If ARI > 0.7 after just 10 rounds of local training, it proves the core hypothesis: clients in the same domain learn similar LoRA patterns.
- This one experiment validates the entire research direction.

**What to expect:** ARI should be > 0.7 (good clustering). If it's < 0.5, we need to adjust (more training rounds, higher rank, or different features).

---

### 7. Utils — `src/utils/metrics.py` and `visualization.py`

**Metrics:** Tracks accuracy, loss, per-domain accuracy, communication cost, and fairness (accuracy gap between domains).

**Visualization:** Plots convergence curves, per-domain accuracy, t-SNE of LoRA features, and data distribution across clients.

---

## The Three Experiments to Run

### Experiment 1: `python -m experiments.01_fedavg_baseline`
**Purpose:** Establish the centralized upper bound.
**Output:** Convergence curve, per-domain accuracy, final accuracy number.
**What to look for:**
- Does accuracy plateau or keep improving at round 50?
- Are there big gaps between domains? (fairness issue)

### Experiment 2: `python -m experiments.02_gossip_baseline`
**Purpose:** Establish the decentralized lower bound (domain-unaware gossip).
**Output:** Convergence curve, per-domain accuracy, total messages exchanged.
**What to look for:**
- How much worse is gossip vs FedAvg? (this is the gap you'll close)
- Which domains suffer most from uniform averaging?

### Experiment 3: `python -m experiments.03_clustering_validation`
**Purpose:** Validate that LoRA features can recover domain structure.
**Output:** Clustering metrics (ARI, NMI), t-SNE visualization.
**What to look for:**
- ARI > 0.7 means clustering works → green light for hierarchical gossip
- The t-SNE plot should show 5 distinct clusters matching true domains

---

## How to Run Everything

```bash
# 1. Navigate to project
cd project-3-hierarchical-gossip

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run experiments (in order)
python -m experiments.01_fedavg_baseline
python -m experiments.02_gossip_baseline
python -m experiments.03_clustering_validation

# Results will be saved in ./logs/
```

---

## What These Results Set Up for Week 3-4

Once you have these three numbers, you can write this table in your paper:

| Method | Avg Accuracy | Domain Fairness Gap | Communication |
|--------|-------------|-------------------|---------------|
| FedAvg (centralized) | X% | Y% | Centralized |
| Standard Gossip (ring) | A% | B% | Decentralized |
| **Hierarchical Gossip (ours)** | _TBD (Week 3-4)_ | _TBD_ | Decentralized |

The goal: your hierarchical gossip should be closer to FedAvg than standard gossip, especially on the fairness gap (per-domain variance), while using fewer total messages.

---

## Key Design Decisions and Why

| Decision | Alternative | Why we chose this |
|----------|------------|-------------------|
| CIFAR-100 (not LLM) | Llama-2-7B | Debug in minutes, not hours. Scale later. |
| ResNet-18 + custom LoRA | PEFT library | Full control over A,B extraction for gossip |
| 5 domains × 3 clients | 3 domains × 9 clients | Matches proposal structure, clear domain separation |
| Dirichlet alpha=0.5 | Uniform or extreme non-IID | Moderate non-IID — realistic but not pathological |
| Ring topology for gossip | Fully connected | Hardest case for decentralized. If it works on ring, it works anywhere |
| LoRA rank 16 | Rank 4 or 64 | Sweet spot: enough capacity, fast training, clear differences between domains |

---

## File Dependency Map

```
configs/default_config.yaml
         │
         ▼
src/data/cifar100_domains.py  ──→  Creates client data splits
         │
         ▼
src/models/lora_resnet.py  ──→  Creates model with LoRA
         │
         ▼
src/federated/client.py  ──→  Wraps model + data into trainable client
         │
    ┌────┴────┐
    ▼         ▼
fedavg.py   gossip.py  ──→  Two aggregation strategies
                │
                ▼
src/clustering/domain_clustering.py  ──→  Clusters clients by LoRA similarity
         │
         ▼
src/utils/{metrics,visualization}.py  ──→  Track and plot results
         │
         ▼
experiments/01,02,03  ──→  Runnable experiment scripts
```
