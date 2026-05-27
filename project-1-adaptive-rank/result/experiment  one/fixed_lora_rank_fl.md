# Fixed LoRA Rank for Heterogeneous Federated Learning
## A Study on Capacity–Computation Mismatch

---

## Abstract

Federated Learning (FL) enables collaborative model training across distributed clients with heterogeneous computational resources. Low-Rank Adaptation (LoRA) has recently emerged as an efficient fine-tuning method that significantly reduces the number of trainable parameters. However, existing approaches typically assume a **uniform LoRA rank across all clients**, disregarding variations in computational capability.

This paper investigates the limitations of fixed LoRA rank in heterogeneous federated environments. Through extensive experiments across multiple modalities (image, text, tabular, and audio), we demonstrate that uniform rank allocation leads to **capacity–computation mismatch**, resulting in inefficient resource utilization and aggregation bias. Specifically, low-capability clients under-utilize model capacity, while high-capability clients suffer from representational bottlenecks despite increased computation.

We analyze these effects both mathematically and empirically, showing that client influence is primarily dictated by computational effort rather than parameterization. Based on these findings, we argue that **adaptive LoRA rank allocation is necessary** to align model capacity with client resources, improve fairness, and enhance global model performance.

---

## 1. Introduction

Federated Learning (FL) enables distributed clients to collaboratively train models without sharing raw data. However, FL systems are inherently **heterogeneous**, with clients differing in computational power, data availability, and communication constraints.

Low-Rank Adaptation (LoRA) reduces training complexity by decomposing weight updates into low-rank matrices, significantly lowering parameter count and communication overhead. Despite this advantage, most implementations assume a **fixed LoRA rank across clients**, which is unrealistic in heterogeneous environments.

This paper investigates:

> *How does a fixed LoRA rank affect federated learning performance under heterogeneous client computational capabilities?*

---

## 2. Methodology

### 2.1 Federated Learning

The global model is updated using **Federated Averaging (FedAvg)**:

$$
w^{(t+1)} = \sum_{k=1}^{K} \frac{n_k}{\sum_{j=1}^{K} n_j} \, w_k^{(t)}
$$

where:
- $w_k^{(t)}$ — local model weights of client $k$ at round $t$
- $n_k$ — number of samples processed by client $k$
- $K$ — total number of clients

---

### 2.2 LoRA Parameterization

LoRA decomposes the weight update as:

$$
W' = W + BA
$$

where:
- $A \in \mathbb{R}^{r \times d}$ — low-rank down-projection matrix
- $B \in \mathbb{R}^{d \times r}$ — low-rank up-projection matrix
- $r \ll d$ — rank, much smaller than the original dimension

This reduces the parameter count from $d^2$ to $2rd$:

$$
d^2 \;\longrightarrow\; 2rd
$$

---

### 2.3 Heterogeneous Clients

Clients are assigned different computational budgets while sharing identical LoRA rank and architecture:

| Client | Compute Budget $E_k$ |
|--------|----------------------|
| Client 1 | $E_1 = 1$ |
| Client 2 | $E_2 = 2$ |
| Client 3 | $E_3 = 3$ |

---

### 2.4 Experimental Setup

**Modalities tested:**

| Modality | Model Architecture |
|----------|--------------------|
| Image    | CNN, MLP           |
| Text     | LSTM, Transformer  |
| Tabular  | MLP                |
| Audio    | 1D CNN             |

**Metrics evaluated:**
- Accuracy
- Per-client contribution (%)
- Aggregation weights
- Cumulative samples processed

---

## 3. Results and Analysis

### 3.1 Contribution Imbalance

Client contribution to global aggregation scales proportionally with compute budget:

$$
\text{Contribution}_k \;\propto\; E_k
$$

This means high-compute clients dominate the aggregation process, regardless of parameterization quality.

---

### 3.2 Capacity–Computation Mismatch

The uniform rank creates a two-sided mismatch:

| Client Type | Symptom | Root Cause |
|-------------|---------|------------|
| **Low-capability** | Under-trained parameters, low contribution | Insufficient compute to exploit the full rank |
| **High-capability** | Representational bottleneck | Rank too low to capture additional signal despite high compute |

---

### 3.3 Accuracy Trends

Global accuracy improves steadily across rounds, but the trajectory is disproportionately shaped by high-capability clients, as their updates carry more weight in the FedAvg aggregation step.

---

### 3.4 Aggregation Bias

Since the number of processed samples $n_k$ is proportional to compute budget $E_k$, client influence over the global model is governed by:

$$
\text{Influence}_k \;\propto\; n_k \;\propto\; E_k
$$

This creates a structural **aggregation bias**: compute-rich clients steer the global model regardless of data quality or rank efficiency.

---

### 3.5 Computational Inefficiency *(New Contribution)*

#### Definitions

We define per-client efficiency and computational loss as:

$$
\text{Efficiency}_k = \frac{\Delta \mathcal{L}_k}{E_k}
$$

$$
\text{Computational Loss}_k = E_k - \alpha \cdot \Delta \mathcal{L}_k
$$

where $\Delta \mathcal{L}_k$ is the loss reduction achieved by client $k$ and $\alpha$ is a scaling factor.

#### Case 1 — High Compute + Low Rank (Computation Waste)

When compute is abundant but rank is too small, performance saturates:

$$
\lim_{E_k \to \infty} \Delta \mathcal{L}_k \;\rightarrow\; \text{constant}
$$

Extra computation beyond this point yields no additional improvement — **compute is wasted**.

#### Case 2 — Low Compute + High Rank (Parameter Waste)

When compute is scarce, the model cannot effectively train all rank dimensions:

$$
\Delta \mathcal{L}_k \;\propto\; \frac{E_k}{r}
$$

The client trains only a fraction of the allocated parameters — **parameters are wasted**.

#### System-Level Summary

| Client Type | Problem | Wasted Resource |
|-------------|---------|-----------------|
| High compute + Low rank | Representational bottleneck | Computation |
| Low compute + High rank | Under-training | Parameters |

---

## 4. Discussion

The findings reveal a fundamental issue in current federated LoRA approaches:

- FL is **resource-driven**, not parameter-driven — the client with the most compute dominates aggregation, not the one with the best-fitted model.
- A uniform rank causes systematic imbalance in how each client influences the global model.
- The mismatch between model capacity (governed by rank $r$) and computational capability (governed by $E_k$) leads to both inefficiency and unfairness.

> **Key issue:** Misalignment between model capacity and computational capability.

---

## 5. Conclusion

This paper demonstrates that a **fixed LoRA rank in heterogeneous federated learning** creates a fundamental capacity–computation mismatch:

- High-resource clients **waste computation** against a rank ceiling.
- Low-resource clients **waste parameters** they cannot train effectively.
- Aggregation becomes **biased toward compute-rich clients**, regardless of data quality.

### Final Statement

> **Adaptive LoRA rank is necessary for efficient, fair, and scalable federated learning.**

---

## 6. Future Work

### 6.1 Adaptive Rank Allocation

Scale rank proportionally to client compute capacity:

$$
r_k \;\propto\; \text{compute capacity of client } k
$$

### 6.2 Joint Optimization of Aggregation Weights

Incorporate both sample count and rank into aggregation:

$$
\alpha_k = f(n_k,\, r_k)
$$

### 6.3 Dynamic Rank Scheduling

Adjust rank dynamically during training as client resources fluctuate, rather than fixing it at initialization.

### 6.4 Real-World Constraints

Extend the framework to account for:
- Bandwidth limitations
- Energy budgets
- Device availability and dropout

### 6.5 Non-IID Data

Evaluate robustness of adaptive rank strategies under non-identically distributed data across clients, and develop fairness-aware rank allocation mechanisms.

---

## Keywords

`Federated Learning` · `LoRA` · `Adaptive Models` · `Distributed Systems` · `Heterogeneous Computing`
