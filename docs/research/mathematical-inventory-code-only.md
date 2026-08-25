# AH-LoRA — Mathematical Inventory (Code Only)

**Scope:** every equation below is transcribed from source in this repository and verified by reading the implementing lines. **Nothing in this document comes from the three proposal PDFs or from `Project.md`.** Mechanisms that appear only in the proposals are listed, unelaborated, in §E so the boundary between "hypothesis" and "implemented" is unambiguous.

Audited 2026-07-21 against `main` @ `7885349`.

| Legend | Meaning |
|---|---|
| **[OK]** | Implemented and mathematically sound |
| **[DEGENERATE]** | Implemented correctly but produces no useful variation on the current data |
| **[DEFECT]** | Implemented incorrectly — see §D |

**Contents:** [A. Rank allocation](#a-rank-allocation) · [B. Weighting](#b-weighting) · [C. Merge operator](#c-merge-operator) · [D. Defects](#d-defects-in-the-implemented-math) · [E. Absent from code](#e-absent-from-code) · [F. Summary](#f-summary)

---

## A. Rank allocation

### A1. Gradient stable rank — **[OK]**

`project-1-adaptive-rank/rank_allocation/rank_selector.py:48-70`

$$s(G) \;=\; \operatorname{median}_{\,p \,\in\, \mathcal{P}_{2\mathrm{D}}}\!\left(\frac{\lVert G_p \rVert_F^2}{\lVert G_p \rVert_2^2}\right)$$

- $\mathcal{P}_{2\mathrm{D}}$ = trainable 2-D parameters with non-`None` gradients, accumulated over `num_batches = 3` mini-batches.
- Terms with $\lVert G_p \rVert_2^2 \le 10^{-12}$ are skipped; empty set returns $1.0$.
- Spectral norm via `torch.linalg.matrix_norm(grad, ord=2)`.

Interpretation: the stable rank of a matrix lies in $[1, \operatorname{rank}(G)]$ and measures how many directions carry comparable gradient energy. Local, label-free, no peers required, ~3 backward passes.

### A2. Capability fraction — **[OK]**

`project-1-adaptive-rank/rank_allocation/rank_selector.py:14-21`

$$c_i \;=\; \frac{\operatorname{index}(b_i)}{\lvert \mathcal{B}\rvert - 1}, \qquad \mathcal{B} = \operatorname{sorted}(\{16, 64, 256\})$$

so $c \in \{0,\; 0.5,\; 1\}$. Returns $0.0$ for an unknown batch size, $1.0$ if $\lvert\mathcal{B}\rvert = 1$. Batch size is used as a proxy for client compute capability.

### A3. Rank equation — **[DEFECT — see D1]**

`project-1-adaptive-rank/rank_allocation/rank_selector.py:38-45`

$$\mathcal{R}_i = \{\, r \in [2,4,6,8,12,16,24,32] \;:\; r \le R_i^{\max} \,\}$$

$$\operatorname{floor}_i = \max\!\big(\min \mathcal{R}_i,\; c_i R_i^{\max}\big)$$

$$r_i \;=\; q_{\mathcal{R}_i}\!\Big(\max\big\{\operatorname{floor}_i,\; \min\big(s(G),\, R_i^{\max}\big)\big\}\Big)$$

- $R^{\max}$ from `config.py:10`: `BATCH_TO_MAX_RANK = {16: 4, 64: 8, 256: 16}`.
- $q_{\mathcal{R}}$ = nearest candidate, ties broken toward the **smaller** rank (`_nearest_candidate`, line 11: `key=lambda r: (abs(r - rank), r)`).
- Stateless: `estimate_optimal_rank` takes no `current_rank` argument, so there is no temporal dependence, hysteresis, or update cost.

### A4. Domain complexity sub-metrics — **[OK]** (except $T_i$, see D4)

`project-2-domain-aware-allocation/src/complexity/domain_complexity.py:95-257`. Features are 512-d ResNet-18 avgpool activations under a frozen ImageNet backbone.

**Label entropy** (`:108-114`)

$$E_i \;=\; \frac{-\sum_{c} p_c \log_2\!\big(p_c + 10^{-10}\big)}{\log_2 n_c}, \qquad p_c = \frac{\text{count}_c}{\sum_{c'}\text{count}_{c'}}$$

Returns $0.0$ when $n_c \le 1$.

**Feature diversity** (`:129-140`) — cosine distance, subsampled to $n = \min(200, N)$

$$D_i \;=\; \min\!\left(\frac{1}{2}\cdot\frac{2}{n(n-1)}\sum_{j<k}\big(1 - \cos(f_j, f_k)\big),\;\; 1\right)$$

The $\tfrac12$ maps cosine distance $\in [0,2]$ into $[0,1]$. Note the subsample uses `np.random.choice` **without a seed**, so this metric is not reproducible run-to-run.

**Intrinsic dimensionality** (`:156-166`)

$$I_i \;=\; \frac{\min\{\,k \;:\; \sum_{l \le k} \lambda_l \ge 0.95\,\}}{d}, \qquad d = 512$$

with $\lambda_l$ the PCA explained-variance ratios. Returns $0.0$ if $\min(N, d) < 2$.

**Task difficulty** (`:209-234`)

$$T_i \;=\; \operatorname{clip}\!\left(\big(1 - \operatorname{acc}_{\text{probe}}\big)\cdot\frac{\log n_c}{\log 10},\;\; 0,\; 1\right)$$

where $\operatorname{acc}_{\text{probe}}$ is the accuracy of a fresh `nn.Linear(512, n_c)` head trained for **50 full-batch Adam steps at lr 0.01** and evaluated **on the same samples it was trained on**.

**Data imbalance** (Gini) (`:249-257`) — with sorted counts $x_{(1)} \le \dots \le x_{(n)}$

$$B_i \;=\; \operatorname{clip}\!\left(\frac{2\sum_{k=1}^{n} k\,x_{(k)} \;-\; (n+1)\sum_{k} x_k}{n \sum_{k} x_k},\;\; 0,\; 1\right)$$

### A5. Composite complexity score — **[DEGENERATE — see D5]**

`project-2-domain-aware-allocation/src/complexity/domain_complexity.py:292-304`

$$\Phi_i \;=\; \operatorname{clip}\!\Big(\big(0.3\,E_i + 0.2\,D_i + 0.2\,I_i + 0.2\,T_i + 0.1\,B_i\big)\cdot 1.2^{\,\mathbb{1}[B_i > 0.8]},\;\; 0,\; 1\Big)$$

Weights from `DEFAULT_WEIGHTS` (`:25-31`). Note $B_i$ enters twice — once as a weighted term, once as the penalty trigger.

### A6. Complexity → rank — **[DEGENERATE]**

`project-2-domain-aware-allocation/src/allocation/rank_allocator.py:79-81`

$$r_i \;=\; \operatorname{clip}\!\big(q_{\mathcal{R}}\big(16\,(1 + 3\Phi_i)\big),\;\; 4,\;\; 64\big), \qquad \mathcal{R} = \{4, 8, 16, 32, 64\}$$

Order of operations is snap-then-clamp (`:80` then `:81`).

### A7. Budget-constrained weighted assignment — **[OK, but signal-starved]**

`project-2-domain-aware-allocation/src/allocation/weighted_assignment_policy.py:40-79, 82-102, 251-284`

$$\sigma_i \;=\; \max\!\big(\lambda_c \Phi_i + \lambda_e E_i + \lambda_b B_i,\;\; 0\big), \qquad (\lambda_c, \lambda_e, \lambda_b) = (1.0,\; 0.5,\; 0.5)$$

$$w_i \;=\; \frac{\sigma_i}{\sum_j \sigma_j} \qquad \left(\text{or } w_i = \tfrac{1}{N} \text{ if } \textstyle\sum_j \sigma_j \le 0\right)$$

$$\tilde r_i \;=\; r_{\min} + w_i\big(B_{\text{tot}} - N r_{\min}\big), \qquad r_i \;=\; q_{\mathcal{R}}(\tilde r_i)$$

then greedy budget correction (`:105-187`) adjusts ranks up/down by weight order until $\sum_i r_i = B_{\text{tot}}$, respecting $r_i \ge r_{\min}$. Raises `ValueError` if $B_{\text{tot}} < N r_{\min}$.

This is a correct simplex allocation under a hard budget — the cleanest allocation mathematics in the repository.

### A8. LoRA scaling — **[OK]**

`project-2-domain-aware-allocation/src/allocation/weighted_assignment_policy.py:296`, `dynamic_allocation_policy.py:12-21`, applied at `src/models/lora_resnet.py:31,45`

$$\text{scale}_i \;=\; \frac{\alpha_i}{r_i}, \qquad \text{forward:}\quad y = W_0 x + b + \frac{\alpha}{r}\big(x A^\top B^\top\big)$$

Experiments 04–10 use a global $\alpha = 64$; `configs/default_config.yaml:12` sets `lora_alpha: 32` and is used by the oracle search — so oracle ranks were found at $\alpha=32$ and evaluated at $\alpha=64$.

---

## B. Weighting

These are **all** the weighting formulas that exist in code.

### B1. Quality-weighted aggregation (P1) — **[OK]**

`project-1-adaptive-rank/Federated/client.py:40-59`, `Federated/fedavg_aggregation.py:9-14`

$$q_i \;=\; \frac{1}{1 + \bar\ell_i}, \qquad \bar\ell_i = \frac{1}{\min(5, |\mathcal{L}_i|)}\sum_{b=1}^{\min(5,|\mathcal{L}_i|)} \ell_{i,b}$$

$$\alpha_i \;=\; \frac{n_i\, q_i}{\sum_j n_j\, q_j} \qquad \left(\text{or } \alpha_i = \tfrac{1}{N} \text{ if the denominator} \le 0\right)$$

### B2. Sample-weighted aggregation (P2) — **[OK]**

`project-2-domain-aware-allocation/src/federated/hetero_fedavg.py:70-71, 102`

$$\alpha_i \;=\; \frac{n_i}{\sum_j n_j}$$

### B3. Gossip mixing (P3) — **[DEFECT — see D3]**

`project-3-hierarchical-gossip/src/federated/gossip.py:88-96, 119`

$$\pi(i) \sim \operatorname{Unif}\big(\mathcal{N}(i)\big), \qquad A_i \leftarrow \tfrac{1}{2}A_i + \tfrac{1}{2}A_{\pi(i)}, \qquad B_i \leftarrow \tfrac{1}{2}B_i + \tfrac{1}{2}B_{\pi(i)}$$

applied simultaneously from a snapshot of all states taken before the round.

### B4. Observation

**No implemented weight depends on domain, cluster, task similarity, or peer utility.** The three formulas above are: dataset size, dataset size × inverse loss, and the constant $\tfrac12$. Grep-verified: no affinity matrix, no softmax over peers, no temperature parameter, no transfer matrix, no doubly-stochastic construction anywhere in the repository.

---

## C. Merge operator

Implemented independently in two projects with the same structure.

`project-1-adaptive-rank/Federated/fedavg_aggregation.py:33-53, 81-85`
`project-2-domain-aware-allocation/src/models/lora_resnet.py:138-189`

$$\Delta W \;=\; \sum_i \alpha_i\, B_i A_i, \qquad U\Sigma V^\top = \operatorname{SVD}(\Delta W)$$

$$B_{\text{new}} \;=\; U_{:,\,:r}\,\Sigma_{:r}^{1/2}, \qquad A_{\text{new}} \;=\; \Sigma_{:r}^{1/2}\, V^\top_{:r,\,:}, \qquad r = \min(r_{\text{target}}, |\Sigma|)$$

Zero-padded to $r_{\text{target}}$ when $r < r_{\text{target}}$ (P1 `:47-51`).

**P2 additionally applies a scale correction** (`lora_resnet.py:186-189`):

$$B_{\text{new}} \leftarrow \frac{B_{\text{new}}}{\sqrt{\alpha/r}}, \qquad A_{\text{new}} \leftarrow \frac{A_{\text{new}}}{\sqrt{\alpha/r}}$$

so that the forward pass's $\frac{\alpha}{r}BA$ reproduces $\Delta W$ exactly. This correction is mathematically right.

**Why this operator matters.** $\Delta W$ has shape $(d_{\text{out}} \times d_{\text{in}})$ independent of rank, so heterogeneous-rank clients become comparable; and $\Delta W$ is invariant to the gauge freedom $(B, A) \mapsto (BQ^{\top}, QA)$ for invertible $Q$, which factor-space averaging is not. By Eckart–Young the truncated SVD is the optimal rank-$r$ approximation of the aggregate. **This is the single most reusable asset in the repository.**

---

## D. Defects in the implemented math

### D1. The rank equation annihilates its own signal at the top capability tier

In A3, when $c_i = 1$ the floor equals $R_i^{\max}$. Since the demand term is capped at $R_i^{\max}$:

$$\max\big(R_i^{\max},\; \min(s(G), R_i^{\max})\big) \;=\; R_i^{\max} \qquad \forall\, s(G) \in [0, \infty)$$

The gradient measurement is computed and then discarded for that client. More generally the pathology occurs whenever $\operatorname{floor}_i = R_i^{\max}$.

**Empirical confirmation.** `project-1-adaptive-rank/result/exp1/federated_lora_summary.csv` reports "Average Adaptive Rank = 7.33" for all five tasks. $7.33 = 22/3$ and $22 = 2 + 4 + 16$ — each client's *minimum* attainable rank. Any single upward excursion across the 15 (5 rounds × 3 clients) entries would give $112/15 = 7.47$. So rank was constant at $(2, 4, 16)$ for every round of every task: **the adaptive mechanism produced a lookup-table constant.**

**Minimal fix:** make the capability term a soft prior rather than a hard floor, e.g. $\operatorname{floor}_i = \max(r_{\min},\, \gamma\, c_i R_i^{\max})$ with $\gamma < 1$, so the demand term can bind.

### D2. Rank projection discards the singular values

`project-1-adaptive-rank/rank_allocation/LoRa_rank_projection.py:38-49` unpacks the SVD as `_, _, Vh` and returns

$$\operatorname{proj}(t, r) \;=\; V^\top_{:r,\,:}$$

Not $\Sigma_{:r} V^\top_{:r,:}$. Because $V^\top$ has orthonormal rows by construction, **the output has unit-norm rows regardless of the input's magnitude — all scale information is destroyed.** The docstring claims "SVD truncation to target_rank principal components," but principal components require $\Sigma$.

**When it fires:** `load_global_state` calls this on every LoRA key whose shape differs from the local model's. In adaptive mode the global state is held at `FIXED_RANK = 32` while clients train at ranks 2/4/16, so **every client hits the compression branch on every download of every round.** Both $A$ (rank on dim 0) and $B$ (rank on dim 1) are affected, so the reconstructed $BA$ has no meaningful magnitude.

This is a plausible material contributor to the −0.5 to −18.35 point accuracy deficit currently attributed to "lower rank."

**Minimal fix** — capture and reapply $\Sigma$:

```python
_, S, Vh = torch.linalg.svd(mat, full_matrices=False)
compressed = S[:target_rank, None] * Vh[:target_rank, :]
```

**Correct fix** — do not project $A$ and $B$ independently at all. Reconstruct $\Delta W = BA$, then refactor with `_factorize_delta`, which is already implemented correctly in the same project (`fedavg_aggregation.py:33-53`).

### D3. Gossip mixing is not mass-preserving, and averages incompatible gauges

Two independent problems in B3.

**(a) Not doubly stochastic.** Each client averages toward one neighbor of its own choosing, so the induced mixing matrix $W$ satisfies $\sum_j W_{ij} = 1$ (row-stochastic) but in general $\sum_i W_{ij} \ne 1$. The network mean $\frac{1}{N}\sum_i \Delta W_i$ is therefore not preserved across a round. Standard gossip convergence results require $W$ doubly stochastic and symmetric with spectral gap $1 - |\lambda_2(W)| > 0$.

**(b) Factor-space averaging.** For invertible $Q$, the pairs $(B, A)$ and $(BQ^\top, QA)$ represent the identical update $\Delta W$. Averaging factors across clients whose bases have drifted into different gauges does not average their models:

$$\tfrac{1}{2}(B_1 + B_2)\cdot\tfrac{1}{2}(A_1 + A_2) \;=\; \tfrac{1}{4}\big(B_1A_1 + B_2A_2 + \underbrace{B_1A_2 + B_2A_1}_{\text{cross terms}}\big) \;\ne\; \tfrac{1}{2}\big(B_1A_1 + B_2A_2\big)$$

It also cannot be applied at all when $r_1 \ne r_2$ (non-conformable shapes). The correct operator is already available in §C.

### D4. `task_difficulty` is a constant

The probe in A4 is trained and evaluated on the same samples, and the result is scaled by $\log n_c / \log 10 > 1$ for $n_c > 10$ before clipping to $[0,1]$. For $n_c = 20$ the expression saturates at 1.0 whenever $\operatorname{acc}_{\text{probe}} \le 0.2314$.

Measured: $T_i = 1.0$ for **all 15 clients** (`results/experiment_01_data_and_complexity/complexity_scores.json`). A 0.2-weighted term contributes a fixed $+0.2$ offset to every score and carries zero information. Experiment 07 records `"task_difficulty": {"note": "constant metric, correlation is undefined"}`.

### D5. The complexity signal has no dynamic range → both allocation policies collapse

Measured $\Phi_i \in [0.6004,\, 0.6224]$ across all 15 clients — a spread of $0.0220$.

Propagating through A6: $16(1 + 3 \times 0.613) = 45.4$, which snaps to 32 for every client → **rank 32 × 15**.

Propagating through A7 with $N = 15$, $B_{\text{tot}} = 240$, $r_{\min} = 4$: $\tilde r_i = 4 + 180 w_i$, and $w_i \approx 1/15$ gives $\tilde r_i \approx 16$ for all $i$. Measured raw ranks: $[15.691,\, 16.172]$ → **rank 16 × 15**.

The formulas are correct; the input signal cannot differentiate. Any normalized weight vector that is near-uniform yields a near-uniform allocation by construction. Fixes are either a signal with genuine variance, or a sharpening transform such as $w_i \propto \exp(\sigma_i / \tau)$ with small $\tau$, or rank-order rather than proportional allocation.

Independently, the premise is unsupported: complexity vs. oracle rank gives Spearman $\rho = -0.0349$ ($p = 0.902$), $R^2 = 0.0688$ (`results/experiment_03_correlation_analysis/experiment03_log.txt`).

### D6. P2 aggregates pre-scaling quantities under heterogeneous ranks

`lora_resnet.py:156` merges $\Delta W_i = B_i A_i$, but client $i$'s forward pass applies $\frac{\alpha}{r_i} B_i A_i$. With $\alpha$ held constant and ranks differing, the true effective updates differ from the merged quantities by $\alpha / r_i$ — a factor-16 spread between $r = 4$ and $r = 64$ on the standard menu — while being averaged as though equal.

Harmless at uniform rank (all runs to date), incorrect the moment ranks differ. The operator should be defined on the **scaled** update:

$$\Delta W_i \;=\; \frac{\alpha_i}{r_i} B_i A_i$$

Related: because $(\alpha, r)$ enter the forward pass only through $\frac{\alpha}{r}BA$, $\alpha$ is not identifiable independently of $r$ and the learning rate. Holding $\alpha_i / r_i$ constant across clients prevents heterogeneous ranks from silently inducing heterogeneous effective learning rates — otherwise any rank-heterogeneity experiment is confounded by an optimizer artifact.

---

## E. Absent from code

Grep-verified across all `.py` files in all three projects — **zero occurrences**, i.e. no mathematics exists for any of the following:

Convergence-speed monitoring or loss-history tracking · any neighbor-derived quantity in P1 or P2 · domain drift detection (KS or otherwise) · network congestion / bandwidth / communication-cost terms in any rank decision · fairness metrics (Gini over accuracy, accuracy gap) in P1 · multiplicative rank updates $r \leftarrow 2r$ or $r \leftarrow r/2$ · resource profiling (GPU memory, compute capability, CPU, RAM) in P2 · any hardware-derived rank ceiling · affinity or similarity matrices over peers · softmax or temperature over peers · cross-domain transfer matrices · doubly-stochastic projection (Sinkhorn or Metropolis–Hastings) · spectral gap computation · any convergence bound, assumption set, or proof.

`compute_communication_cost` exists at `project-2-domain-aware-allocation/src/utils/metrics.py:81-95` but is never called.

---

## F. Summary

**Rank allocation: the mathematics exists and is reusable.** Two independent demand signals — $s(G)$ from gradient geometry (A1) and $\Phi_i$ from data statistics (A5) — plus a capability term (A2), a menu quantizer, and a budget-constrained simplex allocator (A7). Usable after fixing D1 (one line) and supplying a signal with variance (D5). The *fusion* of $s(G)$ and $\Phi_i$ into a single demand term is not implemented in either project.

**Weighting: the mathematics does not exist.** Every implemented weight is a dataset-size proportion, an inverse-loss quality score, or the constant $\tfrac12$ (B1–B3). No formula in this repository accepts domain membership, cluster identity, or peer utility as input.

**Merge operator: implemented, sound, and the key asset** (C) — modulo the scaling correction of D6.

**Structural note.** Server-side aggregation weights need only satisfy $\sum_i \alpha_i = 1$ — one row summing to one — which is all B1 and B2 ever required. Decentralized mixing requires the full matrix $W$ to be doubly stochastic and symmetric with positive spectral gap. That is a strictly stronger constraint, it cannot be satisfied by porting B1 or B2, and no construction meeting it exists in the repository. The composition that would meet it —

$$a_{ij} = \operatorname{affinity}(i,j), \qquad \hat W_{ij} \propto \exp(a_{ij}/\tau)\,\mathbb{1}\big[j \in \mathcal{N}(i) \cup \{i\}\big], \qquad W = \operatorname{Sinkhorn}(\hat W), \qquad W_{ii} \ge w_{\min}$$

— is unwritten, and is the mathematical core of the Project 3 contribution.
