# Quantity-skew decentralized LoRA: prospective comparison

Protocol version 1, 2026-09-17. This protocol is recorded before full-budget
results. All scientific execution is on `gpu003`. Training uses an isolated
source snapshot; each run archives its actual source and SHA256 manifest.

## Question and comparator

Can clients with heterogeneous adapter-rank limits and unequal quantities of
data from the **same task** train an effective adapter through decentralized
communication? Is adaptive rank plus sample-size weighting competitive with
published decentralized factor gossip, at what communication and memory cost?

The selected publication is Ghiasvand, Alizadeh and Pedarsani,
*Decentralized Low-Rank Fine-Tuning of Large Language Models*, REALM 2025,
ACL Anthology [2025.realm-1.24](https://aclanthology.org/2025.realm-1.24/).
The published July 2025 PDF is the reference; later arXiv v5 has changed scores.
The literature/source audit is in
[`2026-09-17-paper-baseline-shortlist.md`](2026-09-17-paper-baseline-shortlist.md).
No verified author implementation was found. Our baseline is therefore a
**paper-based independent reimplementation**, not an exact numerical replication.

Dec-LoRA averages A and B separately on graph edges. Our product-space control
and proposed variants average effective updates and truncate by SVD. This is
closely related to existing DeCAF (arXiv:2505.21382); effective-product averaging
is not claimed as new. Sample weights are standard empirical-risk weights, not
new domain-allocation theory. Any contribution must be supported by measured
behavior of the combined rank/weight/peer protocol.

## Pinned assets and paper reconstruction

| Item | Fixed setting | Relationship to publication |
|---|---|---|
| Task | GLUE SST-2; accuracy | In main published comparison |
| Model | FacebookAI/roberta-base, revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b` | Published model family/size |
| Data | nyu-mll/glue, revision `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c` | Canonical public dataset |
| Split | All 67,349 training rows, official 872-row labeled validation | Paper table instead lists 66,675/674; unexplained difference disclosed |
| Clients/graph | 10, ring, self/left/right proposal weights 1/3 | Published main ring comparison |
| Baseline rank | 16; extra resource-feasible rank4 control | Published rank16 main comparison; rank4 is a separate control |
| Local batch, LR | 32, constant 1e-3 | Published values |
| Equal-size anchor | 20 rounds, one local epoch per round | Interprets ambiguous K=1 as one local epoch; not asserted as author's exact code |
| Quantity extension | 20 rounds, 211 local updates of exactly 32 examples per peer/round | Explicit controlled extension; approximately anchor's work, equal across peers |
| Target modules | Query/value in all 12 blocks, 589,824 LoRA parameters at r16 | Fits paper's approx 0.60M, but module list is an assumption |
| Classifier | Train and communicate complete RoBERTa classification head | Not specified by paper; counted separately (~592k parameters) |
| Alpha/init | alpha16; A common N(0,0.02^2), B=0 | Normal/zero construction specified; scale/std not specified |
| Token length | Maximum128, dynamic batch padding, no changed text | Unspecified paper detail |
| Optimizer | AdamW, decay0, clip norm1, reset for each peer/round in **every** arm | Unspecified paper detail; common reset avoids stale SVD-coordinate moments |
| Dropout | Pretrained model's configured dropout; adapter dropout0 | Unspecified paper detail |
| Precision | fp32, deterministic Torch algorithms | Explicit implementation choice |
| Evaluation | Assembled adapter each round; best validation accuracy primary, final accuracy secondary | Paper uses best validation accuracy of averaged models |

The official unlabeled SST-2 test is not used. Repeated labeled-validation
evaluation follows the paper's selection endpoint; these are validation
results, not a hidden-test generalization estimate. No score from our changed
partition may be presented as a direct win over the paper's printed 93.81%.

## Quantity-only split and computation budget

Use the size ratio vector `[1,1,1,1,2,2,2,4,4,4]`. Largest-remainder allocation
sets exact integer client totals. Bipartite integer rounding makes every
client/class count a floor or ceiling of its proportional quota, with exact
row/column totals. Persist full memberships, label counts, hashes, and coverage.
Clients are disjoint and exhaust the official training set.

The partition seed is `seed+10000`, quantity-to-peer assignment seed
`seed+20000`, and independently shuffled capacity assignment seed `seed+30000`.
Capacity tiers are `[4,4,4,4,8,8,8,16,16,16]`. Peer IDs define ring position;
quantity and capacity permutations are independent and matched across arms.

The quantity extension streams independently reshuffled **local** data across
epoch boundaries to perform 211 updates of batch32 at every peer each round.
This gives exactly 1,350,400 training-example exposures over20 rounds in each
arm. A full20-epoch pass over SST-2 would give1,346,980. Smaller clients repeat
their local examples more often; no cross-client examples are introduced.
Adaptive probes add training-only work, measured separately. Equal local
updates avoid confounding sample weights with the larger number of local
optimizer steps that full unequal-size epochs would produce.

## Methods and equations

For client i, local loss and the sample-weighted objective are

\[
F_i(\theta)=\frac{1}{n_i}\sum_{z\in D_i}\ell(\theta;z),\qquad
\pi_i=\frac{n_i}{\sum_jn_j},\qquad F(\theta)=\sum_i\pi_iF_i(\theta).
\]

This is the intended statistical weighting, not a claim that finite local
Adam steps and rank projections exactly follow centralized optimization.
Static client quantities produce **static sample-size weights**.

Let Q have ring self/neighbor entries1/3. Sample-weighted gossip uses

\[
P_{ij}=Q_{ij}\min(1,\pi_j/\pi_i)\ (i\ne j),\qquad
P_{ii}=1-\sum_{j\ne i}P_{ij}.
\]

The kernel obeys detailed balance and preserves pi. Uniform variants use
pi_i=1/10, hence P=Q. Counts are exchanged once between neighbors. Weighting
also changes mixing speed; log its reversible spectral gap and weighted
effective-update disagreement. This is not a pure objective-only ablation.

The faithful Dec-LoRA operation is

\[
A_i^+=\sum_jP_{ij}A_j,\qquad B_i^+=\sum_jP_{ij}B_j.
\]

Our effective-state operation, used for both fixed/adaptive internal controls,
is

\[
X_i=\frac{\alpha}{r_i}B_iA_i,\quad
Y_i=\sum_jP_{ij}X_j,\quad X_i^+=\mathcal P_{r_i}(Y_i).
\]

Weighted concatenated factors are decomposed by thin QR and a core SVD, giving
the same best-rank approximation as a dense SVD without forming dense768x768
updates at every receiver. Independent dense-oracle tests cover scaling,
different ranks, weights, gauge changes, truncation, and zero contributions.
The linear classification head is averaged with the same weights.

Adaptive clients use the existing P1 stateful controller: capability-bounded
candidate ranks, two warmup rounds, half-capability minimum, EMA/hysteresis,
and quality-drop restoration. Its demand is the median stable rank of A/B
gradients on one fixed training batch at the client's capability ceiling:

\[
\operatorname{srank}(G)=\frac{\|G\|_F^2}{\|G\|_2^2},\qquad
r_i^{\rm target}=\operatorname{round}_{\mathcal R_i}
\max\{2,\gamma c_iR_i^{\max},\min(s_i,R_i^{\max})\},\quad\gamma=0.5.
\]

Probe quality is1/(1+training cross entropy); no validation labels enter rank
decisions. Factor-gradient stable rank is coordinate-dependent; it is not
assumed to identify the required global model rank. Ceiling-rank probes are
charged as additional work and memory. Expanding rank preserves alpha/r scaling
and initializes new A rows with nonzero random values and new B columns at zero
so new coordinates can learn. The controller can reduce persistent rank but
cannot claim peak memory below the capability probe.

| Arm | Training rank | Mixer | Scientific role |
|---|---|---|---|
| declora16 | Uniform16 | Paper factor gossip, uniform | Literature-based unconstrained comparator |
| declora4 | Uniform4 | Paper factor gossip, uniform | Fits all clients' rank ceilings |
| product16_sample | Uniform16 | Effective product, sample weights | Unconstrained mechanism control related to DeCAF |
| fixed_uniform | At heterogeneous caps | Effective product, uniform | Factorial reference |
| fixed_sample | At same caps | Effective product, sample weights | Weighting effect |
| adaptive_uniform | P1 within same caps | Effective product, uniform | Rank adaptation effect |
| adaptive_sample | P1 within same caps | Effective product, sample weights | Proposed combined method |

The four heterogeneous arms form a2x2 ablation. All seven quantity arms use
identical split, graph, per-peer sample stream, optimizer convention, and work
budget. Rank16 baselines exceed weaker peers' declared training caps and are
labeled accordingly; matching them is a stronger accuracy question, not a
resource-feasible baseline claim. These rank caps constrain adapter state,
not total device memory by themselves.

## Assembly, execution, and resource scope

Each nonlocal training contribution is serialized, transferred on an actual
declared simulation edge, decoded, and recorded in a byte/hash ledger. Final
assembly gathers factor records along a BFS spanning tree and assembles at the
first capacity16 peer (chosen before training). The effective adapter is

\[
X_{\rm global}=\mathcal P_{16}\left(\sum_i\pi_iX_i\right).
\]

Baseline evaluation averages its factors; the rank4 baseline deploys at rank4.
The output is not broadcast back to low-capacity peers. Evaluation assemblies
never alter training state. Their costs are reported separately; production
cost includes count setup, training exchange, and the final assembly once.

Execution is a single-process decentralized transport simulation on gpu003,
not a multi-machine network deployment or a cryptographic privacy experiment.
The assembly peer sees all routed adapter records. Log actual serialized tensor
and metadata bytes, classifier traffic, root/inbox storage, structural QR/SVD
workspace estimates, and measured whole-process CUDA peaks. Whole-process
peaks include the shared backbone, gradients, ceiling probes, and deployment
adapter; they are not measured independent-client peaks. Raw data stay in
logical client shards, but no leakage protection or privacy guarantee is claimed.

## Finite experimental ladder and decision rule

1. Remote partition/merge/transport/model tests. Failure stops dependent runs.
2. Matched real RoBERTa/SST-2 smoke runs: bounded training subset and steps;
   no scientific efficacy conclusion. Check saved-checkpoint predictions in a
   fresh process with a separate metric implementation.
3. One-seed full-budget screen (seed42) of the seven quantity arms and one
   equal-size Dec-LoRA anchor. Preserve failed attempts and runtime failures.
4. Freeze any code fixes before replication. Full seed set42,43,44,45,46 for
   the seven retained quantity arms, with seed42 retained if protocol unchanged.
   Compare paired differences, sample SD and uncertainty; do not select seeds.

Primary comparison is adaptive_sample minus declora16 best validation accuracy
at the fixed budget, with declora4 as the feasible comparator. Report final
accuracy, all internal ablations, exposure counts, and byte/work tradeoffs too.
One seed is exploratory. A nonsignificant difference does not establish parity;
no post-hoc noninferiority margin will be invented. Negative outcomes complete
the study; the split will not be tuned until the proposed method wins.

The new results do not overwrite the earlier disjoint-domain negative result.
Before any scientific success claim, require complete matched runs, invariant
checks, checkpoint verification, honest resource accounting, and source-pinned
artifacts pushed to GitHub.
