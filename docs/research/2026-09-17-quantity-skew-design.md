# Quantity-skew decentralized LoRA: independent design audit

Date: 2026-09-17. This is a prospective design and read-only code audit, not a new experiment result. All scientific validation and training must execute on the authorized `gpu003` SSH machine. Historical negative results remain part of the record.

## Research question and terminology

The new question is whether resource-constrained clients with different amounts of data, drawn from the same task and approximately the same class distribution, can train and assemble an accurate LoRA adapter using peer communication, heterogeneous/adaptive ranks, and sample-count weighting. It removes the previous disjoint-domain label ownership; it does not by itself remove repeated rank-projection loss.

Use **quantity skew**, **adaptive rank**, and **sample-size aggregation** in this experiment. When client datasets do not change, weights proportional to dataset size are static, not dynamic domain weights. They are standard empirical-risk/FedAvg weighting, not a new weighting algorithm. A defensible contribution would concern the combination, a better feasible aggregation protocol, or a measured accuracy/resource tradeoff against a reproduced decentralized baseline.

The primary literature comparison must name a real paper and pin its public implementation, model, data, protocol, and endpoint before running the comparison. A method using the paper's dataset but a different backbone, adapter placement, preprocessing, or train budget is an adaptation benchmark, not a reproduction of the paper's reported accuracy.

## Required two-track comparison

1. **Published-protocol reproduction:** reproduce the chosen paper's dataset, split, model/checkpoint, adapter sites, topology, number of clients, optimizer, local work, communication budget, and evaluation endpoint. Record every unavoidable deviation. Run our method under the same setting and explicitly name any changed rank/resource setting.
2. **Controlled quantity-skew extension:** run both the reproduced baseline and our method on exactly the same newly defined unequal-size, class-stratified partition. Compare against our fixed-rank and unweighted controls. Do not compare our easier quantity-only partition with the paper's published non-IID number.

If the paper's code or model cannot run on available resources, document the precise obstacle and select a reproducible alternative before presenting a competitive result. An existing cached ResNet-18-feature classification-head experiment is useful as an inexpensive mechanism screen, but cannot substitute for a paper's full-model/ViT/LLM benchmark.

## Exact partition and objective

Start from the official training split; remove a fixed stratified validation split if the source protocol does not already supply one. Never allocate official test examples to training or the rank controller.

Choose positive client quantity proportions `q_i` before observing results. Use a fixed unequal pattern or a seeded distribution, but persist the complete realized count vector. A useful initial pattern is repeated size tiers with a predeclared 1:2:4 ratio; an equal-size partition is a sanity control, and a more extreme ratio is a separately declared stress test. Do not choose the skew after looking at the accuracy.

Allocate exact target counts with largest-remainder rounding:

\[
 n_i=\lfloor Nq_i\rfloor + e_i,\qquad
 e_i\in\{0,1\},\quad \sum_i n_i=N.
\]

Break fractional ties with a stored deterministic seed/order. Construct an integer client-by-class allocation close to `n_i N_c/N`, with exact client row totals `n_i` and exact class column totals `N_c`. One valid implementation starts with floors and uses a deterministic bipartite residual allocation; independent per-class rounding alone does not ensure exact requested client totals. Shuffle example indices independently within each class, assign according to that matrix, then shuffle each client's indices. A single random global permutation split at the desired sizes is also a valid IID quantity-skew design, but should be named randomized IID rather than exactly stratified; rare-class omissions must then be reported.

Required split artifacts: dataset fingerprint, train/validation/test index hashes, exact per-client index lists or reconstructible seeded algorithm, quantities, class-count matrix, class-proportion divergence, coverage, overlap counts, and size-to-capacity mapping. Assert disjoint client training sets, exact training union, no validation/test leakage, no empty clients, and equality of split hashes across arms.

Randomize quantity tiers, rank-cap tiers, and graph placement independently in the primary design. Otherwise large datasets can systematically receive larger ranks or more favorable neighbors, confounding sample weighting with capability/topology. Use matched permutations across methods and record them. A correlation between data quantity and hardware capacity is a separate deployment scenario, not an implicit assumption.

With local empirical loss `F_i`, the dataset objective is

\[
 F(\theta)=\sum_i\pi_i F_i(\theta),\qquad
 \pi_i=\frac{n_i}{\sum_j n_j},\qquad
 F_i(\theta)=\frac1{n_i}\sum_{z\in D_i}\ell(\theta;z).
\]

Uniform client weights instead optimize a client-average objective. Both can be useful, but they answer different questions when data quantities differ. Multiplying sample weights by rank or local quality changes this objective; that must be a separately labeled method, not an unexplained correction to sample-size weighting.

## Decentralized mixing and adapter assembly

The existing `src/federated/domain_weights.py::weighted_metropolis` is suitable for fixed positive sample weights. For a symmetric stochastic graph-supported proposal `Q`, use

\[
 P_{ij}=Q_{ij}\min(1,\pi_j/\pi_i),\quad i\ne j;\qquad
 P_{ii}=1-\sum_{j\ne i}P_{ij}.
\]

Then `P 1 = 1`, `pi^T P = pi^T`, and `pi_i P_ij = pi_j P_ji`. Count information can be sent once at setup for static datasets. Neighbor count ratios suffice for local transition construction; global normalization is needed for reporting/final assembly, not for the acceptance ratio. Never apply Sinkhorn balancing afterward: it would change the intended stationary objective.

Changing sample weights also changes mixing speed. With severe quantity ratios, large-mass peers can have small exit probabilities; this creates a consensus bottleneck even under IID labels. Log the spectral gap using the reversible symmetric similarity transform `diag(sqrt(pi)) P diag(1/sqrt(pi))`, along with weighted disagreement `sum_i pi_i ||X_i - sum_j pi_j X_j||_F^2`. The existing runner's `consensus_distance` is deliberately an unweighted diagnostic; add a separately named field rather than silently changing historical semantics. The weighting ablation measures the combined effect of objective and mixer, not a pure change to statistical sample weighting while all optimization dynamics stay fixed.

For each LoRA layer, communicate factors but merge the effective update

\[
 X_i=\frac{\alpha}{r_i}B_iA_i,\qquad
 Y_i=\sum_jP_{ij}X_j,\qquad
 X_i^+=\mathcal P_{r_i}(Y_i).
\]

The existing `merge.py` carries the correct `alpha/r` convention. Averaging `A` and `B` separately is not a substitute: products introduce cross terms and depend on factor coordinates. Pin the source paper's own scaling convention, and distinguish a faithful baseline from an intentionally corrected aggregation variant.

Assembly is

\[
 X_{\rm global}=\mathcal P_R\!\left(\sum_i\pi_i X_i\right).
\]

The existing `peer_assembly.py::tree_weighted_assembly` implements an exact graph-restricted reduction before one final rank projection, with a predeclared deployment peer. This is peer-routed aggregation with a temporary reduction root, not a claim that the algorithm has no aggregation role anywhere. `R` and the deployment peer must be chosen before seeing test accuracy. Include setup, training, rank-control, final reduction, and final dissemination costs; label evaluation-only intermediate assemblies separately.

## Why quantity skew does not repair every failure

The code and existing no-training diagnostics establish a structural concern independent of label distribution. Under a fixed common singular basis, let `d_ik = 1[r_i >= k]` and let `c_k(t)` be the vector of peer coefficients of component `k`. Repeated mixing and projection obey

\[
 c_k(t+1)=\operatorname{diag}(d_k)P c_k(t).
\]

For complete averaging, define `p_k=sum_i pi_i d_ik`. Starting from a shared oracle adapter projected to each peer, the assembled coefficient after `t` mixing cycles is `sigma_k p_k^(t+1)`. If any positive-weight peer cannot retain that component, `p_k<1`. On a connected fixed-rank network the corresponding killed-chain recurrence also attenuates it over repeated cycles. The common-basis and no-training assumptions matter: this is a failure mechanism, not a theorem that all training with heterogeneous LoRA must fail.

Giving all clients the same class distribution can reduce gradient conflict and may substantially improve accuracy. It cannot make a missing coefficient persist in a peer that cannot store it. If larger datasets are assigned to higher-rank clients, size weighting can slow attenuation of high components, but that correlation must not be introduced silently to improve results.

Keep the original effective-model-state method as an explicitly faithful arm. Any preserved shared model, gradient-first allreduce, error-feedback residual, consensus-only common subspace, component-ownership protocol, or rank-aware renormalization is a named protocol variant. The earlier masked-gradient repair stores the full reference-rank model and optimizer state, so it cannot serve as evidence for reduced client model memory. Dense residual error feedback also consumes dense per-layer memory and must be charged.

## A finite factorial design

After the publication protocol is selected, use its mandatory settings. For the quantity-skew extension, freeze one main dataset/task and one main skew, graph, and local-work budget before confirmatory runs.

| Arm | Rank policy | Aggregation objective | Purpose |
|---|---|---|---|
| Published decentralized baseline | Faithful to paper | Faithful to paper | Actual literature comparator |
| Fixed heterogeneous / uniform | Each client at its feasible cap | Uniform client | Factorial reference |
| Fixed heterogeneous / sample | Same caps | `n_i/N` | Sample-weighting effect |
| Adaptive / uniform | Existing P1 policy within same caps | Uniform client | Rank adaptation effect |
| Adaptive / sample | Same adaptive policy and caps | `n_i/N` | Proposed combined method |

The four internal arms form a 2-by-2 comparison. Their paired differences estimate weighting, adaptation, and interaction effects; they do not alone establish superiority over the paper. Include a uniform-rank sample-weighted peer arm as a mechanism diagnostic. If that rank exceeds a low-tier client's cap, label it resource-unconstrained. A uniform rank equal to the smallest cap is a feasible alternative with a different accuracy/capacity tradeoff. A pooled reference can remain contextual, but the primary competitive claim is against the chosen decentralized baseline.

Use the same initialization, samples, graph, client capacities, batch ordering, seed set, endpoint, and tuning budget across the factorial arms. Match scaling and optimizer behavior unless the change is a declared method component. Fit hyperparameters on validation data with the same search budget for every competitor. Count failed/diverged attempts and do not overwrite them.

Fixed local steps per round cleanly isolate weighting when quantities differ. If the chosen paper uses fixed local epochs, reproduce that convention: larger clients then execute more optimizer steps and also receive larger sample weights. This is standard FedAvg-style practice, not automatically an implementation error, but the resulting trajectories are not equivalent to equal-step optimization. Report both local steps and sample exposures, and use an equal-step diagnostic if needed; do not assert that equal rounds imply equal computation.

Choose one primary endpoint before confirmatory runs: the paper's metric at its fixed budget. If the paper averages over peers, report that endpoint as well as the assembled global adapter; neither can silently replace the other. Also report accuracy versus transmitted bytes and versus local training work. When comparison claims focus on resources, use a fixed accuracy target or a matched byte/work budget specified in advance, rather than selecting a convenient point after looking at curves.

Three paired seeds are a screening minimum. Five predeclared paired seeds are preferable for a final small benchmark; retain every seed, report mean/sample SD and paired differences with uncertainty. A non-significant difference is not evidence of parity. Any practical non-inferiority margin (for example 1 percentage point for an accuracy task) needs justification and declaration before the confirmatory results. Use the actual paper metric; do not transfer an accuracy margin to perplexity or another score.

## Resource feasibility audit

The implementation currently has true variable-shaped trainable factors, but a rank cap is not yet a complete memory budget. Account for:

- Frozen backbone and activations, factors, gradients, optimizer moments, and adapter copies.
- Ceiling-rank probe model/gradients and the original resident model at the same time.
- Incoming neighbor factors, dense effective updates, SVD workspace, and merge accumulators.
- Rank-control traffic, graph/count setup, and final adapter assembly/dissemination.

`DecentralizedRunner.gossip_round` currently materializes every peer's dense effective state in a single process, which is simulation storage rather than a distributed resident-memory measurement. Individual receivers still need dense local merge/SVD workspace in the implemented algorithm. `tree_weighted_assembly` reports transport-buffer memory but excludes SVD and diagnostic work. A compact concatenation/QR/SVD implementation could avoid forming full dense matrices, but that would need its own equivalence and peak-memory validation, including the temporary concatenated rank.

Report persistent adapter memory separately from measured peak memory. Model-state rank, selected gradient coordinates, communication payload rank, and final deployment rank are distinct quantities. The method should not claim a smaller total client memory footprint merely because fewer parameters are trainable.

## Smallest decisive remote validation ladder

1. **No-training invariants:** on `gpu003`, validate exact partition unions/quantities/class counts; data leakage exclusions; `alpha/r` effective merge against an independent dense calculation; weighted-MH row sums, graph support, detailed balance, and stationarity; graph-restricted final assembly against a direct reference; caps at every rank change; rank-probe isolation from training RNG and held-out data. Failures stop dependent experiments until fixed.
2. **Paper baseline smoke:** one seed, reduced rounds/data strictly labeled as smoke, with original model/data pipeline and metric. Verify loss is finite, checkpoint reload gives the same evaluation, optimizer state policy is faithful, and a independently computed traffic ledger agrees. This is an execution check, not a published-score reproduction.
3. **One-seed full-budget screen:** run the paper baseline and the four factorial arms on one frozen quantity split; add one uniform-cap peer diagnostic. Evaluate on validation data for screening, and persist rank trajectories, objective weights, step/exposure counts, merge residuals, parameter/memory/byte accounting, checkpoints, and source hashes. A cheap cached-feature screen may precede this but cannot satisfy it.
4. **Mechanism gate:** if fixed heterogeneous ranks fail while the uniform-cap peer arm learns, run a no-training rank-retention stress test using this setting and verify that the observed loss is not an accounting/scaling bug. If both fail, first reproduce the baseline training protocol and examine local work/mixing. Limit repairs to a predeclared small set of named variants; never silently replace the proposed method mid-table.
5. **Confirmatory run:** after validation settings are frozen, run the predeclared full seed set and official held-out endpoint for every retained competitor. Commit/push code and protocol before launching. Pull and commit manifests, raw summaries, source-pinned configs, figures, and the final positive or negative result. Include the status of every launched run.

**Go:** required invariants pass; the publication baseline is faithfully executable; datasets, endpoint, graph, ranks, and budgets are explicit; all retained arms have finite outputs and verified checkpoints. A positive early accuracy result is not required to complete the study.

**Stop and fix:** wrong data union, held-out leakage, inactive weights, incorrect `alpha/r`, hidden central access outside the declared simulation, cap violations, incorrect metric, unfairly omitted traffic/memory, or unrecorded source/protocol changes.

**Stop tuning and report a negative result:** after the finite protocol and predeclared validation search finish, the method is worse at the matched scientific endpoint/resource constraints. Do not keep changing splits until a win appears. An unsuccessful but faithful controlled comparison is still a completed, useful research result.

## Source audit pointers

- `project-3-hierarchical-gossip/experiments/protocol_benchmark.py`: current `make_splits` is CIFAR-100 domain-specific; `FeatureClient.train` resets Adam each local round; rank probes use a fixed local training minibatch and a cloned ceiling-rank model.
- `project-3-hierarchical-gossip/experiments/integrated_benchmark.py`: `WeightedRunner`, sample-weight initialization, source/split hashes, actual rank scaling, and explicit final-assembly accounting are useful reusable components; the command-line driver assumes 15 clients and reference rank 16.
- `project-3-hierarchical-gossip/src/federated/domain_weights.py`: correct reversible sample-weight kernel; dynamic quality/domain policy should not be imported into the new sample-only arm.
- `project-3-hierarchical-gossip/src/federated/runner.py`: effective-state averaging followed by receiver-rank SVD; dense error feedback is optional and currently off.
- `project-3-hierarchical-gossip/src/federated/peer_assembly.py`: explicit peer-tree reduction, dense payloads, temporary root, and simulation limitations.
- `project-3-hierarchical-gossip/experiments/diagnose_rank_retention.py`: independently checked common-basis attenuation recurrence; a no-training diagnostic, not training success evidence.
- `project-3-hierarchical-gossip/experiments/diagnose_optimization.py`: prior randomized IID control preserves client sizes but retains old domain-oriented evaluation fields; use a new explicit quantity-skew manifest rather than relabeling those fields.

No tests or new experiments were run for this audit. Selection of the actual paper and its concrete settings belongs in the literature/protocol record before execution.
