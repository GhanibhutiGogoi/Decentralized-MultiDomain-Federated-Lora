# Project 3 — Hierarchical Gossip Aggregation: Research Framing

*Ghanibhuti's project line within AH-LoRA. Written 2026-07-20 after a full audit of the repo (Projects 1–3, `Project.md`, the three proposal PDFs, `paper/main.tex`) and a fact-checked literature survey (deep-research run, 13+ primary sources; novelty map at the end). Companion implementation plan: `docs/superpowers/plans/2026-07-20-project3-affinity-hierarchical-gossip.md`.*

---

## 1. Problem statement

N clients, no server. Each client holds data from one of K **undeclared** domains (nobody is told K or their domain), has heterogeneous compute (→ heterogeneous LoRA ranks r_i on a shared frozen backbone), and communicates only with graph neighbors. Goal: every client ends with a personalized adapter that gains from collaboration — especially within its own (undiscovered) domain — without cross-domain interference, at bounded communication, with a convergence guarantee.

Formally, per the unified paper (`paper/main.tex`): minimize Σᵢ fᵢ(W₀ + ΔWᵢ) subject to rank(ΔWᵢ) ≤ rᵢ, communication restricted to a graph G, with two evaluation protocols reported for every method — **personalized** (each client's model on its own test shard) and **consensus** (one merged adapter on the full test distribution).

## 2. Where the project actually stands (evidence from the repo)

1. **What exists in P3**: flat, domain-blind gossip (`src/federated/gossip.py`) — each round every client averages toward **one random neighbor** with fixed 0.5/0.5 weights, applied to the A and B factors **separately**; homogeneous rank 16; LoRA on the fc layer only; 2-round single-seed smoke tests.
2. **Three protocol defects** that would invalidate any claim made on top of it:
   - *Factor-space averaging* is exactly the operation `main.tex` Proposition 1 declares ill-posed (gauge ambiguity: (B,A) and (BQᵀ, QA) encode the same ΔW; averaging factors mixes incompatible gauges and creates cross terms). FLoRA (arXiv 2409.05976) independently proves the same defect for server-side FL.
   - *One-sided random-neighbor averaging* yields a row-stochastic but not doubly-stochastic update — it does not preserve the network's mean adapter, and it breaks the spectral-gap assumption the paper's own convergence sketch requires.
   - *Personalized-vs-consensus conflation*: experiment 02's gossip (personalized eval) was compared against FedAvg (consensus eval); the paper's methodological correction demands both protocols for every method.
3. **The core-hypothesis experiment failed**: experiment 03 clustered clients by singular-value spectra of A, B, BA and got ARI 0.031 → **−0.140**, *worsening* as adapters specialize, under local-only training (the most favorable condition — no mixing noise). NMI 0.45 → 0.29.
4. **The merge primitives Project 3 needs already exist centrally**: P1's `fedavg_aggregation.py` and P2's `hetero_fedavg.py` both aggregate in ΔW space with truncated-SVD refactorization. Nothing in P3 uses them yet.
5. **P1/P2 hand-offs are real but imperfect**: P1's rank rule works but always trades accuracy for FLOPs (−0.5 to −18 pts at ~77% FLOPs saved); P2's complexity signals do not predict oracle rank (Spearman ρ ≈ −0.03) and both learned allocation policies collapse to uniform ranks ("uniform collapse"). Consequence: Project 3 must be robust to *imperfect* rank allocations, and its heterogeneous-rank experiments should treat allocations as **given inputs** (any spread at matched budget), not as trusted signals.

## 3. Diagnosis of the clustering failure (the pivotal reasoning)

Why did spectral signatures fail where the whole "cluster-then-gossip" pipeline expected them to work?

- The features (`domain_clustering.py:44-60`) are singular-value **magnitudes** — they say *how much* the adapter changed in its top directions, but not *which* directions or *which classes*. All 15 clients share the same init (cloned base model, B=0), the same optimizer, similar data volumes: their spectra evolve almost identically, so the feature vectors converge as training progresses — which is precisely the observed ARI decay.
- On an fc-only LoRA (ΔW is 100×512), domain identity is *structurally* encoded in the **row space**: a client's 20 in-domain classes are the rows of ΔW that accumulate mass. Because B is zero-initialized, early gradients flow through B against a shared A, so divergence across clients concentrates in B's column space = class space. Direction-aware features should separate domains almost trivially in this testbed; spectrum-shape features cannot.
- **Independent corroboration (literature)**: Listo Zec et al.'s decentralized-similarity study (follow-up to DAC) found that with a **pre-trained ResNet-18 fine-tuned on CIFAR-100 superclass clusters** — our exact setting — inverse-loss, cosine-on-weights, and cosine-on-gradients all failed to beat random peer selection (pre-training makes scores uniform across clients); only inverse-L2 weight distance recovered cluster structure. Our exp03 failure is thus a *known phenomenon class*, and our proposed fix (signatures on **ΔW only**, which excludes the shared pretrained mass by construction) attacks its documented cause.

This reframes RQ1 from "domains are not discoverable" (what the paper currently says) to "domains are not discoverable *from spectra*" — an empirically distinguishable and much more publishable claim, whichever way exp05 resolves it.

## 4. Hypotheses

**H1 — Merge operator.** In decentralized gossip, merging scaled effective updates ΔWᵢ = (α/rᵢ)BᵢAᵢ (then refactorizing to each client's own rank via truncated SVD) dominates (i) separate A/B factor averaging and (ii) HetLoRA-style zero-padding, with the gap **growing in rank heterogeneity and training length** (gauge drift accumulates; at homogeneous rank from shared init the difference may be ≈ 0 — a null there does not refute H1). *Effect-size prior from literature:* FLoRA reports zero-padding collapsing 29.5% → 7.97% MMLU vs ΔW-exact aggregation in the server setting.

**H2 — Domain discovery.** Serverless domain discovery succeeds when signatures encode *directions or function*, not spectra. Signature ladder (cheap → rich): S1 per-class row-norm profile of ΔW; S2 top-k right-singular subspace of ΔW compared by principal angles; S3 logits on a small shared unlabeled probe set; S4 cross-evaluation of received adapters on local held-out train data (zero extra messages during gossip); S5 inverse-L2 distance between ΔWs (the literature's sole survivor on pretrained backbones). Prediction: S1/S5 reach ARI ≥ 0.7 by round ~5–10 on the current testbed; spectra stay ≈ 0. Secondary prediction: separability survives *mid-gossip* (mixing does not erase it faster than local training re-creates it) — a necessary condition for any clustering-dependent protocol.

**H3 — Hierarchy/topology.** Given usable affinities, two-tier gossip (dense intra-cluster mixing + sparse inter-cluster bridges with learned transfer weights T_kℓ = softmax(v_ℓ(k)/κ)) Pareto-dominates flat gossip on (personalized accuracy, bytes). The **oracle-cluster arm** (ground-truth domains) decouples "does hierarchy help?" from "can we discover clusters?" — if oracle-hierarchy fails to beat flat gossip at matched bytes, the premise itself is dead on this testbed regardless of discovery quality (kill criterion).

**H4 — Soft beats hard under ambiguity.** Affinity-weighted mixing (per-neighbor softmax weights with a self-weight floor, Sinkhorn-projected toward doubly-stochastic; no discrete clusters anywhere) is more robust than hard clustering when domains overlap or discovery is noisy; hard two-tier wins on message count when domains are crisp. This is the hedge that makes the project un-blockable by G1: DAC (2022) already argues hard assignment is brittle — our contribution is doing this for LoRA states with rank-aware merging, not the soft-clustering idea itself.

**H5 — Heterogeneous ranks as first-class citizens.** With H1's merge, gossip tolerates any budget-matched rank spread from P1/P2's allocators with minor loss vs homogeneous-at-same-budget, and a mid-training rank change is a purely local O(1) operation (refactorize to the new rank after the next merge) — the property that makes P1's dynamic rank adjustment composable with decentralization, which no surveyed system offers.

**H6 — Theory.** Per-round truncated-SVD refactorization is a biased/contractive compression of the local state with quality ω tied to the discarded tail mass. Composing CHOCO-SGD's gossip-with-compression analysis (biased compressors, ω ≤ 1) with the Beznosikov et al. biased-compressor taxonomy + error feedback yields an O(1/√T)-type rate with additive terms in (spectral gap of the two-tier mixing matrix, ζ_out·β for inter-cluster leakage, ε_r tail mass). Both theory anchors verified in the survey. Corollary worth testing empirically: **error feedback for truncation** (accumulate the refactorization residual locally, re-inject next round) — absent from every surveyed LoRA-gossip system; cheap to implement; potentially a standalone contribution.

## 5. Experimental design

Testbed (continuity with the existing code): CIFAR-100 → 5 domains (superclass groups) × 3 clients, Dirichlet(0.5) within domain, frozen ImageNet ResNet-18, LoRA(r=16, α=32) on fc. Feature-cache protocol (backbone frozen ⇒ cache 512-d features once; drops augmentation, uniformly for all arms) makes 50-round × 3-seed batteries run in minutes on MPS.

| Experiment | Question | Arms | Gate |
|---|---|---|---|
| 04 baseline battery | protocol-correct floors/ceilings | local-only, FedAvg(ΔW), gossip-factor, gossip-ΔW (all MH-mixed, 50r × 3 seeds) | G2: no inversions; quantify FedAvg-vs-gossip consensus gap |
| 05 signature validation | H2 (RQ1 rescue) | {spectral, S1, S2, S3, S4, S5} × {local-only, mid-gossip} × stages {2,5,10,20} | G1: ARI ≥ 0.7 → hard viable; 0.4–0.7 → soft only; < 0.4 → pivot to layer4+fc adapters |
| 06 affinity/hierarchical gossip | H3, H4 (the contribution) | flat-ΔW, soft AWG, hard two-tier, oracle-cluster two-tier, IFCA-style server skyline | oracle arm = kill criterion for hierarchy premise |
| 07 heterogeneous ranks | H1 (full), H5 | budget-240 spreads: uniform 16 vs {4..64} mixes; ΔW vs zero-pad; mid-run re-rank at round 25 | effect-size vs rank spread |
| diagnostics | H6 | measured spectral gaps, tail-mass ε_r, consensus distance, 1/√T fit; optional error-feedback arm | assumption-validation subsection |

Metrics everywhere: personalized acc, consensus acc, fairness gap Γ (max−min per-domain), participation (fraction of clients beating local-only), ARI/NMI trajectories, cumulative bytes (factors only — never ship dense ΔW: r(d_in+d_out) vs d_in·d_out), rounds-to-target, refactorization tail mass and wall-clock.

Multi-seed discipline: {42, 43, 44}; mean ± std; single-seed numbers are smoke tests by definition.

## 6. Novelty map (survey-verified, 2026-07-20)

**The intersection we claim — serverless + heterogeneous ranks + undeclared domains + affinity/hierarchical mixing + dynamic rank — is unoccupied**, but each pairwise slice is taken; the paper must cite and differentiate:

- **DeCAF** (Zhang et al., arXiv 2505.21382; Neural Networks) — *the closest work*: decentralized gossip LoRA with truncated-SVD refactorization per consensus round. Flat topology, homogeneous ranks, no domain discovery. Our H1 mechanism exists here in embryo — the contribution list must lead with heterogeneity + discovery + topology, not ΔW-merging alone. **Read in full before writing the paper's contribution section.**
- **Dec-LoRA** (Ghiasvand et al., arXiv 2501.15361) — serverless factor-space LoRA gossip, O(1/√T), homogeneous rank, ring/ER topologies, no clustering. Natural flat baseline; also evidence factor-gossip "works" at homogeneous rank (consistent with H1's shared-init caveat).
- **ADF-LoRA** (arXiv 2511.18291) — alternating-factor gossip with DS mixing; documents block-divergence of naive factor alternation under gossip (ammunition for the merge-operator story).
- **HetLoRA** (Cho et al., **EMNLP 2024 main, pp. 12903–12913 — confirmed**, fix references.bib), **FlexLoRA** (arXiv 2402.11505), **FLoRA** (arXiv 2409.05976): the server-side merge-operator lineage (zero-pad / ΔW-SVD / stacking). Port zero-pad and ΔW variants into gossip as exp07 comparators. NOTE: the "NeurIPS 2024" venue attributions for FlexLoRA and FLoRA in `references.bib` are **unconfirmed** — cite arXiv unless proceedings confirm.
- **DFCA** (arXiv 2510.15300) — decentralized IFCA: functional self-assignment to k cluster models, k known a priori, full models, O(1/√T). Differentiate: we discover K, use adapters not full models, and weight rather than assign.
- **DAC** (Listo Zec et al. 2022) + its 2024 similarity-metric follow-up — soft decentralized clustering; the follow-up contains the published negative result mirroring our exp03 (pretrained backbones flatten most similarity signals; inverse-L2 survives). Both baseline and framing citation for H2/H4.
- **PFedDST** (arXiv 2502.07750), **L2C** (Li et al., CVPR 2022) — learned peer-selection/mixing weights over full models. L2C is the main threat to "learned transfer weights"; differentiate via LoRA-state affinities, rank-awareness, two-tier communication budget, and heterogeneous ranks.
- **cFedLoRA** (ADMA 2025, LNAI 16197, pp. 191–205) — server-side clustering of LoRA updates (parameter-similarity works there because it clusters *updates*, consistent with our ΔW-only signature argument).
- **Novelty support**: the IJCAI-25 FedLoRA survey (Yang et al., Survey Track, pp. 10779–10787) covers **zero** gossip/peer-to-peer methods — "decentralized" appears once, in the FL definition. The serverless column of `main.tex` Table 1 stands, provided DeCAF is added and differentiated.
- **Theory anchors verified**: CHOCO-SGD (Koloskova et al. 2019; rate O(1/(nT) + 1/(Tδ²ω)²), biased compressors ω ≤ 1 in gossip) and Beznosikov–Horváth–Richtárik–Safaryan (JMLR 24(276), 2023; biased-compressor classes, error-feedback rate template). Neither covers gossip + per-round SVD truncation of *states* — that composition is our H6 gap to fill.
- Cross-check pending from the still-running synthesis pass: Werner et al. (TMLR 2023) on early-training clustering error (supports EMA-smoothed affinities), FL-TAC/FedLFC/FedHLT (centralized cluster-aware LoRA).

## 7. Risks & kill criteria

1. **No signature clears ARI 0.4 (G1 fail)** → domain signal is genuinely absent from fc-only adapters → pivot: extend LoRA to layer4 + fc (richer ΔW), or accept discovery-free soft mixing as the headline (H4 path). Either way exp05's grid is the paper's RQ1 figure — the negative result is publishable *because* the signature ladder localizes the cause.
2. **Oracle-cluster hierarchy ≤ flat gossip at matched bytes (exp06)** → hierarchy premise dead on this testbed → headline becomes affinity-weighted decentralized LoRA (AWG); the two-tier machinery remains as a communication-efficiency section at scale.
3. **ΔW-merge ≈ factor-merge even at high rank spread (exp07)** → measure gauge drift directly (principal angles between clients' A-row-spaces over rounds); if drift is genuinely small under shared init, report honestly and lean on zero-pad comparison + FLoRA's server-side effect size; consider longer local epochs (more drift) as a stress arm.
4. **Feature-cache protocol challenge** (no augmentation) → all arms share it; if a reviewer-facing number needs the full pipeline, rerun the headline table with `--no-cache` once at the end.
5. **15 clients is small for clustering claims** → frame exp05/06 as controlled testbed; scale-out (50–100 clients, feature-cached, still cheap) is a Phase-3+ robustness section; every-node-holds-all-signatures is honest at N=15 and replaced by push-sum at scale (stated, not implemented).

## 8. Immediate next steps

Execute the implementation plan (`docs/superpowers/plans/2026-07-20-project3-affinity-hierarchical-gossip.md`): Tasks 0–5 are pure infrastructure (merge kernels, MH mixing, runner, feature cache — all unit-tested), Task 6 = exp04 baseline battery (fills the paper's first table cells), Tasks 7–8 = exp05 signature validation (decides G1 within days, since the cache makes the grid cheap). Then plan Phase 3 with the G1/G2 evidence in hand.
