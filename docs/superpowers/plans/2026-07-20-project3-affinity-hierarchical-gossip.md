# Project 3: Affinity-Weighted & Hierarchical Gossip — Implementation Plan (Phases 0–2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn Project 3 from a flat, factor-averaging gossip smoke test into a protocol-correct decentralized LoRA testbed with (a) gauge-invariant ΔW-space merging, (b) doubly-stochastic mixing, (c) trustworthy 50-round × 3-seed baselines, and (d) a rescued domain-discovery experiment using direction-aware signatures — reaching the two decision gates (G1: is domain structure discoverable? G2: do the baselines behave sanely?) that determine the design of the hierarchical/affinity gossip contribution.

**Architecture:** All new code lives inside `project-3-hierarchical-gossip/` (repo convention: the three projects are self-contained; P2/P3 already duplicate `lora_resnet.py`/`cifar100_domains.py` rather than cross-import). New modules: `src/federated/merge.py` (ΔW kernels, ported from the P1/P2 centralized prototypes), `src/federated/mixing.py` (Metropolis–Hastings doubly-stochastic mixing), `src/federated/runner.py` (pluggable-merge decentralized runner that generalizes `GossipProtocol`), `src/data/feature_cache.py` (frozen-backbone feature cache — the key speed unlock), `src/clustering/signatures.py` (direction-aware domain signatures). Existing `GossipProtocol`, experiments 01–03, and their results stay untouched for continuity.

**Tech Stack:** Python 3.10+, torch ≥ 2.0, torchvision, numpy, scikit-learn, pyyaml, matplotlib (all already in `requirements.txt`), plus `pytest` (added by Task 1). No other new dependencies.

## Research context (why these tasks)

Full hypothesis write-up is in the accompanying research report; the short version driving this plan:

- **H1 (merge operator):** `gossip.py:88-96` averages A and B factors separately — exactly the operation `paper/main.tex` Proposition 1 declares ill-posed (gauge ambiguity, spurious cross terms), and it cannot handle heterogeneous ranks at all. Merging the *scaled effective update* ΔW = (α/r)·B@A and refactorizing per-client via truncated SVD is gauge-invariant, rank-agnostic, and already prototyped centrally in P1 (`fedavg_aggregation.py`) and P2 (`hetero_fedavg.py`). Phase 0 ports it to gossip.
- **H2 (domain discovery):** Experiment 03's clustering failure (ARI 0.03 → −0.14 *under local-only training*, the most favorable condition) is explained by the signature, not the setting: `domain_clustering.py:44-60` uses singular-value *magnitudes* of A, B, BA — direction-blind features. On an fc-only LoRA over ResNet-18, domain identity (which 20 of 100 classes a client holds) lives in *which rows of ΔW have mass* and *which subspaces changed*, not in the spectrum's shape. Phase 2 tests a ladder of direction/function-aware signatures (row-norm profiles → ΔW subspaces → probe logits → peer cross-evaluation → inverse-L2 distance on ΔW). Literature echo (deep-research, 2026-07-20): Listo Zec et al.'s decentralized-similarity study (arXiv 2409.16066-line follow-up to DAC) reports that with a **pre-trained ResNet-18 fine-tuned on CIFAR-100 superclass clusters — our exact setting —** inverse-loss, cosine-on-weights, and cosine-on-gradients all failed to beat random peer selection (pretraining makes scores too uniform) while **inverse L2 weight distance still recovered cluster structure**. Our signatures operate on ΔW only (the pretrained mass is excluded by construction), which is precisely the property their failing full-weight cosines lacked — but their result mandates including the inverse-L2 signature and tempers confidence in cross-evaluation (they found inverse-loss noisy → wrong peers → catastrophic forgetting; hence the EMA in Phase 3).
- **Protocol integrity:** every current result is a 2-round single-seed smoke test; gossip mixes via one random neighbor with 0.5/0.5 (row-stochastic only, not mass-preserving — misaligned with the paper's doubly-stochastic assumption); personalized-vs-consensus evaluation is conflated (main.tex §"protocol correction" demands both be reported for every method). Phase 0–1 fix all three before any claim is made.
- **Speed unlock:** the backbone is frozen and shared, so 512-d penultimate features never change. Caching them once turns every local epoch into a linear-head pass — the 50-round × 3-seed battery becomes minutes instead of days, on CPU/MPS.

**Decision gates after Phase 2:**
- **G1 (from Experiment 05):** best signature ARI ≥ 0.7 by stage 10 → hard clustering viable, build two-tier hierarchical gossip. ARI 0.4–0.7 → soft affinity-weighted mixing only (no discrete clusters). ARI < 0.4 → domain structure is not in fc-adapters; pivot to richer adapters (LoRA on layer4 + fc) before continuing.
- **G2 (from Experiment 04):** ΔW-merge ≥ factor-merge at homogeneous rank (expected ≈ equal from shared init; the H1 gap emerges under rank heterogeneity in Phase 4), FedAvg consensus ≥ gossip consensus (spectral-gap expectation), local-only quantifies the participation floor. Any inversion → debug before Phase 3.

## Global Constraints

- All new code and experiments live under `project-3-hierarchical-gossip/`; no imports from `project-1-*` or `project-2-*` (repo convention; port with a provenance comment instead).
- Do not modify `GossipProtocol.gossip_round`/`_pairwise_average` semantics, experiments 01–03, or anything under `results/experiment_01..03/` — they are the recorded baseline lineage. New behavior goes in new modules/experiments.
- Seeds for any reported number: `[42, 43, 44]`; single-seed numbers are labeled smoke tests.
- Every experiment README must report BOTH evaluation protocols (personalized: each client's model on its own test shard; consensus: one merged adapter at reference rank 16 on the full CIFAR-100 test set) and must state the feature-cache protocol note.
- No test-set information may influence training or affinity decisions: cross-evaluation utilities use a held-out slice of each client's TRAIN shard; the S3 probe set uses test images as unlabeled inputs only.
- LoRA config stays `rank=16, alpha=32` for Phases 0–2 (continuity with experiments 01–03); heterogeneous ranks enter in Phase 4.
- Work on branch `project3-gossip-phase0`; one commit per task minimum, messages in the repo's imperative style ("Add ...", "Fix ...").
- Run all commands from `project-3-hierarchical-gossip/` unless stated otherwise. Tests: `python -m pytest tests/ -q`.

---

### Task 0: Branch + test scaffolding

**Files:**
- Create: `project-3-hierarchical-gossip/tests/__init__.py` (empty)
- Create: `project-3-hierarchical-gossip/tests/conftest.py`
- Modify: `project-3-hierarchical-gossip/requirements.txt` (append `pytest`)

**Interfaces:**
- Produces: pytest discovers `tests/` with project root on `sys.path`, so tests can `from src.federated.merge import ...` exactly like the experiment scripts do.

- [ ] **Step 1: Create branch**

```bash
cd /Users/ghanibhutigogoi/Documents/Decentralized-MultiDomain-Federated-Lora
git checkout -b project3-gossip-phase0
```

- [ ] **Step 2: Write conftest.py**

```python
"""Make `src.*` importable in tests, mirroring the experiments' sys.path pattern."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

- [ ] **Step 3: Append pytest to requirements.txt** (one new line: `pytest`)

- [ ] **Step 4: Verify pytest runs (0 tests, exit code 0)**

Run: `cd project-3-hierarchical-gossip && python -m pytest tests/ -q`
Expected: `no tests ran`

- [ ] **Step 5: Commit**

```bash
git add project-3-hierarchical-gossip/tests project-3-hierarchical-gossip/requirements.txt
git commit -m "Add pytest scaffolding for project 3"
```

---

### Task 1: ΔW-space merge kernels (`merge.py`)

**Files:**
- Create: `project-3-hierarchical-gossip/src/federated/merge.py`
- Test: `project-3-hierarchical-gossip/tests/test_merge.py`

**Interfaces:**
- Consumes: the `lora_state` format produced by `get_lora_state` (`src/models/lora_resnet.py:105-116`): `{layer_name: {'A': tensor[r, in], 'B': tensor[out, r]}}`, CPU tensors.
- Produces (used by Tasks 3, 5, 6, 7, 8):
  - `scaled_delta_w(lora_state, alpha) -> {layer: tensor[out, in]}`
  - `refactorize(delta_w, rank, alpha) -> ({'A': tensor[rank, in], 'B': tensor[out, rank]}, tail_mass: float)`
  - `merge_states(states: list[lora_state], weights: list[float], target_rank: int, alpha: float) -> (lora_state, {layer: tail_mass})`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_merge.py
import torch

from src.federated.merge import scaled_delta_w, refactorize, merge_states


def make_state(out_f=10, in_f=20, rank=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {'fc': {'A': torch.randn(rank, in_f, generator=g),
                   'B': torch.randn(out_f, rank, generator=g)}}


def test_scaled_delta_w_shape_and_value():
    state = make_state(rank=4)
    d = scaled_delta_w(state, alpha=32)['fc']
    expected = (32 / 4) * state['fc']['B'] @ state['fc']['A']
    assert d.shape == (10, 20)
    assert torch.allclose(d, expected)


def test_refactorize_roundtrip_exact_when_rank_sufficient():
    state = make_state(rank=4)
    d = scaled_delta_w(state, alpha=32)['fc']
    factors, tail = refactorize(d, rank=4, alpha=32)
    recon = (32 / 4) * factors['B'] @ factors['A']
    assert factors['A'].shape == (4, 20) and factors['B'].shape == (10, 4)
    assert torch.allclose(recon, d, atol=1e-4)
    assert tail < 1e-6


def test_gauge_invariance():
    """(B, A) and (B Qᵀ, Q A) encode the same ΔW for orthogonal Q — paper Prop. 1."""
    state = make_state(rank=4)
    q, _ = torch.linalg.qr(torch.randn(4, 4))
    rotated = {'fc': {'A': q @ state['fc']['A'], 'B': state['fc']['B'] @ q.T}}
    d1 = scaled_delta_w(state, 32)['fc']
    d2 = scaled_delta_w(rotated, 32)['fc']
    assert torch.allclose(d1, d2, atol=1e-4)


def test_hetero_rank_merge_equals_dense_average():
    s1, s2 = make_state(rank=2, seed=1), make_state(rank=6, seed=2)
    merged, tails = merge_states([s1, s2], [0.5, 0.5], target_rank=8, alpha=32)
    dense = 0.5 * scaled_delta_w(s1, 32)['fc'] + 0.5 * scaled_delta_w(s2, 32)['fc']
    recon = (32 / 8) * merged['fc']['B'] @ merged['fc']['A']
    assert torch.allclose(recon, dense, atol=1e-4)  # rank(dense) <= 8: no truncation
    assert tails['fc'] < 1e-6


def test_truncation_reports_tail_mass():
    s1, s2 = make_state(rank=6, seed=1), make_state(rank=6, seed=2)
    merged, tails = merge_states([s1, s2], [0.5, 0.5], target_rank=2, alpha=32)
    assert merged['fc']['A'].shape == (2, 20)
    assert 0.0 < tails['fc'] < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_merge.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.federated.merge'`

- [ ] **Step 3: Write the implementation**

```python
# src/federated/merge.py
"""ΔW-space merge kernels for (heterogeneous-rank) LoRA aggregation.

Ported from the AH-LoRA centralized prototypes (project-1
Federated/fedavg_aggregation.py `_factorize_delta`, project-2
src/federated/hetero_fedavg.py `aggregate_delta_w`), adapted to project-3's
lora_state format {layer: {'A': [r, in], 'B': [out, r]}}.

All merging happens on the *scaled effective update*

    ΔW = (alpha / r) * B @ A

which is gauge-invariant and has shape (out, in) regardless of rank
(paper/main.tex §4.2, Proposition 1). Refactorization back to a client's own
rank is the Eckart–Young-optimal truncated SVD; `tail_mass` reports the
squared spectral mass discarded (the ε_r of the convergence analysis).
"""

import torch


def scaled_delta_w(lora_state, alpha):
    """Per-layer effective update ΔW = (alpha / r) * B @ A."""
    deltas = {}
    for layer, p in lora_state.items():
        r = p['A'].shape[0]
        deltas[layer] = (alpha / r) * (p['B'] @ p['A'])
    return deltas


def refactorize(delta_w, rank, alpha):
    """Factor a dense ΔW back to LoRA (A, B) at `rank`.

    Satisfies (alpha / rank) * B @ A == ΔW exactly when rank >= rank(ΔW),
    else the best rank-`rank` approximation. Returns (factors, tail_mass).
    """
    M = delta_w * (rank / alpha)  # undo the scaling the forward pass applies
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    k = min(rank, S.shape[0])
    total = float((S ** 2).sum())
    kept = float((S[:k] ** 2).sum())
    tail_mass = (1.0 - kept / total) if total > 0 else 0.0
    sqrt_s = torch.sqrt(S[:k])
    B = U[:, :k] * sqrt_s.unsqueeze(0)   # [out, k]
    A = sqrt_s.unsqueeze(1) * Vh[:k, :]  # [k, in]
    if k < rank:  # pad up to the requested rank (degenerate matrices)
        B = torch.cat([B, torch.zeros(B.shape[0], rank - k)], dim=1)
        A = torch.cat([A, torch.zeros(rank - k, A.shape[1])], dim=0)
    return {'A': A, 'B': B}, tail_mass


def merge_states(states, weights, target_rank, alpha):
    """Weighted ΔW-space merge of LoRA states, refactorized to `target_rank`.

    states may have different ranks; weights must sum to 1.
    Returns (merged_lora_state, {layer: tail_mass}).
    """
    merged, tails = {}, {}
    for layer in states[0]:
        avg = None
        for state, w in zip(states, weights):
            p = state[layer]
            r = p['A'].shape[0]
            d = (alpha / r) * (p['B'] @ p['A'])
            avg = w * d if avg is None else avg + w * d
        merged[layer], tails[layer] = refactorize(avg, target_rank, alpha)
    return merged, tails
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_merge.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add project-3-hierarchical-gossip/src/federated/merge.py project-3-hierarchical-gossip/tests/test_merge.py
git commit -m "Add gauge-invariant delta-W merge kernels with SVD refactorization"
```

---

### Task 2: Doubly-stochastic mixing (`mixing.py`)

**Files:**
- Create: `project-3-hierarchical-gossip/src/federated/mixing.py`
- Modify: `project-3-hierarchical-gossip/src/federated/gossip.py:47-86` (extract `build_topology` as a module-level function; keep the method delegating to it)
- Test: `project-3-hierarchical-gossip/tests/test_mixing.py`

**Interfaces:**
- Consumes: adjacency dict `{client_id: [neighbor_ids]}` as built by `GossipProtocol._build_topology`.
- Produces (used by Tasks 3, 8, 9):
  - `build_topology(client_ids: list, topology: str, seed: int = 42) -> dict` (moved from gossip.py, behavior-identical)
  - `metropolis_hastings_weights(neighbors: dict, client_ids: list | None = None) -> (np.ndarray, list)` — symmetric doubly-stochastic W plus the id ordering
  - `spectral_gap(W: np.ndarray) -> float`
  - `sinkhorn_project(M: np.ndarray, n_iters: int = 200) -> np.ndarray` (used by the Phase-3 soft variant; built and tested now because it is 10 lines and pins the DS contract)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mixing.py
import numpy as np

from src.federated.gossip import build_topology
from src.federated.mixing import (
    metropolis_hastings_weights,
    spectral_gap,
    sinkhorn_project,
)


def test_build_topology_ring_matches_legacy():
    ids = list(range(5))
    nbrs = build_topology(ids, 'ring')
    assert nbrs[0] == [4, 1]
    assert nbrs[3] == [2, 4]


def test_mh_weights_doubly_stochastic_and_symmetric():
    ids = list(range(15))
    nbrs = build_topology(ids, 'ring')
    W, order = metropolis_hastings_weights(nbrs, ids)
    assert order == ids
    assert np.allclose(W, W.T)
    assert np.allclose(W.sum(axis=1), 1.0, atol=1e-9)
    assert np.allclose(W.sum(axis=0), 1.0, atol=1e-9)
    assert (W >= -1e-12).all()


def test_mh_ring_has_positive_spectral_gap():
    ids = list(range(15))
    W, _ = metropolis_hastings_weights(build_topology(ids, 'ring'), ids)
    gap = spectral_gap(W)
    assert 0.0 < gap < 1.0


def test_sinkhorn_projects_to_doubly_stochastic():
    rng = np.random.default_rng(0)
    M = rng.uniform(0.1, 1.0, size=(6, 6))
    P = sinkhorn_project(M)
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(P.sum(axis=0), 1.0, atol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_mixing.py -q`
Expected: FAIL with `ImportError: cannot import name 'build_topology' from 'src.federated.gossip'`

- [ ] **Step 3: Extract `build_topology` in gossip.py**

In `src/federated/gossip.py`, add a module-level function containing the exact body of `_build_topology` (parameterized by `client_ids`, `topology`, `seed`), and change the method to delegate. The method's behavior (including the `random_regular` RNG usage) must not change:

```python
def build_topology(client_ids, topology, seed=42):
    """Build a communication graph. Moved out of GossipProtocol so other
    runners can reuse it; behavior identical to the original method."""
    n = len(client_ids)
    ids = client_ids
    neighbors = {cid: [] for cid in ids}

    if topology == 'ring':
        for i, cid in enumerate(ids):
            prev_id = ids[(i - 1) % n]
            next_id = ids[(i + 1) % n]
            neighbors[cid] = [prev_id, next_id]

    elif topology == 'fully_connected':
        for cid in ids:
            neighbors[cid] = [other for other in ids if other != cid]

    elif topology == 'random_regular':
        k = min(4, n - 1)
        rng = random.Random(seed)
        for cid in ids:
            possible = [other for other in ids if other != cid]
            selected = rng.sample(possible, k)
            neighbors[cid] = selected
        for cid in ids:
            for neighbor in neighbors[cid]:
                if cid not in neighbors[neighbor]:
                    neighbors[neighbor].append(cid)

    return neighbors
```

and inside the class:

```python
    def _build_topology(self, topology):
        return build_topology(self.client_ids, topology, self.seed)
```

(Delete the old method body; keep the docstring on the module function.)

- [ ] **Step 4: Write mixing.py**

```python
# src/federated/mixing.py
"""Doubly-stochastic mixing matrices for gossip rounds.

The legacy GossipProtocol averages toward one random neighbor per round,
which is row-stochastic but not mass-preserving. The paper's convergence
assumptions (main.tex, Assumption 1) require a symmetric doubly-stochastic
mixing matrix with positive spectral gap; Metropolis–Hastings weights give
exactly that for any undirected graph.
"""

import numpy as np


def metropolis_hastings_weights(neighbors, client_ids=None):
    """Symmetric doubly-stochastic W from an adjacency dict.

    W[i, j] = 1 / (1 + max(deg_i, deg_j)) on edges; W[i, i] = 1 - row sum.
    Returns (W, ids) where ids fixes the row/column ordering.
    """
    ids = list(client_ids) if client_ids is not None else sorted(neighbors)
    index = {cid: k for k, cid in enumerate(ids)}
    deg = {cid: len(neighbors[cid]) for cid in ids}
    n = len(ids)
    W = np.zeros((n, n))
    for cid in ids:
        for nid in neighbors[cid]:
            W[index[cid], index[nid]] = 1.0 / (1.0 + max(deg[cid], deg[nid]))
    for i in range(n):
        W[i, i] = 1.0 - W[i].sum()
    return W, ids


def spectral_gap(W):
    """1 - second-largest |eigenvalue|; > 0 iff the gossip mixes."""
    eigs = np.sort(np.abs(np.linalg.eigvals(W)))[::-1]
    return float(1.0 - eigs[1])


def sinkhorn_project(M, n_iters=200, eps=1e-9):
    """Approximately project a nonnegative matrix to doubly stochastic
    by alternating row/column normalization (used by soft affinity mixing)."""
    P = np.asarray(M, dtype=float) + eps
    for _ in range(n_iters):
        P = P / P.sum(axis=1, keepdims=True)
        P = P / P.sum(axis=0, keepdims=True)
    return P / P.sum(axis=1, keepdims=True)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_mixing.py tests/test_merge.py -q`
Expected: `9 passed`

- [ ] **Step 6: Verify legacy experiment still imports** (behavior guard for the gossip.py edit)

Run: `python -c "import sys; sys.path.insert(0, '.'); from src.federated.gossip import GossipProtocol, build_topology; print('ok')"`
Expected: `ok`

- [ ] **Step 7: Commit**

```bash
git add project-3-hierarchical-gossip/src/federated/mixing.py project-3-hierarchical-gossip/src/federated/gossip.py project-3-hierarchical-gossip/tests/test_mixing.py
git commit -m "Add Metropolis-Hastings doubly-stochastic mixing and extract build_topology"
```

---

### Task 3: Decentralized runner with pluggable merge (`runner.py`)

**Files:**
- Create: `project-3-hierarchical-gossip/src/federated/runner.py`
- Test: `project-3-hierarchical-gossip/tests/test_runner.py`

**Interfaces:**
- Consumes: `FederatedClient`-shaped objects (`client_id`, `domain_id`, `train()`, `evaluate()`, `get_lora_state()`, `set_lora_state(state)`); `build_topology`, `metropolis_hastings_weights`, `merge_states`.
- Produces (used by Tasks 6, 7 and Phase 3):
  - `class DecentralizedRunner(clients, topology='ring', merge='delta_w', alpha=32, seed=42, eval_every=1, consensus_fn=None, mixing_matrix=None)`
  - `.run(n_rounds, verbose=True) -> history` with keys `rounds, avg_accuracy, per_domain_accuracy, per_client_accuracy, consensus_accuracy, bytes_cumulative, mean_tail_mass`
  - `.consensus_state(target_rank) -> lora_state` (uniform ΔW average over all clients)
  - `merge` modes: `'delta_w'` (merge_states at each client's own rank) and `'factor'` (weighted average of A and B directly — the legacy semantics generalized to W rows; homogeneous ranks only)
  - `consensus_fn(lora_state) -> float` is supplied by the experiment driver (it owns a template model + full-test loader); when None, consensus_accuracy stays empty.
  - `mixing_matrix`: optional `(W, ids)` override; default = MH weights on the chosen topology. Phase 3 passes affinity-derived matrices here — this is the extension seam.

- [ ] **Step 1: Write the failing tests** (FakeClient keeps this test torch-only, no data/models)

```python
# tests/test_runner.py
import numpy as np
import torch

from src.federated.merge import scaled_delta_w
from src.federated.runner import DecentralizedRunner


class FakeClient:
    """Minimal client: rank-6 state whose true matrix rank is 1, so
    neighborhood averages (rank <= 3) never get truncated at target rank 6
    and the doubly-stochastic mean-preservation invariant holds exactly."""

    def __init__(self, client_id, domain_id, seed):
        g = torch.Generator().manual_seed(seed)
        A = torch.zeros(6, 20)
        B = torch.zeros(10, 6)
        A[0] = torch.randn(20, generator=g)
        B[:, 0] = torch.randn(10, generator=g)
        self.state = {'fc': {'A': A, 'B': B}}
        self.client_id = client_id
        self.domain_id = domain_id

    def train(self):
        return {'loss': 0.0, 'accuracy': 0.0, 'n_samples': 1}

    def evaluate(self):
        return {'loss': 0.0, 'accuracy': 0.5, 'n_samples': 1}

    def get_lora_state(self):
        return {'fc': {'A': self.state['fc']['A'].clone(),
                       'B': self.state['fc']['B'].clone()}}

    def set_lora_state(self, state):
        self.state = state


def mean_delta(clients, alpha=32):
    ds = [scaled_delta_w(c.get_lora_state(), alpha)['fc'] for c in clients]
    return torch.stack(ds).mean(dim=0)


def test_delta_w_round_preserves_global_mean():
    clients = [FakeClient(i, i % 2, seed=i) for i in range(5)]
    before = mean_delta(clients)
    runner = DecentralizedRunner(clients, topology='ring', merge='delta_w')
    runner.mix_round()
    after = mean_delta(clients)
    assert torch.allclose(before, after, atol=1e-4)


def test_mix_round_counts_bytes():
    clients = [FakeClient(i, 0, seed=i) for i in range(4)]
    runner = DecentralizedRunner(clients, topology='ring', merge='delta_w')
    n_bytes, _ = runner.mix_round()
    # ring of 4: each client receives 2 neighbor states of (6*20 + 10*6) floats
    assert n_bytes == 4 * 2 * 4 * (6 * 20 + 10 * 6)


def test_factor_mode_matches_manual_average():
    clients = [FakeClient(i, 0, seed=i) for i in range(3)]
    states = {c.client_id: c.get_lora_state() for c in clients}
    runner = DecentralizedRunner(clients, topology='fully_connected', merge='factor')
    W, ids = runner.W, runner.ids
    runner.mix_round()
    k = ids.index(0)
    expected_A = sum(W[k, j] * states[ids[j]]['fc']['A'] for j in range(len(ids)))
    assert torch.allclose(clients[0].get_lora_state()['fc']['A'], expected_A, atol=1e-5)


def test_run_returns_history_with_consensus():
    clients = [FakeClient(i, i % 2, seed=i) for i in range(4)]
    runner = DecentralizedRunner(
        clients, topology='ring', merge='delta_w',
        consensus_fn=lambda state: 0.25,
    )
    history = runner.run(n_rounds=2, verbose=False)
    assert history['rounds'] == [0, 1]
    assert len(history['avg_accuracy']) == 2
    assert history['consensus_accuracy'] == [0.25, 0.25]
    assert len(history['bytes_cumulative']) == 2
    assert history['bytes_cumulative'][1] == 2 * history['bytes_cumulative'][0] // 1 * 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_runner.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.federated.runner'`

- [ ] **Step 3: Write runner.py**

```python
# src/federated/runner.py
"""Decentralized training runner with pluggable merge and mixing.

Generalizes GossipProtocol: instead of one random neighbor with a 0.5/0.5
factor average, every round each client merges ALL graph neighbors' states
using a doubly-stochastic mixing matrix, with a choice of merge operator:

    'delta_w' : gauge-invariant scaled-ΔW average + per-client SVD
                refactorization to the client's own rank (paper §4.2)
    'factor'  : direct weighted average of A and B factors (legacy
                semantics generalized to W rows; homogeneous ranks only)

The legacy GossipProtocol and experiments 01-03 are left untouched.
"""

import numpy as np
import torch
from tqdm import tqdm

from src.federated.gossip import build_topology
from src.federated.merge import merge_states
from src.federated.mixing import metropolis_hastings_weights


BYTES_PER_FLOAT = 4


class DecentralizedRunner:

    def __init__(self, clients, topology='ring', merge='delta_w', alpha=32,
                 seed=42, eval_every=1, consensus_fn=None, mixing_matrix=None):
        if merge not in ('delta_w', 'factor'):
            raise ValueError(f"Unknown merge mode: {merge}")
        self.clients = clients
        self.client_ids = [c.client_id for c in clients]
        self.merge = merge
        self.alpha = alpha
        self.eval_every = eval_every
        self.consensus_fn = consensus_fn
        self.neighbors = build_topology(self.client_ids, topology, seed)
        if mixing_matrix is not None:
            self.W, self.ids = mixing_matrix
        else:
            self.W, self.ids = metropolis_hastings_weights(
                self.neighbors, self.client_ids)
        self.history = {
            'rounds': [], 'avg_accuracy': [], 'per_domain_accuracy': [],
            'per_client_accuracy': [], 'consensus_accuracy': [],
            'bytes_cumulative': [], 'mean_tail_mass': [],
        }
        self._total_bytes = 0

    # -- core ------------------------------------------------------------

    def mix_round(self):
        """One synchronous mixing round. Returns (bytes_this_round, mean_tail)."""
        states = {c.client_id: c.get_lora_state() for c in self.clients}
        new_states, n_bytes, tail_sum, tail_n = {}, 0, 0.0, 0

        for k, cid in enumerate(self.ids):
            row = self.W[k]
            involved = [j for j in range(len(self.ids)) if row[j] > 1e-12]
            merge_input = [states[self.ids[j]] for j in involved]
            weights = [float(row[j]) for j in involved]
            for j in involved:
                nid = self.ids[j]
                if nid != cid:
                    for p in states[nid].values():
                        n_bytes += BYTES_PER_FLOAT * (p['A'].numel() + p['B'].numel())

            if self.merge == 'delta_w':
                own_rank = next(iter(states[cid].values()))['A'].shape[0]
                merged, tails = merge_states(
                    merge_input, weights, own_rank, self.alpha)
                tail_sum += sum(tails.values())
                tail_n += len(tails)
            else:  # 'factor'
                merged = {
                    layer: {
                        'A': sum(w * s[layer]['A'] for s, w in zip(merge_input, weights)),
                        'B': sum(w * s[layer]['B'] for s, w in zip(merge_input, weights)),
                    }
                    for layer in merge_input[0]
                }
            new_states[cid] = merged

        for c in self.clients:
            c.set_lora_state(new_states[c.client_id])
        self._total_bytes += n_bytes
        mean_tail = (tail_sum / tail_n) if tail_n else 0.0
        return n_bytes, mean_tail

    def consensus_state(self, target_rank):
        """Uniform ΔW average over ALL clients, refactorized to target_rank.
        Offline diagnostic for the consensus evaluation protocol."""
        states = [c.get_lora_state() for c in self.clients]
        w = [1.0 / len(states)] * len(states)
        merged, _ = merge_states(states, w, target_rank, self.alpha)
        return merged

    # -- loop ------------------------------------------------------------

    def run(self, n_rounds=50, verbose=True):
        iterator = tqdm(range(n_rounds), desc=f"Decentralized[{self.merge}]") \
            if verbose else range(n_rounds)
        for round_idx in iterator:
            for client in self.clients:
                client.train()
            _, mean_tail = self.mix_round()
            evals = self._evaluate_all()
            self.history['rounds'].append(round_idx)
            self.history['avg_accuracy'].append(evals['avg_accuracy'])
            self.history['per_domain_accuracy'].append(evals['per_domain'])
            self.history['per_client_accuracy'].append(evals['per_client'])
            self.history['bytes_cumulative'].append(self._total_bytes)
            self.history['mean_tail_mass'].append(mean_tail)
            if self.consensus_fn is not None:
                self.history['consensus_accuracy'].append(
                    self.consensus_fn(self.consensus_state(target_rank=16)))
            if verbose:
                iterator.set_postfix({'acc': f"{evals['avg_accuracy']:.3f}"})
        return self.history

    def _evaluate_all(self):
        per_client, per_domain, counts = {}, {}, {}
        for client in self.clients:
            acc = client.evaluate()['accuracy']
            per_client[client.client_id] = acc
            d = client.domain_id
            per_domain[d] = per_domain.get(d, 0.0) + acc
            counts[d] = counts.get(d, 0) + 1
        for d in per_domain:
            per_domain[d] /= counts[d]
        return {
            'avg_accuracy': sum(per_client.values()) / len(per_client),
            'per_domain': per_domain,
            'per_client': per_client,
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_runner.py -q`
Expected: `4 passed` (fix the final bytes assertion if the arithmetic differs — the invariant to enforce is bytes_cumulative[1] == 2 × bytes of one round)

- [ ] **Step 5: Run the whole suite**

Run: `python -m pytest tests/ -q`
Expected: `13 passed`

- [ ] **Step 6: Commit**

```bash
git add project-3-hierarchical-gossip/src/federated/runner.py project-3-hierarchical-gossip/tests/test_runner.py
git commit -m "Add DecentralizedRunner with delta-W and factor merge modes"
```

---

### Task 4: Frozen-backbone feature cache (`feature_cache.py`)

**Files:**
- Create: `project-3-hierarchical-gossip/src/data/feature_cache.py`
- Test: `project-3-hierarchical-gossip/tests/test_feature_cache.py`

**Interfaces:**
- Consumes: `partition_domain_data_dirichlet`, `get_domain_classes`, `DOMAIN_NAMES` from `src/data/cifar100_domains.py` (partitioning must be bit-identical to the image pipeline for the same seed); `LoRALinear` from `src/models/lora_resnet.py`.
- Produces (used by Tasks 6, 7 and beyond):
  - `compute_backbone_features(data_dir='./data', cache_path='./data/resnet18_features.pt', device='cpu', batch_size=256) -> cache_path` — one-off; saves `{'train_X': [50000, 512], 'train_y': [50000], 'test_X': [10000, 512], 'test_y': [10000]}` (float32/int64, eval transform, no augmentation)
  - `create_cached_federated_datasets(cache_path, n_domains=5, clients_per_domain=3, dirichlet_alpha=0.5, batch_size=64, seed=42, val_fraction=0.1) -> (clients_data, full_test_loader)` — same per-client dict contract as `create_federated_datasets` (`train_loader, test_loader, domain_id, domain_name, n_samples, classes`) **plus** `'val_loader'` (held-out `val_fraction` of the client's train shard, seeded; for cross-evaluation affinities — never test data)
  - `create_lora_head(num_classes=100, rank=16, alpha=32, in_features=512, seed=0, device='cpu') -> nn.Sequential` with the LoRALinear registered under module name `'fc'` so `get_lora_state` returns key `'fc'`, matching the full model. The frozen base Linear is seeded so every client shares the same random frozen fc, exactly like `clone_model(base_model)` does in the image pipeline.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_feature_cache.py
import torch

from src.data.feature_cache import (
    FeatureTensorDataset,
    _split_train_val,
    create_lora_head,
)
from src.models.lora_resnet import get_lora_state, set_lora_state


def test_lora_head_state_key_is_fc():
    head = create_lora_head(seed=0)
    state = get_lora_state(head)
    assert list(state.keys()) == ['fc']
    assert state['fc']['A'].shape == (16, 512)
    assert state['fc']['B'].shape == (100, 16)


def test_lora_head_base_frozen_and_shared_across_seeds():
    h1, h2 = create_lora_head(seed=0), create_lora_head(seed=0)
    assert torch.equal(h1.fc.linear.weight, h2.fc.linear.weight)
    assert not h1.fc.linear.weight.requires_grad
    assert h1.fc.lora_A.requires_grad and h1.fc.lora_B.requires_grad


def test_lora_head_forward_matches_manual_formula():
    head = create_lora_head(seed=0)
    x = torch.randn(3, 512)
    with torch.no_grad():
        head.fc.lora_B.normal_()  # nonzero so the LoRA path contributes
        expected = head.fc.linear(x) + (x @ head.fc.lora_A.T @ head.fc.lora_B.T) * (32 / 16)
        assert torch.allclose(head(x), expected, atol=1e-5)


def test_split_train_val_disjoint_and_seeded():
    idx = list(range(100))
    t1, v1 = _split_train_val(idx, val_fraction=0.1, seed=7)
    t2, v2 = _split_train_val(idx, val_fraction=0.1, seed=7)
    assert t1 == t2 and v1 == v2
    assert len(v1) == 10
    assert set(t1).isdisjoint(v1)
    assert sorted(t1 + v1) == idx


def test_feature_tensor_dataset_indexing():
    X = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    y = torch.arange(10)
    ds = FeatureTensorDataset(X, y, indices=[3, 7])
    feat, label = ds[1]
    assert torch.equal(feat, X[7]) and label.item() == 7
    assert len(ds) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_feature_cache.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.feature_cache'`

- [ ] **Step 3: Write feature_cache.py**

```python
# src/data/feature_cache.py
"""One-off frozen-backbone feature extraction for fast LoRA-head training.

The ResNet-18 backbone is frozen and shared by every client, so its 512-d
penultimate features never change during training. Caching them once turns a
local epoch into a linear-head pass, making 50-round × 3-seed batteries
tractable on CPU/MPS.

PROTOCOL NOTE (state this in every results README that uses the cache):
features use the deterministic eval transform — train-time RandomCrop /
RandomHorizontalFlip augmentation is dropped, uniformly for every arm.
Numbers are therefore not comparable to results/experiment_01/02 smoke tests.
"""

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models

from src.data.cifar100_domains import (
    DOMAIN_NAMES,
    get_domain_classes,
    get_transforms,
    partition_domain_data_dirichlet,
)
from src.models.lora_resnet import LoRALinear


def compute_backbone_features(data_dir='./data',
                              cache_path='./data/resnet18_features.pt',
                              device='cpu', batch_size=256):
    """Run the frozen ImageNet ResNet-18 backbone over CIFAR-100 once."""
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    backbone.fc = nn.Identity()
    backbone = backbone.to(device).eval()

    out = {}
    for split, train in (('train', True), ('test', False)):
        ds = datasets.CIFAR100(root=data_dir, train=train, download=True,
                               transform=get_transforms(train=False))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=2)
        feats, labels = [], []
        with torch.no_grad():
            for x, y in loader:
                feats.append(backbone(x.to(device)).cpu())
                labels.append(y)
        out[f'{split}_X'] = torch.cat(feats).float()
        out[f'{split}_y'] = torch.cat(labels).long()
    torch.save(out, cache_path)
    return cache_path


class FeatureTensorDataset(Dataset):
    """(feature, label) pairs addressed through an index list."""

    def __init__(self, X, y, indices):
        self.X, self.y = X, y
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return self.X[j], self.y[j]


def _split_train_val(indices, val_fraction, seed):
    """Deterministic disjoint train/val split of a client's shard."""
    rng = np.random.default_rng(seed)
    order = list(indices)
    rng.shuffle(order)
    n_val = max(1, int(len(order) * val_fraction))
    return sorted(order[n_val:]), sorted(order[:n_val])


def create_cached_federated_datasets(cache_path, n_domains=5,
                                     clients_per_domain=3,
                                     dirichlet_alpha=0.5, batch_size=64,
                                     seed=42, val_fraction=0.1):
    """Same contract as create_federated_datasets, over cached features,
    with an extra per-client 'val_loader' (held-out slice of the TRAIN
    shard; used for cross-evaluation affinities — never test data)."""
    cache = torch.load(cache_path)
    train_y = cache['train_y'].numpy()
    test_y = cache['test_y'].numpy()

    clients_data = {}
    client_id = 0
    for domain_id in range(n_domains):
        domain_classes = get_domain_classes(domain_id)
        train_indices = np.where(np.isin(train_y, domain_classes))[0]
        test_indices = np.where(np.isin(test_y, domain_classes))[0]

        splits = partition_domain_data_dirichlet(
            train_indices, train_y, clients_per_domain,
            alpha=dirichlet_alpha, seed=seed + domain_id)
        test_per_client = np.array_split(test_indices, clients_per_domain)

        for local_id in range(clients_per_domain):
            tr_idx, val_idx = _split_train_val(
                splits[local_id], val_fraction, seed=seed * 1000 + client_id)
            mk = lambda idx, sh: DataLoader(
                FeatureTensorDataset(cache['train_X'], cache['train_y'], idx)
                if sh != 'test' else
                FeatureTensorDataset(cache['test_X'], cache['test_y'], idx),
                batch_size=batch_size, shuffle=(sh == 'train'))
            clients_data[client_id] = {
                'train_loader': mk(tr_idx, 'train'),
                'val_loader': mk(val_idx, 'val'),
                'test_loader': mk(test_per_client[local_id].tolist(), 'test'),
                'domain_id': domain_id,
                'domain_name': DOMAIN_NAMES[domain_id],
                'n_samples': len(tr_idx),
                'classes': domain_classes,
            }
            client_id += 1

    full_test_loader = DataLoader(
        FeatureTensorDataset(cache['test_X'], cache['test_y'],
                             range(len(test_y))),
        batch_size=256, shuffle=False)
    return clients_data, full_test_loader


def create_lora_head(num_classes=100, rank=16, alpha=32, in_features=512,
                     seed=0, device='cpu'):
    """LoRA head equivalent to the full model's fc, for cached features.

    The frozen base Linear is seeded so all clients share the same random
    frozen fc (the image pipeline achieves this by cloning one base model).
    Registered as module 'fc' so get_lora_state returns the key 'fc'.
    """
    g = torch.Generator().manual_seed(seed)
    base = nn.Linear(in_features, num_classes)
    with torch.no_grad():
        base.weight.copy_(torch.empty_like(base.weight).normal_(
            0, 0.01, generator=g))
        base.bias.zero_()
    head = LoRALinear(base, rank=rank, alpha=alpha)
    model = nn.Sequential(OrderedDict([('fc', head)]))
    return model.to(device)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_feature_cache.py -q`
Expected: `5 passed`

Note: `create_lora_head` seeds the base Linear explicitly but the LoRA A init still uses global RNG — that is fine (experiments set `torch.manual_seed` before building, and clients are clones of one head, mirroring `clone_model`). If `test_lora_head_base_frozen_and_shared_across_seeds` fails on A-matrix mismatch, it is comparing base weights only — keep it that way.

- [ ] **Step 5: One-time cache build + real-data smoke check** (downloads CIFAR-100 on first run; ~2 min on MPS)

Run:
```bash
python - <<'EOF'
import sys; sys.path.insert(0, '.')
import torch
from src.data.feature_cache import compute_backbone_features, create_cached_federated_datasets
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
path = compute_backbone_features(device=device)
clients, full = create_cached_federated_datasets(path)
assert len(clients) == 15
assert all('val_loader' in c for c in clients.values())
x, y = next(iter(clients[0]['train_loader']))
print('feature batch:', x.shape, 'labels:', y.shape, '-> ok')
EOF
```
Expected: `feature batch: torch.Size([64, 512]) labels: torch.Size([64]) -> ok`

- [ ] **Step 6: Commit**

```bash
git add project-3-hierarchical-gossip/src/data/feature_cache.py project-3-hierarchical-gossip/tests/test_feature_cache.py
git commit -m "Add frozen-backbone feature cache with per-client val split"
```

---

### Task 5: Config + shared experiment utilities

**Files:**
- Modify: `project-3-hierarchical-gossip/configs/default_config.yaml`
- Create: `project-3-hierarchical-gossip/experiments/common.py`
- Test: `project-3-hierarchical-gossip/tests/test_common.py`

**Interfaces:**
- Produces (used by Tasks 6, 7):
  - config keys: `training.n_rounds: 50`, `training.seeds: [42, 43, 44]` (existing `training.seed: 42` kept for the legacy scripts), `evaluation: {eval_every: 1, consensus_rank: 16}`, `cache: {enabled: true, path: ./data/resnet18_features.pt, val_fraction: 0.1}`
  - `common.load_config(path=None) -> dict`
  - `common.build_cached_clients(config, seed, device) -> (clients: list[FederatedClient], clients_data, full_test_loader)` — builds one seeded `create_lora_head` template, clones per client (identical init across clients, like `clone_model`), wraps in `FederatedClient` (passing `val_loader` through as attribute `client.val_loader`)
  - `common.make_consensus_fn(template_model, full_test_loader, device) -> callable(lora_state) -> accuracy`
  - `common.evaluate_model(model, loader, device) -> {'accuracy', 'loss'}`
  - `common.summarize_seeds(per_seed_finals: list[dict]) -> {'mean': ..., 'std': ...}`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_common.py
import torch

from experiments.common import evaluate_model, make_consensus_fn, summarize_seeds
from src.data.feature_cache import FeatureTensorDataset, create_lora_head
from torch.utils.data import DataLoader


def _toy_loader():
    X = torch.randn(32, 512)
    y = torch.randint(0, 100, (32,))
    return DataLoader(FeatureTensorDataset(X, y, range(32)), batch_size=16)


def test_evaluate_model_returns_accuracy_and_loss():
    model = create_lora_head(seed=0)
    out = evaluate_model(model, _toy_loader(), device='cpu')
    assert 0.0 <= out['accuracy'] <= 1.0
    assert out['loss'] > 0.0


def test_consensus_fn_loads_state_and_evaluates():
    template = create_lora_head(seed=0)
    donor = create_lora_head(seed=0)
    with torch.no_grad():
        donor.fc.lora_B.normal_()
    from src.models.lora_resnet import get_lora_state
    fn = make_consensus_fn(template, _toy_loader(), device='cpu')
    acc = fn(get_lora_state(donor))
    assert 0.0 <= acc <= 1.0


def test_summarize_seeds():
    s = summarize_seeds([{'acc': 0.1}, {'acc': 0.2}, {'acc': 0.3}])
    assert abs(s['mean']['acc'] - 0.2) < 1e-9
    assert s['std']['acc'] > 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_common.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.common'` (add `experiments/__init__.py` if pytest cannot import the package — create it empty in that case)

- [ ] **Step 3: Update default_config.yaml** (append/modify these keys, preserving all existing ones)

```yaml
training:
  # ... existing keys unchanged, including seed: 42 ...
  n_rounds: 50          # was 2 (smoke); experiments override via --rounds
  seeds: [42, 43, 44]

evaluation:
  eval_every: 1
  consensus_rank: 16

cache:
  enabled: true
  path: ./data/resnet18_features.pt
  val_fraction: 0.1
```

- [ ] **Step 4: Write experiments/common.py**

```python
# experiments/common.py
"""Shared plumbing for experiments 04+: config, cached clients, consensus eval."""

import copy
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.feature_cache import (
    compute_backbone_features,
    create_cached_federated_datasets,
    create_lora_head,
)
from src.federated.client import FederatedClient
from src.models.lora_resnet import set_lora_state


def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'configs', 'default_config.yaml')
    with open(path) as f:
        return yaml.safe_load(f)


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_cached_clients(config, seed, device):
    """Cached-feature clients with identical head init (clone semantics)."""
    cache_path = config['cache']['path']
    if not os.path.exists(cache_path):
        compute_backbone_features(device=device, cache_path=cache_path)
    clients_data, full_test_loader = create_cached_federated_datasets(
        cache_path,
        n_domains=config['data']['n_domains'],
        clients_per_domain=config['data']['clients_per_domain'],
        dirichlet_alpha=config['data']['dirichlet_alpha'],
        batch_size=config['data']['batch_size'],
        seed=seed,
        val_fraction=config['cache']['val_fraction'],
    )
    torch.manual_seed(seed)
    template = create_lora_head(
        num_classes=config['model']['num_classes'],
        rank=config['model']['lora_rank'],
        alpha=config['model']['lora_alpha'],
        seed=seed,
        device=device,
    )
    clients = []
    for cid, data in clients_data.items():
        model = copy.deepcopy(template).to(device)
        client = FederatedClient(
            client_id=cid, model=model,
            train_loader=data['train_loader'],
            test_loader=data['test_loader'],
            domain_id=data['domain_id'],
            lr=config['training']['learning_rate'],
            local_epochs=config['training']['local_epochs'],
            device=device,
        )
        client.val_loader = data['val_loader']
        clients.append(client)
    return clients, clients_data, full_test_loader


@torch.no_grad()
def evaluate_model(model, loader, device='cpu'):
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss()
    correct, total, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    return {'accuracy': correct / total, 'loss': loss_sum / total}


def make_consensus_fn(template_model, full_test_loader, device='cpu'):
    """Callable(lora_state) -> accuracy of the consensus adapter on the
    full test set. Uses a dedicated deepcopy so client models are untouched."""
    probe = copy.deepcopy(template_model)

    def consensus_fn(lora_state):
        set_lora_state(probe, lora_state)
        return evaluate_model(probe, full_test_loader, device)['accuracy']

    return consensus_fn


def summarize_seeds(per_seed_finals):
    keys = per_seed_finals[0].keys()
    mean = {k: float(np.mean([r[k] for r in per_seed_finals])) for k in keys}
    std = {k: float(np.std([r[k] for r in per_seed_finals])) for k in keys}
    return {'mean': mean, 'std': std}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_common.py -q`
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add project-3-hierarchical-gossip/configs/default_config.yaml project-3-hierarchical-gossip/experiments/common.py project-3-hierarchical-gossip/tests/test_common.py
git commit -m "Add multi-seed config and shared experiment utilities"
```

---

### Task 6: Experiment 04 — protocol-correct baseline battery

**Files:**
- Create: `project-3-hierarchical-gossip/experiments/04_baseline_battery.py`
- Create (by running it): `project-3-hierarchical-gossip/results/experiment_04_baseline_battery/{summary.json, history_seed*.json, README.md, personalized_curves.png, consensus_curves.png}`

**Interfaces:**
- Consumes: everything from Tasks 1–5.
- Produces: `summary.json` with, per arm, per-seed final metrics + mean/std for `personalized_acc`, `consensus_acc`, `fairness_gap` (max−min per-domain personalized), `participation` (fraction of clients beating the local-only arm's same-client accuracy), `bytes_total`. Gate G2 verdict block. This file feeds `paper/main.tex` Table `\todo{--}` cells for the homogeneous-rank rows.

Arms (all rank 16, alpha 32, ring topology unless noted):
1. `local_only` — no communication (participation floor).
2. `fedavg` — server ΔW-merge (sample-weighted) broadcast each round; consensus skyline. Personalized protocol = global model on each client's shard.
3. `gossip_factor` — DecentralizedRunner, `merge='factor'`, MH mixing (legacy semantics, protocol-corrected).
4. `gossip_deltaw` — DecentralizedRunner, `merge='delta_w'`, MH mixing (H1 arm).

- [ ] **Step 1: Write the driver**

```python
# experiments/04_baseline_battery.py
"""Experiment 4: protocol-correct baseline battery (50 rounds x 3 seeds).

Arms: local_only | fedavg | gossip_factor | gossip_deltaw
Both evaluation protocols are recorded for every arm:
  personalized: each client's model on its own domain test shard
  consensus:    uniform delta-W average at rank 16 on the full test set
Run:  python -m experiments.04_baseline_battery [--rounds N] [--seeds 42 43 44]
      [--topology ring] [--arms local_only fedavg gossip_factor gossip_deltaw]
"""

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.common import (
    build_cached_clients, evaluate_model, load_config,
    make_consensus_fn, pick_device, summarize_seeds,
)
from src.federated.merge import merge_states
from src.federated.runner import DecentralizedRunner
from src.models.lora_resnet import get_lora_state, set_lora_state

RESULTS_DIR = './results/experiment_04_baseline_battery'


def run_local_only(clients, n_rounds, consensus_fn):
    history = {'avg_accuracy': [], 'per_client_accuracy': [],
               'per_domain_accuracy': [], 'consensus_accuracy': [],
               'bytes_cumulative': []}
    for _ in range(n_rounds):
        for c in clients:
            c.train()
        per_client = {c.client_id: c.evaluate()['accuracy'] for c in clients}
        per_domain = {}
        for c in clients:
            per_domain.setdefault(c.domain_id, []).append(per_client[c.client_id])
        history['per_client_accuracy'].append(per_client)
        history['per_domain_accuracy'].append(
            {d: float(np.mean(v)) for d, v in per_domain.items()})
        history['avg_accuracy'].append(float(np.mean(list(per_client.values()))))
        states = [c.get_lora_state() for c in clients]
        merged, _ = merge_states(states, [1 / len(states)] * len(states), 16, 32)
        history['consensus_accuracy'].append(consensus_fn(merged))
        history['bytes_cumulative'].append(0)
    return history


def run_fedavg(clients, n_rounds, consensus_fn, alpha=32, rank=16):
    n_samples = np.array([len(c.train_loader.dataset) for c in clients], float)
    weights = (n_samples / n_samples.sum()).tolist()
    history = {'avg_accuracy': [], 'per_client_accuracy': [],
               'per_domain_accuracy': [], 'consensus_accuracy': [],
               'bytes_cumulative': []}
    total_bytes = 0
    for _ in range(n_rounds):
        for c in clients:
            c.train()
        states = [c.get_lora_state() for c in clients]
        global_state, _ = merge_states(states, weights, rank, alpha)
        # upload + download of factors per client per round
        per_state = sum(4 * (p['A'].numel() + p['B'].numel())
                        for p in states[0].values())
        total_bytes += 2 * per_state * len(clients)
        for c in clients:
            c.set_lora_state(copy.deepcopy(global_state))
        per_client = {c.client_id: c.evaluate()['accuracy'] for c in clients}
        per_domain = {}
        for c in clients:
            per_domain.setdefault(c.domain_id, []).append(per_client[c.client_id])
        history['per_client_accuracy'].append(per_client)
        history['per_domain_accuracy'].append(
            {d: float(np.mean(v)) for d, v in per_domain.items()})
        history['avg_accuracy'].append(float(np.mean(list(per_client.values()))))
        history['consensus_accuracy'].append(consensus_fn(global_state))
        history['bytes_cumulative'].append(total_bytes)
    return history


def finals(history, local_only_final_per_client=None):
    last_pc = history['per_client_accuracy'][-1]
    last_pd = history['per_domain_accuracy'][-1]
    out = {
        'personalized_acc': float(np.mean(list(last_pc.values()))),
        'consensus_acc': history['consensus_accuracy'][-1],
        'fairness_gap': float(max(last_pd.values()) - min(last_pd.values())),
        'bytes_total': history['bytes_cumulative'][-1],
    }
    if local_only_final_per_client is not None:
        wins = [1 for cid, acc in last_pc.items()
                if acc >= local_only_final_per_client[cid]]
        out['participation'] = len(wins) / len(last_pc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rounds', type=int, default=None)
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--topology', default='ring')
    ap.add_argument('--arms', nargs='+',
                    default=['local_only', 'fedavg', 'gossip_factor', 'gossip_deltaw'])
    args = ap.parse_args()

    config = load_config()
    n_rounds = args.rounds or config['training']['n_rounds']
    seeds = args.seeds or config['training']['seeds']
    device = pick_device()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"device={device} rounds={n_rounds} seeds={seeds} arms={args.arms}")

    all_finals = {arm: [] for arm in args.arms}
    for seed in seeds:
        local_final_pc = None
        histories = {}
        for arm in args.arms:
            torch.manual_seed(seed)
            clients, _, full_test = build_cached_clients(config, seed, device)
            consensus_fn = make_consensus_fn(clients[0].model, full_test, device)
            if arm == 'local_only':
                h = run_local_only(clients, n_rounds, consensus_fn)
                local_final_pc = h['per_client_accuracy'][-1]
            elif arm == 'fedavg':
                h = run_fedavg(clients, n_rounds, consensus_fn,
                               alpha=config['model']['lora_alpha'],
                               rank=config['model']['lora_rank'])
            else:
                merge = 'factor' if arm == 'gossip_factor' else 'delta_w'
                runner = DecentralizedRunner(
                    clients, topology=args.topology, merge=merge,
                    alpha=config['model']['lora_alpha'], seed=seed,
                    consensus_fn=consensus_fn)
                h = runner.run(n_rounds, verbose=True)
            histories[arm] = h
            all_finals[arm].append(finals(h, local_final_pc))
            print(f"  seed {seed} {arm}: {all_finals[arm][-1]}")
        with open(os.path.join(RESULTS_DIR, f'history_seed{seed}.json'), 'w') as f:
            json.dump(histories, f)

    summary = {arm: {'per_seed': runs, **summarize_seeds(runs)}
               for arm, runs in all_finals.items()}
    g2 = {
        'deltaw_vs_factor_personalized':
            summary['gossip_deltaw']['mean']['personalized_acc']
            - summary['gossip_factor']['mean']['personalized_acc'],
        'fedavg_consensus_minus_gossip_consensus':
            summary['fedavg']['mean']['consensus_acc']
            - summary['gossip_deltaw']['mean']['consensus_acc'],
    }
    summary['gate_G2'] = g2
    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke run (2 rounds, 1 seed) before the real run**

Run: `python -m experiments.04_baseline_battery --rounds 2 --seeds 42`
Expected: completes without error; `results/experiment_04_baseline_battery/summary.json` exists with all four arms and a `gate_G2` block; personalized accuracies > 0.01 (above chance) for the communicating arms.

- [ ] **Step 3: Full run (50 rounds × 3 seeds)**

Run: `python -m experiments.04_baseline_battery`
Expected: finishes in well under an hour with the feature cache. Sanity expectations (not hard assertions): `local_only` personalized comfortably above chance and likely the strongest personalized arm on in-domain shards; `fedavg` the strongest consensus arm; `gossip_deltaw ≈ gossip_factor` on personalized accuracy at homogeneous rank (shared init ⇒ small gauge drift; the H1 gap is expected under rank heterogeneity in Phase 4, so a null here does NOT refute H1); tail_mass small but nonzero.

- [ ] **Step 4: Write the results README**

Create `results/experiment_04_baseline_battery/README.md` with: the arms table (personalized AND consensus, mean±std over 3 seeds), fairness gap, participation, bytes; the feature-cache protocol note (no augmentation; numbers not comparable to experiments 01–02); the G2 verdict paragraph (state explicitly whether any inversion occurred and what it means for Phase 3); the explicit statement that personalized-vs-consensus cross-arm comparisons are the only valid ones (per `paper/main.tex` protocol correction).

- [ ] **Step 5: Commit**

```bash
git add project-3-hierarchical-gossip/experiments/04_baseline_battery.py project-3-hierarchical-gossip/results/experiment_04_baseline_battery/
git commit -m "Add Experiment 04 protocol-correct baseline battery"
```

---

### Task 7: Direction-aware domain signatures (`signatures.py`)

**Files:**
- Create: `project-3-hierarchical-gossip/src/clustering/signatures.py`
- Test: `project-3-hierarchical-gossip/tests/test_signatures.py`

**Interfaces:**
- Consumes: `lora_state` dicts; `scaled_delta_w` from Task 1; a probe tensor `[n_probe, 512]` (cached test features, inputs only); `client.val_loader` from Task 4/5.
- Produces (used by Task 8 and Phase 3):
  - `signature_row_norms(lora_state, alpha) -> np.ndarray[out_features]` — L2 norm of each ΔW row, L1-normalized (S1; ~“which classes moved”)
  - `signature_subspace(lora_state, alpha, k=8) -> np.ndarray[in_features, k]` — top-k right singular vectors of ΔW (S2)
  - `subspace_affinity(V1, V2) -> float` — `||V1ᵀ V2||_F² / k ∈ [0, 1]` (mean squared principal-angle cosine; gauge- and basis-invariant)
  - `signature_probe_logits(model, probe_x, device='cpu') -> np.ndarray[n_probe * n_classes]` (S3; functional signature)
  - `cross_eval_utility(client, neighbor_state, n_batches=2) -> float` — accuracy of the neighbor's adapter on the client's val batches, own state restored afterward (S4)
  - `signature_delta_vec(lora_state, alpha) -> np.ndarray[out*in]` — flattened ΔW (S5; feeds the inverse-L2 affinity that Listo Zec et al. found to be the only surviving signature on pretrained backbones: `aff = 1/(1+||d_i−d_j||₂)`, min-max rescaled)
  - `affinity_matrix(signatures: list, kind: 'cosine' | 'subspace' | 'inv_l2') -> np.ndarray[n, n]` symmetric, diagonal 1
  - `cluster_from_affinity(affinity, n_clusters=5) -> np.ndarray[n]` — AgglomerativeClustering(metric='precomputed', linkage='average') on distance = 1 − affinity

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_signatures.py
import numpy as np
import torch

from sklearn.metrics import adjusted_rand_score

from src.clustering.signatures import (
    affinity_matrix,
    cluster_from_affinity,
    signature_row_norms,
    signature_subspace,
    subspace_affinity,
)


def block_state(rows, out_f=20, in_f=30, rank=4, seed=0):
    """Adapter whose ΔW has mass only on `rows` (a synthetic 'domain')."""
    g = torch.Generator().manual_seed(seed)
    B = torch.zeros(out_f, rank)
    for r, row in enumerate(rows):
        B[row, r % rank] = 1.0 + 0.1 * torch.randn(1, generator=g).item()
    A = torch.randn(rank, in_f, generator=g)
    return {'fc': {'A': A, 'B': B}}


def test_row_norms_localize_on_active_rows():
    state = block_state(rows=[0, 1, 2, 3])
    v = signature_row_norms(state, alpha=32)
    assert v.shape == (20,)
    assert abs(v.sum() - 1.0) < 1e-6
    assert v[:4].sum() > 0.99  # nearly all mass on the active rows


def test_subspace_affinity_gauge_invariant_and_bounded():
    s = block_state(rows=[0, 1, 2, 3], seed=1)
    V = signature_subspace(s, alpha=32, k=3)
    q, _ = torch.linalg.qr(torch.randn(4, 4))
    rot = {'fc': {'A': q @ s['fc']['A'], 'B': s['fc']['B'] @ q.T}}
    V2 = signature_subspace(rot, alpha=32, k=3)
    a = subspace_affinity(V, V2)
    assert 0.99 < a <= 1.0 + 1e-9  # same ΔW -> same subspace


def test_two_synthetic_domains_recovered_from_row_norm_affinity():
    domain_a = [block_state([0, 1, 2, 3], seed=i) for i in range(4)]
    domain_b = [block_state([10, 11, 12, 13], seed=10 + i) for i in range(4)]
    sigs = [signature_row_norms(s, 32) for s in domain_a + domain_b]
    aff = affinity_matrix(sigs, kind='cosine')
    labels = cluster_from_affinity(aff, n_clusters=2)
    truth = [0] * 4 + [1] * 4
    assert adjusted_rand_score(truth, labels) == 1.0


def test_affinity_matrix_symmetric_unit_diagonal():
    sigs = [signature_row_norms(block_state([0, 1], seed=i), 32) for i in range(3)]
    aff = affinity_matrix(sigs, kind='cosine')
    assert np.allclose(aff, aff.T)
    assert np.allclose(np.diag(aff), 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_signatures.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.clustering.signatures'`

- [ ] **Step 3: Write signatures.py**

```python
# src/clustering/signatures.py
"""Direction-aware client signatures for serverless domain discovery.

Experiment 03 showed that singular-VALUE features (domain_clustering.py)
do not recover domain structure: spectra are direction-blind, and on an
fc-only LoRA the domain identity lives in WHICH rows (classes) and WHICH
subspaces of ΔW carry mass. These signatures keep that information:

  S1 row_norms    : per-class row-norm profile of ΔW      (out_features floats)
  S2 subspace     : top-k right singular vectors of ΔW    (in_features x k)
  S3 probe_logits : logits on a small shared probe set    (functional)
  S4 cross_eval   : accuracy of a peer's adapter on local val data (utility)

Privacy note (for the paper): S1 approximates the client's label
distribution; S3/S4 leak only function outputs. Quantify in Phase 5.
"""

import copy

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from src.federated.merge import scaled_delta_w


def _stacked_delta(lora_state, alpha):
    deltas = scaled_delta_w(lora_state, alpha)
    return torch.cat([deltas[layer] for layer in sorted(deltas)], dim=0)


def signature_row_norms(lora_state, alpha):
    """L1-normalized per-row L2 norms of ΔW ('which classes moved')."""
    d = _stacked_delta(lora_state, alpha)
    norms = torch.linalg.norm(d, dim=1).numpy()
    total = norms.sum()
    return norms / total if total > 0 else norms


def signature_subspace(lora_state, alpha, k=8):
    """Top-k right singular vectors of ΔW (input directions that changed)."""
    d = _stacked_delta(lora_state, alpha)
    _, _, Vh = torch.linalg.svd(d, full_matrices=False)
    return Vh[:k].T.numpy()  # [in_features, k]


def subspace_affinity(V1, V2):
    """Mean squared principal-angle cosine: ||V1ᵀV2||_F² / k in [0, 1]."""
    k = V1.shape[1]
    return float(np.linalg.norm(V1.T @ V2, 'fro') ** 2 / k)


@torch.no_grad()
def signature_probe_logits(model, probe_x, device='cpu'):
    """Flattened logits on a fixed shared probe batch (inputs only)."""
    model = model.to(device).eval()
    return model(probe_x.to(device)).cpu().numpy().ravel()


@torch.no_grad()
def cross_eval_utility(client, neighbor_state, n_batches=2):
    """Accuracy of `neighbor_state` on the client's held-out val batches.
    The client's own state is restored afterward. Zero extra communication
    during gossip: the neighbor state was already received for merging."""
    own = client.get_lora_state()
    client.set_lora_state(copy.deepcopy(neighbor_state))
    client.model.eval()
    correct, total = 0, 0
    for b, (x, y) in enumerate(client.val_loader):
        if b >= n_batches:
            break
        out = client.model(x.to(client.device))
        correct += (out.argmax(1) == y.to(client.device)).sum().item()
        total += y.size(0)
    client.set_lora_state(own)
    return correct / total if total else 0.0


def signature_delta_vec(lora_state, alpha):
    """Flattened ΔW (for inverse-L2 affinity; Listo Zec et al. 2024 found
    L2 weight distance the only signature surviving a pretrained backbone)."""
    return _stacked_delta(lora_state, alpha).numpy().ravel()


def affinity_matrix(signatures, kind='cosine'):
    n = len(signatures)
    aff = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            if kind == 'cosine':
                a, b = signatures[i], signatures[j]
                denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
                v = float(np.dot(a, b) / denom)
            elif kind == 'subspace':
                v = subspace_affinity(signatures[i], signatures[j])
            elif kind == 'inv_l2':
                v = float(1.0 / (1.0 + np.linalg.norm(signatures[i] - signatures[j])))
            else:
                raise ValueError(kind)
            aff[i, j] = aff[j, i] = v
    if kind == 'inv_l2':  # rescale off-diagonal to [0, 1] for comparability
        off = ~np.eye(n, dtype=bool)
        lo, hi = aff[off].min(), aff[off].max()
        if hi > lo:
            aff[off] = (aff[off] - lo) / (hi - lo)
        np.fill_diagonal(aff, 1.0)
    return aff


def cluster_from_affinity(affinity, n_clusters=5):
    dist = 1.0 - affinity
    np.fill_diagonal(dist, 0.0)
    model = AgglomerativeClustering(
        n_clusters=n_clusters, metric='precomputed', linkage='average')
    return model.fit_predict(dist)
```

Note: if the installed scikit-learn predates the `metric=` rename, use `affinity='precomputed'` — check with `python -c "import sklearn; print(sklearn.__version__)"` and match the API.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_signatures.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add project-3-hierarchical-gossip/src/clustering/signatures.py project-3-hierarchical-gossip/tests/test_signatures.py
git commit -m "Add direction-aware domain signatures (row norms, subspace, probe, cross-eval)"
```

---

### Task 8: Experiment 05 — signature validation (Gate G1)

**Files:**
- Create: `project-3-hierarchical-gossip/experiments/05_signature_validation.py`
- Create (by running): `project-3-hierarchical-gossip/results/experiment_05_signature_validation/{signature_results.json, README.md, ari_trajectories.png}`

**Interfaces:**
- Consumes: Tasks 1–7; `extract_lora_features` + `cluster_clients` from `src/clustering/domain_clustering.py` (spectral baseline, exact Experiment-03 reproduction for continuity).
- Produces: per (training_mode ∈ {local_only, gossip_deltaw}) × (signature ∈ {spectral_baseline, row_norms, subspace_k8, probe_logits, cross_eval}) × (stage ∈ {2, 5, 10, 20}): ARI and NMI vs true domains. Gate G1 verdict in JSON + README. This is the RQ1 experiment for `paper/main.tex` (currently a written-in negative result — this either overturns it with a signature fix or hardens it).

Design notes baked into the driver:
- Stages replicate `03_clustering_validation.py:136` (`[2, 5, 10, 20]`), trained incrementally, same seeds `[42, 43, 44]`.
- The probe set: 256 cached TEST features chosen with a fixed `np.random.default_rng(0)` sample — inputs only, labels never touched.
- `cross_eval` affinity: full 15×15 matrix via `cross_eval_utility` (cheap on cached features), symmetrized `0.5 (U + Uᵀ)`, then min-max normalized to [0, 1] before `cluster_from_affinity`.
- Under `gossip_deltaw` mode, signatures are computed AFTER the mix step each stage — this measures whether mixing homogenizes away separability (it must not, or Phase-3 clustering can never work mid-protocol).

- [ ] **Step 1: Write the driver**

```python
# experiments/05_signature_validation.py
"""Experiment 5: can direction-aware signatures recover domain structure
where spectral features (Experiment 03) failed?

Grid: {local_only, gossip_deltaw} x {spectral_baseline, row_norms,
subspace_k8, probe_logits, cross_eval} x stages [2, 5, 10, 20] x 3 seeds.
Gate G1: max ARI at stage <= 10 decides hard vs soft vs pivot (see plan).
Run: python -m experiments.05_signature_validation [--seeds 42 43 44] [--stages 2 5 10 20]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from experiments.common import build_cached_clients, load_config, pick_device
from src.clustering.domain_clustering import cluster_clients
from src.clustering.signatures import (
    affinity_matrix, cluster_from_affinity, cross_eval_utility,
    signature_delta_vec, signature_probe_logits, signature_row_norms,
    signature_subspace,
)
from src.federated.runner import DecentralizedRunner

RESULTS_DIR = './results/experiment_05_signature_validation'
SIGNATURES = ['spectral_baseline', 'row_norms', 'subspace_k8',
              'probe_logits', 'cross_eval', 'delta_inv_l2']


def make_probe(config, n_probe=256):
    cache = torch.load(config['cache']['path'])
    idx = np.random.default_rng(0).choice(
        cache['test_X'].shape[0], size=n_probe, replace=False)
    return cache['test_X'][idx]


def compute_labels(signature, clients, alpha, probe_x, device):
    ids = [c.client_id for c in clients]
    if signature == 'spectral_baseline':
        states = {c.client_id: c.get_lora_state() for c in clients}
        _, assignments, _ = cluster_clients(states, n_clusters=5)
        return [assignments[cid] for cid in ids]
    if signature == 'row_norms':
        sigs = [signature_row_norms(c.get_lora_state(), alpha) for c in clients]
        aff = affinity_matrix(sigs, 'cosine')
    elif signature == 'subspace_k8':
        sigs = [signature_subspace(c.get_lora_state(), alpha, k=8) for c in clients]
        aff = affinity_matrix(sigs, 'subspace')
    elif signature == 'probe_logits':
        sigs = [signature_probe_logits(c.model, probe_x, device) for c in clients]
        aff = affinity_matrix(sigs, 'cosine')
    elif signature == 'cross_eval':
        n = len(clients)
        U = np.zeros((n, n))
        states = [c.get_lora_state() for c in clients]
        for i, ci in enumerate(clients):
            for j in range(n):
                U[i, j] = 1.0 if i == j else cross_eval_utility(ci, states[j])
        aff = 0.5 * (U + U.T)
        lo, hi = aff.min(), aff.max()
        aff = (aff - lo) / (hi - lo) if hi > lo else aff
        np.fill_diagonal(aff, 1.0)
    elif signature == 'delta_inv_l2':
        sigs = [signature_delta_vec(c.get_lora_state(), alpha) for c in clients]
        aff = affinity_matrix(sigs, 'inv_l2')
    return list(cluster_from_affinity(aff, n_clusters=5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=None)
    ap.add_argument('--stages', type=int, nargs='+', default=[2, 5, 10, 20])
    args = ap.parse_args()

    config = load_config()
    seeds = args.seeds or config['training']['seeds']
    device = pick_device()
    alpha = config['model']['lora_alpha']
    os.makedirs(RESULTS_DIR, exist_ok=True)
    probe_x = make_probe(config)

    records = []
    for mode in ['local_only', 'gossip_deltaw']:
        for seed in seeds:
            torch.manual_seed(seed)
            clients, _, _ = build_cached_clients(config, seed, device)
            true = {c.client_id: c.domain_id for c in clients}
            truth = [true[c.client_id] for c in clients]
            runner = (DecentralizedRunner(clients, topology='ring',
                                          merge='delta_w', alpha=alpha, seed=seed)
                      if mode == 'gossip_deltaw' else None)
            prev = 0
            for stage in args.stages:
                for _ in range(stage - prev):
                    for c in clients:
                        c.train()
                    if runner is not None:
                        runner.mix_round()
                prev = stage
                for sig in SIGNATURES:
                    pred = compute_labels(sig, clients, alpha, probe_x, device)
                    records.append({
                        'mode': mode, 'seed': seed, 'stage': stage,
                        'signature': sig,
                        'ari': adjusted_rand_score(truth, pred),
                        'nmi': normalized_mutual_info_score(truth, pred),
                    })
                    print(records[-1])

    by = {}
    for r in records:
        key = f"{r['mode']}|{r['signature']}|{r['stage']}"
        by.setdefault(key, []).append(r['ari'])
    aggregated = {k: {'ari_mean': float(np.mean(v)), 'ari_std': float(np.std(v))}
                  for k, v in by.items()}
    early = {k: v['ari_mean'] for k, v in aggregated.items()
             if int(k.split('|')[2]) <= 10 and not k.endswith('spectral_baseline|2')}
    best_key = max(early, key=early.get)
    best = early[best_key]
    verdict = ('HARD_CLUSTERING_VIABLE' if best >= 0.7 else
               'SOFT_AFFINITY_ONLY' if best >= 0.4 else 'PIVOT_RICHER_ADAPTERS')
    out = {'records': records, 'aggregated': aggregated,
           'gate_G1': {'best_signature_by_stage10': best_key,
                       'best_ari': best, 'verdict': verdict}}
    with open(os.path.join(RESULTS_DIR, 'signature_results.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print('\nGATE G1:', json.dumps(out['gate_G1'], indent=2))


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke run (1 seed, stages 2 5)**

Run: `python -m experiments.05_signature_validation --seeds 42 --stages 2 5`
Expected: completes; every signature produces ARI/NMI records for both modes; `spectral_baseline` ARI near 0 (reproducing Experiment 03's failure); at least one of `row_norms`/`probe_logits`/`cross_eval` visibly above the spectral baseline by stage 5.

- [ ] **Step 3: Full run**

Run: `python -m experiments.05_signature_validation`
Expected: `signature_results.json` with a `gate_G1` verdict. Plot ARI trajectories (matplotlib, one panel per mode, one line per signature, mean over seeds) into `ari_trajectories.png` — add the plotting block at the end of `main()` if not already emitted.

- [ ] **Step 4: Write the results README** — table of ARI mean±std per (mode, signature, stage); explicit comparison to Experiment 03's spectral numbers (0.031 → −0.140); the G1 verdict and what it selects in Phase 3; privacy caveats for S1 (label-distribution proxy) and S3/S4 (function leakage only); note that the gossip-mode rows measure discoverability *mid-protocol*, which is what Phase 3 actually consumes.

- [ ] **Step 5: Update `paper/main.tex` RQ1 subsection stub** — do NOT rewrite claims yet; add one `\todo{}` note pointing at `experiment_05_signature_validation/signature_results.json` as the pending evidence that may overturn the written-in negative result. (Full paper integration is Phase 5.)

- [ ] **Step 6: Commit**

```bash
git add project-3-hierarchical-gossip/experiments/05_signature_validation.py project-3-hierarchical-gossip/results/experiment_05_signature_validation/ paper/main.tex
git commit -m "Add Experiment 05 signature validation with G1 gate"
```

---

## Phase 3–5 roadmap (planned after gates G1/G2 — each gets its own plan doc)

These are designed but deliberately not task-decomposed yet, because their shape depends on the gate outcomes. Interfaces named here are commitments; details will be planned with fresh evidence.

**Phase 3 — the contribution (Experiment 06).** Two methods over `DecentralizedRunner`'s `mixing_matrix` seam:
- *Soft (primary, robust to G1=SOFT):* `AffinityWeightedGossip` — per-round neighbor affinities from an EMA over `cross_eval_utility` (zero extra messages) and/or the G1-winning signature; row weights `softmax(affinity / τ)` with a self-weight floor `w_ii ≥ w_min` (poisoning/dissimilar-peer guard), then `sinkhorn_project` toward doubly-stochastic. Knobs: τ, w_min, EMA β.
- *Hard (if G1=HARD):* two-tier hierarchical gossip — every node clusters the gossiped signatures locally (15 clients ⇒ every node can hold all signatures; push-sum generalization noted as future work), intra-cluster MH mixing every round, representative bridges every k rounds with transfer weights `T_kl = softmax(v_l(k) / κ)` (paper main.tex:693-698).
- Arms: best Phase-1 gossip, AWG, hard-hierarchical (discovered), **oracle-hierarchical (ground-truth clusters — decouples discovery quality from topology benefit)**, IFCA-style server clustering (non-serverless skyline). Report both protocols + bytes; matched-communication comparisons.

**Phase 4 — heterogeneous ranks + P1/P2 integration (Experiment 07).** Per-client ranks from P2's `weighted_assignment` output format and P1's `BATCH_TO_MAX_RANK` mapping (contract documented in `04_allocation_comparison.py:113`); budget-matched spreads (total 240 = 15×16): uniform-16 vs {4,8,16,32,64} mixtures. Arms: ΔW-merge vs HetLoRA-style zero-padding (add `merge='zero_pad'` mode to the runner: pad factors to the neighborhood max rank, average, truncate). This is where H1's gap is expected to open. Plus dynamic re-rank at round 25 (refactorize to the new rank locally — O(1), no peer coordination) demonstrating the paper's "rank changes are local" claim. Fills RQ3/RQ4 cells.

**Phase 5 — theory + paper.** `src/analysis/convergence_diagnostics.py`: spectral gap of realized mixing matrices (`spectral_gap`), per-round `mean_tail_mass` (the ε_r term — already logged by the runner), consensus-distance trajectory, empirical 1/√T fit. Frame per-round truncation as a contractive/biased compression of the state: CHOCO-SGD (Koloskova et al. 2019) covers biased compressors with quality ω ≤ 1 in gossip, and Beznosikov–Horváth–Richtárik–Safaryan (JMLR 2023) provides the biased-compressor taxonomy + error-feedback rate template — composing the two is the theorem skeleton for main.tex. **Optional algorithmic arm from this framing: error feedback for truncation** — each client accumulates its refactorization residual (ΔW̄ − ΔW_refactorized) locally and re-injects it next round; no LoRA-gossip system in the survey does this, so it is both a theory-alignment device and a potential contribution.

**Novelty map & references (from the 2026-07-20 deep-research survey — fold into main.tex related work):**
- *Closest threat:* **DeCAF** (Zhang et al., arXiv 2505.21382; Neural Networks) — decentralized gossip LoRA with truncated-SVD refactorization after each consensus round, i.e. our H1 mechanism already exists in flat, homogeneous-rank form. Our deltas: heterogeneous per-client ranks, undeclared-domain discovery, affinity/hierarchical topology, dynamic rank, both-protocol evaluation. Read in full before writing the contribution list.
- *Flat serverless LoRA baselines:* **Dec-LoRA** (Ghiasvand et al., arXiv 2501.15361; ACL 2025 REALM wkshp) — factor-space gossip, homogeneous rank, O(1/√T); **ADF-LoRA** (arXiv 2511.18291) — alternating-factor gossip, DS mixing, documents phase-mismatch failure of naive alternation under gossip.
- *Merge-operator lineage (server-based):* **HetLoRA** (Cho et al., EMNLP 2024 main, pp. 12903–12913 — CONFIRMED; fix references.bib) zero-pad+truncate, self-pruning; **FlexLoRA** (Bai et al., arXiv 2402.11505) ΔW-average + per-client truncated-SVD redistribution — port both to gossip as comparators; **FLoRA** (Wang et al., arXiv 2409.05976) stacking aggregation + proof that factor averaging introduces cross-client interference (supports our Prop. 1; its zero-padding baseline collapses 29.5%→7.97% MMLU — the effect size H1 predicts under heterogeneity). NOTE: "NeurIPS 2024" attributions for FlexLoRA/FLoRA in references.bib are UNCONFIRMED — verify against proceedings or cite arXiv.
- *Serverless clustering / affinity:* **DFCA** (arXiv 2510.15300) decentralized IFCA, functional self-assignment, k known, full models; **DAC** (Listo Zec et al. 2022) soft decentralized clustering via inverse-loss, argues against hard clusters; its follow-up similarity study (2024) = our signature negative-result precedent; **PFedDST** (arXiv 2502.07750) composite peer score; **L2C** (Li et al., CVPR 2022) learned mixing weights — the main threat to "learned transfer weights", differentiate via LoRA-rank-awareness + two-tier communication budget; **cFedLoRA** (ADMA 2025, Springer LNAI 16197 pp. 191–205) server-side clustering of LoRA updates.
- *Novelty support:* the IJCAI-25 FedLoRA survey (Yang et al., Survey Track pp. 10779–10787) contains **no gossip/peer-to-peer methods at all** — the serverless heterogeneous-LoRA intersection in main.tex Table 1 stands, provided DeCAF is added and differentiated.

Fill all result tables from experiment JSONs; verify every `references.bib` entry flagged "VERIFY" against the list above before submission.

## Self-review checklist (run after writing, fixed inline)

1. **Spec coverage:** H1 → Tasks 1, 3, 6 (+Phase 4 for the hetero gap); H2 → Tasks 7, 8; protocol integrity → Tasks 2, 3, 5, 6; speed → Task 4; gates → Tasks 6, 8. Phases 3–5 explicitly deferred with named interfaces.
2. **Placeholder scan:** no TBDs in Tasks 0–8; deferred work lives only in the clearly-labeled roadmap section.
3. **Type consistency:** `lora_state = {layer: {'A': [r, in], 'B': [out, r]}}` everywhere; `merge_states(states, weights, target_rank, alpha)` signature identical in Tasks 1, 3, 6; `build_topology(client_ids, topology, seed)` identical in Tasks 2, 3; `clients_data[cid]['val_loader']` produced in Task 4, consumed in Tasks 5 (attribute pass-through) and 7/8 (`client.val_loader`).
