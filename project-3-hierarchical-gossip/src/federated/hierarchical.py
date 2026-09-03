"""Hierarchical and affinity-weighted mixing for decentralized LoRA gossip.

Two constructions, both returning symmetric doubly stochastic matrices, so the
mean-preservation and spectral-gap arguments in
docs/research/2026-09-03-project3-convergence-analysis.md apply to them
unchanged:

1. `affinity_mixing` -- soft. Each client weights every neighbour by
   exp(a_ij / tau), the result is projected to doubly stochastic by Sinkhorn
   iteration, and a self-weight floor w_min is enforced by a convex combination
   with the identity. There are no discrete clusters anywhere, which is what
   makes it robust when domains overlap or when discovery is noisy.

2. `two_tier_mixing` -- hard. Given cluster assignments, every round mixes
   densely inside each cluster; every `bridge_every` rounds one representative
   per cluster additionally mixes with the other representatives under a
   transfer matrix T. Far fewer messages than flat gossip, at the price of
   needing clusters to exist.

Cluster assignments and affinities are INPUTS here, never discovered. That is
deliberate: it lets an oracle-cluster arm answer "does hierarchy help?"
separately from "can we find the clusters?", which is gate G1.

Matrices are indexed by `client_ids` when given and by sorted key otherwise,
matching `mixing.metropolis_hastings`.
"""

import numpy as np

from src.federated.mixing import _validate, build_topology, metropolis_hastings


# ---------------------------------------------------------------------------
# Sinkhorn projection
# ---------------------------------------------------------------------------

SINKHORN_TOL = 1e-10


def sinkhorn(matrix, iters=20000, tol=SINKHORN_TOL):
    """Project a non-negative matrix to doubly stochastic by alternating
    row and column normalisation.

    Converges whenever the zero pattern has total support, which holds for any
    symmetric pattern with a strictly positive diagonal: the identity covers
    the diagonal, and each off-diagonal pair (i, j), (j, i) lies on the
    transposition permutation. Both mixers below guarantee that pattern.

    The output is doubly stochastic **to within `tol`**, not exactly -- so a
    quantity that is exactly conserved under an analytic doubly stochastic
    matrix (the network mean under Metropolis-Hastings, say) is conserved here
    to about `tol * max|x|` per round. `tol` matches the check in
    `is_doubly_stochastic`, so a returned matrix passes it.

    The rate slows as the kernel becomes nearly decomposable -- when a low
    temperature makes some subset of nodes want to exchange almost all their
    mass among themselves, the iteration behaves like a Markov chain crossing
    a bottleneck. Each sweep is O(n^2), which at a few dozen clients is
    microseconds, so the default budget is generous rather than tight;
    `affinity_mixing` refuses kernels past 35 nats of spread outright.

    A symmetric input converges to a symmetric fixed point D M D; the result is
    symmetrised exactly at the end so callers can rely on W == W.T rather than
    on it holding to within `tol`.
    """
    w = np.array(matrix, dtype=float, copy=True)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("sinkhorn input must be finite")
    if (w < 0).any():
        raise ValueError("sinkhorn input must be non-negative")
    if (w.sum(axis=1) <= 0).any() or (w.sum(axis=0) <= 0).any():
        raise ValueError("sinkhorn input needs a positive entry in every row and column")

    symmetric_input = np.allclose(w, w.T, atol=1e-12)
    for _ in range(iters):
        w /= w.sum(axis=1, keepdims=True)
        w /= w.sum(axis=0, keepdims=True)
        row_err = np.abs(w.sum(axis=1) - 1.0).max()
        col_err = np.abs(w.sum(axis=0) - 1.0).max()
        if max(row_err, col_err) <= tol:
            break
    else:
        raise RuntimeError(
            f"sinkhorn did not reach tolerance {tol} in {iters} iterations "
            f"(row error {row_err:.3e}, column error {col_err:.3e})"
        )
    if symmetric_input:
        w = 0.5 * (w + w.T)
    return w


# ---------------------------------------------------------------------------
# Soft: affinity-weighted mixing
# ---------------------------------------------------------------------------

def affinity_mixing(affinity, neighbors, tau=1.0, w_min=0.1, client_ids=None,
                    sinkhorn_iters=20000):
    """Symmetric doubly stochastic mixing weighted by pairwise affinity.

        K_ij = exp((a_ij - a_max) / tau)   for j in N(i) or j == i, else 0
        W0   = Sinkhorn(K)
        W    = w_min * I + (1 - w_min) * W0

    Args:
        affinity: (N, N) array of pairwise affinities in `client_ids` order.
            Only the entries on the neighbour support are used. Symmetrised as
            (A + A.T) / 2 -- an asymmetric affinity would give a non-symmetric
            W, and the convergence argument needs symmetry.
        neighbors: symmetric neighbour map, as produced by `build_topology`.
        tau: temperature. Small tau concentrates weight on the most similar
            neighbours; large tau approaches uniform weighting on the support.
        w_min: self-weight floor in [0, 1). Applied as a convex combination
            with the identity, which keeps W symmetric and doubly stochastic
            and guarantees W_ii >= w_min. It also shrinks the spectral gap by
            a factor (1 - w_min): the price of never fully trusting peers.
        client_ids: row/column order. Defaults to sorted(neighbors).

    Returns:
        W of shape (N, N).
    """
    if not float(tau) > 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if not 0.0 <= float(w_min) < 1.0:
        raise ValueError(f"w_min must be in [0, 1), got {w_min}")
    _validate(neighbors)
    order = list(client_ids) if client_ids is not None else sorted(neighbors)
    if len(order) != len(set(order)):
        raise ValueError("client_ids must be unique")
    if set(order) != set(neighbors):
        raise ValueError("client_ids must name exactly the clients in the neighbour map")

    n = len(order)
    # The soft mixer assumes one connected graph. A disconnected map still
    # yields a valid doubly stochastic W (block-diagonal), but its spectral gap
    # is exactly zero and consensus never crosses the components -- a silent
    # trap, not an error, unless refused here. (two_tier_mixing is exempt: its
    # per-round matrix is block-diagonal by design and the bridge connects it.)
    seen, frontier = {order[0]}, [order[0]]
    while frontier:
        for peer in neighbors[frontier.pop()]:
            if peer not in seen:
                seen.add(peer)
                frontier.append(peer)
    if len(seen) != n:
        raise ValueError(
            f"neighbour map is disconnected ({len(seen)} of {n} clients reachable from "
            f"{order[0]!r}); affinity mixing on it would have spectral gap 0"
        )
    a = np.asarray(affinity, dtype=float)
    if a.shape != (n, n):
        raise ValueError(f"affinity must have shape ({n}, {n}), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("affinity must be finite")
    a = 0.5 * (a + a.T)

    index = {cid: k for k, cid in enumerate(order)}
    support = np.eye(n, dtype=bool)
    for cid in order:
        for peer in neighbors[cid]:
            support[index[cid], index[peer]] = True

    # One global shift, not a per-row one: a per-row shift would break the
    # symmetry that Sinkhorn needs to return a symmetric matrix.
    shift = a[support].max()
    spread_in_nats = (shift - a[support].min()) / float(tau)
    # Sinkhorn converges geometrically only while the kernel's dynamic range
    # is moderate. A spread of 35 nats means the smallest supported entry is
    # ~1e-15 of the largest, and past that the iteration crawls toward a
    # near-permutation rather than a mixing matrix. Say so, and name the knob.
    if spread_in_nats > 35.0:
        raise ValueError(
            f"tau={tau} is too small for this affinity range: the kernel would span "
            f"{spread_in_nats:.0f} nats (max/min ratio ~1e{spread_in_nats / 2.3026:.0f}), "
            "which Sinkhorn cannot balance. Raise tau or rescale the affinities."
        )
    kernel = np.zeros((n, n))
    kernel[support] = np.exp((a[support] - shift) / float(tau))

    w0 = sinkhorn(kernel, iters=sinkhorn_iters)
    return float(w_min) * np.eye(n) + (1.0 - float(w_min)) * w0


# ---------------------------------------------------------------------------
# Hard: two-tier hierarchical mixing
# ---------------------------------------------------------------------------

def clusters_from_assignments(assignments, client_ids=None):
    """Group client ids by cluster label, in `client_ids` order within each
    cluster. Returns {cluster_label: [client_id, ...]} with labels sorted."""
    order = list(client_ids) if client_ids is not None else sorted(assignments)
    if set(order) != set(assignments):
        raise ValueError("client_ids must name exactly the clients in assignments")
    groups = {}
    for cid in order:
        groups.setdefault(assignments[cid], []).append(cid)
    return {label: groups[label] for label in sorted(groups)}


def _embed(small, members, index, n):
    """Place a |members| x |members| block into an N x N identity."""
    w = np.eye(n)
    rows = [index[c] for c in members]
    w[np.ix_(rows, rows)] = small
    return w


def two_tier_mixing(assignments, round_idx, bridge_every=5, transfer=None,
                    representatives=None, client_ids=None, intra_topology="fully_connected",
                    bridge_tau=1.0, sinkhorn_iters=20000):
    """Symmetric doubly stochastic mixing matrix for round `round_idx` of a
    two-tier protocol.

    Every round:   W_intra  = block-diagonal Metropolis-Hastings, one block per
                              cluster on `intra_topology` over its members.
    Bridge rounds: W_bridge = affinity mixing among one representative per
                              cluster on a complete graph, with affinity T;
                              identity for every other client.
                   W        = W_intra @ W_bridge @ W_intra

    The sandwich keeps W symmetric (W_intra is symmetric) and doubly
    stochastic (a product of doubly stochastic matrices is doubly stochastic).
    Under the default complete-graph intra topology W_intra is uniform
    averaging within each cluster, which is idempotent, so a bridge round
    moves each cluster's mean toward the other clusters' means and then spreads
    the result back to every member -- exactly the "cluster-level gossip" the
    design intends.

    A round is a bridge round when `round_idx % bridge_every == bridge_every - 1`,
    so the first bridge happens after the clusters have mixed internally.

    Args:
        assignments: {client_id: cluster_label}. An input, never discovered here.
        round_idx: 0-based round counter.
        bridge_every: period of inter-cluster exchange; 1 bridges every round.
        transfer: (K, K) transfer affinities T_kl between clusters, in sorted
            label order. None means uniform. Symmetrised.
        representatives: {cluster_label: client_id}. Defaults to the first
            member of each cluster in `client_ids` order.
        client_ids: row/column order for the returned matrix.
        intra_topology: topology name for within-cluster mixing.
        bridge_tau: temperature for the bridge affinity mixing.

    Returns:
        W of shape (N, N) in `client_ids` order.
    """
    if int(bridge_every) < 1:
        raise ValueError(f"bridge_every must be >= 1, got {bridge_every}")
    order = list(client_ids) if client_ids is not None else sorted(assignments)
    if len(order) != len(set(order)):
        raise ValueError("client_ids must be unique")
    groups = clusters_from_assignments(assignments, client_ids=order)
    labels = list(groups)
    n = len(order)
    index = {cid: k for k, cid in enumerate(order)}

    # --- intra-cluster tier ---------------------------------------------
    w_intra = np.zeros((n, n))
    for label in labels:
        members = groups[label]
        if len(members) == 1:
            w_intra[index[members[0]], index[members[0]]] = 1.0
            continue
        block = metropolis_hastings(build_topology(members, intra_topology), client_ids=members)
        rows = [index[c] for c in members]
        w_intra[np.ix_(rows, rows)] = block

    if int(round_idx) % int(bridge_every) != int(bridge_every) - 1 or len(labels) == 1:
        return w_intra

    # --- inter-cluster tier (bridge round) -------------------------------
    if representatives is None:
        representatives = {label: groups[label][0] for label in labels}
    reps = []
    for label in labels:
        rep = representatives[label]
        if assignments[rep] != label:
            raise ValueError(f"representative {rep!r} is not a member of cluster {label!r}")
        reps.append(rep)

    k = len(labels)
    if transfer is None:
        t = np.ones((k, k))
    else:
        t = np.asarray(transfer, dtype=float)
        if t.shape != (k, k):
            raise ValueError(f"transfer must have shape ({k}, {k}) for {k} clusters, got {t.shape}")
        if not np.all(np.isfinite(t)):
            raise ValueError("transfer must be finite")

    rep_neighbors = build_topology(reps, "fully_connected")
    w_small = affinity_mixing(t, rep_neighbors, tau=bridge_tau, w_min=0.0,
                              client_ids=reps, sinkhorn_iters=sinkhorn_iters)
    w_bridge = _embed(w_small, reps, index, n)
    return w_intra @ w_bridge @ w_intra


def window_product(mixing_fn, start_round, length):
    """Product W_{t+L-1} ... W_{t+1} W_t of a time-varying mixing sequence.

    For a periodic two-tier schedule the relevant object for consensus is not
    any single round's matrix (which is block-diagonal and has no cross-cluster
    gap at all) but the product over one full period, which is what
    `spectral_gap` should be measured on.
    """
    if int(length) < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    product = None
    for r in range(int(start_round), int(start_round) + int(length)):
        w = np.asarray(mixing_fn(r), dtype=float)
        product = w if product is None else w @ product
    return product


def is_doubly_stochastic(w, atol=1e-10):
    """True when rows and columns each sum to 1 and all entries are >= 0."""
    w = np.asarray(w, dtype=float)
    return (
        w.ndim == 2 and w.shape[0] == w.shape[1]
        and np.allclose(w.sum(axis=1), 1.0, atol=atol)
        and np.allclose(w.sum(axis=0), 1.0, atol=atol)
        and bool((w >= -atol).all())
    )
