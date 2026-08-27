"""Doubly stochastic mixing matrices for decentralized gossip.

`gossip.py` has every client average 50/50 toward *one neighbour it picks
itself*. The induced mixing matrix W therefore satisfies sum_j W[i][j] == 1
(row-stochastic) but in general sum_i W[i][j] != 1, because nothing balances
the reverse flow: a node several peers happen to choose receives more mass
than it sends. The network mean (1/N) sum_i x_i is consequently not preserved
across a round, so the protocol converges -- but not to the network average --
and the spectral-gap argument every standard gossip convergence result relies
on does not apply to it.

The Metropolis-Hastings construction below is symmetric and doubly stochastic
by construction, which makes the mean an exact invariant and gives a spectral
gap that can be measured and quoted.

    W[i][j] = 1 / (1 + max(deg_i, deg_j))        j a neighbour of i
    W[i][i] = 1 - sum_{j != i} W[i][j]

The diagonal is always strictly positive: each off-diagonal term is at most
1/(1 + deg_i), so W[i][i] >= 1 - deg_i/(1 + deg_i) = 1/(1 + deg_i) > 0.

Matrices are returned as numpy arrays whose row/column order is `client_ids`
when given and `sorted(neighbors)` otherwise, since client ids need not be
0..N-1.
"""

import numpy as np

TOPOLOGIES = ("ring", "fully_connected", "star", "path")


def build_topology(client_ids, topology="ring", seed=42):
    """Build a symmetric neighbour map over `client_ids`.

    Args:
        client_ids: ordered sequence of client identifiers.
        topology: 'ring', 'fully_connected', 'star' or 'path'. The latter two
            are irregular (their nodes have differing degrees), which is what
            makes the max(deg_i, deg_j) in `metropolis_hastings` bite -- on a
            regular graph that max is a no-op and the construction is
            indistinguishable from the naive 1/(1+deg_i) rule.
        seed: accepted for interface parity with future randomised topologies;
            unused by the deterministic ones below.

    Returns:
        dict client_id -> list of neighbour ids, symmetric, no self-loops and
        no duplicates.
    """
    ids = list(client_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("client_ids must be unique")
    if topology not in TOPOLOGIES:
        raise ValueError(
            f"unknown topology {topology!r}; expected one of {TOPOLOGIES}"
        )

    n = len(ids)
    if topology == "fully_connected":
        return {cid: [o for o in ids if o != cid] for cid in ids}

    if topology == "star":
        if n == 1:
            return {ids[0]: []}
        hub, leaves = ids[0], ids[1:]
        neighbors = {hub: list(leaves)}
        neighbors.update({leaf: [hub] for leaf in leaves})
        return neighbors

    if topology == "path":
        neighbors = {}
        for i, cid in enumerate(ids):
            peers = []
            if i > 0:
                peers.append(ids[i - 1])
            if i < n - 1:
                peers.append(ids[i + 1])
            neighbors[cid] = peers
        return neighbors

    # ring -- dedupe because at n == 2 the two ring neighbours coincide, and
    # at n == 1 a node's only ring neighbour is itself.
    neighbors = {}
    for i, cid in enumerate(ids):
        peers = []
        for other in (ids[(i - 1) % n], ids[(i + 1) % n]):
            if other != cid and other not in peers:
                peers.append(other)
        neighbors[cid] = peers
    return neighbors


def _validate(neighbors):
    for cid, peers in neighbors.items():
        if cid in peers:
            raise ValueError(f"client {cid!r} lists itself as a neighbour: self-loops are not allowed")
        if len(peers) != len(set(peers)):
            raise ValueError(f"client {cid!r} has duplicate neighbours: {peers}")
        for peer in peers:
            if peer not in neighbors:
                raise ValueError(f"client {cid!r} names unknown neighbour {peer!r}")
            if cid not in neighbors[peer]:
                raise ValueError(
                    f"neighbour map is not symmetric: {cid!r} lists {peer!r} "
                    f"but {peer!r} does not list {cid!r}"
                )


def metropolis_hastings(neighbors, client_ids=None):
    """Symmetric doubly stochastic mixing matrix for a neighbour map.

    Args:
        neighbors: dict client_id -> list of neighbour ids. Must be symmetric,
            with no self-loops and no duplicates.
        client_ids: optional row/column order. Defaults to sorted(neighbors).

    Returns:
        numpy array W of shape (N, N).
    """
    _validate(neighbors)
    order = list(client_ids) if client_ids is not None else sorted(neighbors)
    if set(order) != set(neighbors):
        raise ValueError("client_ids must name exactly the clients in the neighbour map")

    index = {cid: k for k, cid in enumerate(order)}
    degree = {cid: len(peers) for cid, peers in neighbors.items()}
    n = len(order)
    w = np.zeros((n, n), dtype=float)

    for cid in order:
        i = index[cid]
        for peer in neighbors[cid]:
            w[i, index[peer]] = 1.0 / (1.0 + max(degree[cid], degree[peer]))
        w[i, i] = 1.0 - w[i].sum()

    # Guaranteed by the bound in the module docstring; assert rather than trust.
    if (np.diag(w) < -1e-12).any():
        raise ValueError("negative self-weight produced; neighbour map is malformed")
    return w


def spectral_gap(w):
    """1 - |lambda_2|, the gap that governs how fast disagreement decays.

    Zero for a disconnected graph, since the eigenvalue 1 is then repeated.
    """
    w = np.asarray(w, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"expected a square matrix, got shape {w.shape}")
    if w.shape[0] < 2:
        raise ValueError("spectral gap needs at least two nodes")

    # Metropolis-Hastings matrices are symmetric, where eigvalsh is both faster
    # and numerically better behaved; fall back to the general solver otherwise.
    if np.allclose(w, w.T, atol=1e-12):
        eigenvalues = np.linalg.eigvalsh(w)
    else:
        eigenvalues = np.linalg.eigvals(w)
    magnitudes = np.sort(np.abs(eigenvalues))[::-1]
    return float(1.0 - magnitudes[1])
