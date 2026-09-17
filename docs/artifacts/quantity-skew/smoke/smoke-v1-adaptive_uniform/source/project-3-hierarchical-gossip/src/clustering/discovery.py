"""Online, label-free domain discovery for decentralized LoRA gossip.

The benchmark previously exposed signatures only as an offline diagnostic.  This
module turns the strongest practical signal (the effective ``Delta W`` direction)
into a stateful mixer: clients publish no labels, each round's adapters are
smoothed with an EMA, an affinity graph is built, and the cluster count is
selected by silhouette score.  The resulting soft matrix is projected to a
symmetric doubly-stochastic mixer, so it can be passed directly to
``DecentralizedRunner``.
"""

from dataclasses import dataclass
import numpy as np

from src.clustering.signatures import affinity_matrix, cluster_from_affinity, signature_delta_vec
from src.federated.hierarchical import affinity_mixing
from src.federated.mixing import build_topology


@dataclass
class DiscoverySnapshot:
    labels: np.ndarray
    affinity: np.ndarray
    n_clusters: int
    confidence: float
    signature_dimension: int = 0


def _choose_k(affinity, min_clusters=2, max_clusters=None):
    """Choose K without labels using the highest average silhouette score."""
    from sklearn.metrics import silhouette_score
    a = np.asarray(affinity, dtype=float)
    n = len(a)
    if n < 3:
        return 1, 0.0
    lo = max(2, int(min_clusters))
    hi = min(n - 1, int(max_clusters or min(8, n - 1)))
    if lo > hi:
        return 1, 0.0
    distance = np.maximum(1.0 - a, 0.0)
    best_k, best_score = lo, -1.0
    for k in range(lo, hi + 1):
        labels = cluster_from_affinity(a, k)
        if len(np.unique(labels)) < 2:
            continue
        score = float(silhouette_score(distance, labels, metric="precomputed"))
        if score > best_score + 1e-12:
            best_k, best_score = k, score
    # Negative silhouettes indicate that the graph has no convincing partition.
    return (best_k if best_score > 0.0 else 1), max(0.0, best_score)


class OnlineDomainDiscovery:
    """Stateful, label-free discovery suitable for ``mixing_fn`` callbacks.

    ``update(states, alpha, client_ids)`` must be called before requesting the
    matrix for a round. ``AdaptiveAffinityMixer`` does this automatically when
    used with :class:`src.federated.runner.DecentralizedRunner`. The current
    runner callback supplies the full client state list, so this implementation
    models a coordinator-visible signature observation; neighborhood-local
    exchange accounting remains future work.
    """

    def __init__(self, beta=0.8, min_clusters=2, max_clusters=8,
                 temperature=0.5, self_weight=0.2, projection_dim=64, projection_seed=0):
        if not 0.0 <= float(beta) < 1.0:
            raise ValueError("beta must be in [0, 1)")
        if not float(temperature) > 0:
            raise ValueError("temperature must be > 0")
        if not 0.0 <= float(self_weight) < 1.0:
            raise ValueError("self_weight must be in [0, 1)")
        if int(projection_dim) < 1:
            raise ValueError("projection_dim must be >= 1")
        self.beta = float(beta)
        self.min_clusters = int(min_clusters)
        self.max_clusters = int(max_clusters)
        self.temperature = float(temperature)
        self.self_weight = float(self_weight)
        self.projection_dim = int(projection_dim)
        self.projection_seed = int(projection_seed)
        self._projection = None
        self._ema = None
        self.snapshot = None

    def update(self, states, alpha, client_ids=None):
        vectors = np.asarray([signature_delta_vec(s, alpha) for s in states], dtype=float)
        if vectors.ndim != 2 or vectors.shape[0] < 1 or not np.isfinite(vectors).all():
            raise ValueError("states must contain at least one finite, compatible adapter")
        # Project the flattened update to a compact, deterministic sketch before
        # sharing/affinity computation. This bounds discovery metadata to
        # ``projection_dim`` floats per client while retaining update direction.
        input_dim = vectors.shape[1]
        out_dim = min(self.projection_dim, input_dim)
        if self._projection is None or self._projection.shape != (input_dim, out_dim):
            rng = np.random.default_rng(self.projection_seed)
            self._projection = rng.normal(0.0, 1.0 / np.sqrt(out_dim), size=(input_dim, out_dim))
        vectors = vectors @ self._projection
        # Per-client L2 normalisation removes rank/optimizer scale and retains
        # the update direction. EMA reduces stage-to-stage cluster flicker.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.maximum(norms, 1e-12)
        self._ema = vectors if self._ema is None else self.beta * self._ema + (1.0 - self.beta) * vectors
        self._ema /= np.maximum(np.linalg.norm(self._ema, axis=1, keepdims=True), 1e-12)
        affinity = affinity_matrix([v for v in self._ema], kind="cosine")
        if len(vectors) == 1:
            self.snapshot = DiscoverySnapshot(np.zeros(1, dtype=int), affinity, 1, 0.0,
                                              signature_dimension=out_dim)
            return self.snapshot
        n_clusters, confidence = _choose_k(affinity, self.min_clusters, self.max_clusters)
        labels = cluster_from_affinity(affinity, n_clusters) if n_clusters > 1 else np.zeros(len(vectors), dtype=int)
        self.snapshot = DiscoverySnapshot(labels, affinity, n_clusters, confidence, signature_dimension=out_dim)
        return self.snapshot

    def matrix(self, client_ids, topology="fully_connected", topology_client_ids=None):
        if self.snapshot is None:
            n = len(client_ids)
            return np.eye(n, dtype=float)
        topology_ids = list(client_ids if topology_client_ids is None else topology_client_ids)
        if set(topology_ids) != set(client_ids) or len(topology_ids) != len(client_ids):
            raise ValueError("topology_client_ids must be a permutation of client_ids")
        neighbors = build_topology(topology_ids, topology=topology)
        return affinity_mixing(self.snapshot.affinity, neighbors,
                               tau=self.temperature, w_min=self.self_weight,
                               client_ids=client_ids)

    def assignments(self, client_ids):
        """Return discovered cluster labels keyed by client id.

        This is intentionally an explicit opt-in for hard hierarchy. Callers
        should use ``snapshot.confidence`` as a gate and fall back to ``matrix``
        when confidence is low.
        """
        if self.snapshot is None:
            return {cid: 0 for cid in client_ids}
        if len(client_ids) != len(self.snapshot.labels):
            raise ValueError("client_ids length does not match the latest snapshot")
        return {cid: int(label) for cid, label in zip(client_ids, self.snapshot.labels)}


class AdaptiveAffinityMixer:
    """Callable mixer that lets ``DecentralizedRunner`` discover online."""

    def __init__(self, client_ids, alpha, discovery=None, topology="fully_connected", topology_client_ids=None):
        self.client_ids = list(client_ids)
        self.topology_client_ids = list(self.client_ids if topology_client_ids is None else topology_client_ids)
        if set(self.topology_client_ids) != set(self.client_ids) or len(self.topology_client_ids) != len(self.client_ids):
            raise ValueError("topology_client_ids must be a permutation of client_ids")
        self.alpha = float(alpha)
        self.discovery = discovery or OnlineDomainDiscovery()
        self.topology = topology

    def update(self, states, alpha=None, client_ids=None):
        self.discovery.update(states, self.alpha if alpha is None else alpha,
                              self.client_ids if client_ids is None else client_ids)

    def __call__(self, round_idx):
        return self.discovery.matrix(self.client_ids, self.topology, self.topology_client_ids)
