"""Small reproducible online-discovery demonstration.

This driver exercises the same stateful mixer used by ``DecentralizedRunner``
without requiring CIFAR downloads.  It is useful as a smoke test and as a
template for wiring discovery into a full benchmark: adapters are generated
from two latent domains, the number of groups is inferred from silhouette
score, and only the final labels are compared with the held-out truth.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clustering.discovery import OnlineDomainDiscovery


def make_state(rows, seed):
    generator = torch.Generator().manual_seed(int(seed))
    a = torch.randn(4, 8, generator=generator)
    b = torch.zeros(10, 4)
    for index, row in enumerate(rows):
        b[row, index % 4] = 1.0
    return {"fc": {"A": a, "B": b}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients-per-domain", type=int, default=5)
    parser.add_argument("--domains", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/online_discovery.json"))
    args = parser.parse_args(argv)
    if args.domains < 2 or args.clients_per_domain < 2:
        parser.error("domains and clients-per-domain must both be at least two")
    states, truth = [], []
    for domain in range(args.domains):
        # Disjoint row blocks represent domains in this controlled diagnostic.
        rows = list(range(domain * (10 // args.domains), (domain + 1) * (10 // args.domains)))
        for client in range(args.clients_per_domain):
            states.append(make_state(rows, args.seed + 101 * domain + client))
            truth.append(domain)
    discovery = OnlineDomainDiscovery(beta=0.5, max_clusters=min(8, len(states) - 1))
    snapshots = []
    for stage in range(1, 4):
        snapshot = discovery.update(states, alpha=32.0)
        snapshots.append({"stage": stage, "n_clusters": snapshot.n_clusters,
                          "confidence": snapshot.confidence,
                          "ari": adjusted_rand_score(truth, snapshot.labels),
                          "nmi": normalized_mutual_info_score(truth, snapshot.labels)})
    result = {"schema_version": 1, "status": "complete", "seed": args.seed,
              "domains": args.domains, "clients": len(states),
              "snapshots": snapshots,
              "final_affinity": discovery.snapshot.affinity.tolist(),
              "final_labels": discovery.snapshot.labels.tolist(),
              "truth_for_scoring_only": truth}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "snapshots")}, indent=2))


if __name__ == "__main__":
    main()
