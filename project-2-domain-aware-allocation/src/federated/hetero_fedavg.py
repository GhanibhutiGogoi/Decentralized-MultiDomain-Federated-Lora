"""
Heterogeneous FedAvg: Federated Averaging with mixed LoRA ranks.

Unlike standard FedAvg which requires all clients to have the same LoRA rank,
this server aggregates in delta_W space (B @ A), which is rank-independent.
After averaging, it SVD-decomposes back to each client's assigned rank.

Aggregation pipeline:
    1. Each client trains locally and sends LoRA params
    2. Convert each client's LoRA (A, B) to delta_W = B @ A
    3. Weighted average of delta_W matrices (all same shape)
    4. For each client, SVD-decompose avg_delta_W to client's rank
    5. Send rank-specific LoRA params back to each client
"""

import copy
import torch
from tqdm import tqdm

from src.models.lora_resnet import (
    get_lora_state,
    set_lora_state,
    merge_lora_to_delta_w,
    decompose_delta_w,
)


class HeteroFedAvgServer:
    """
    Central server for Federated Averaging with heterogeneous LoRA ranks.

    Each client can have a different LoRA rank. Aggregation happens in the
    rank-independent delta_W space, then decomposes back per client.
    """

    def __init__(self, clients, rank_assignments, alpha=32, device='cpu'):
        """
        Args:
            clients: list of FederatedClient objects
            rank_assignments: dict mapping client_id -> rank
            alpha: LoRA alpha (for decomposition scaling)
            device: compute device
        """
        self.clients = clients
        self.rank_assignments = rank_assignments
        self.alpha = alpha
        self.device = device
        self.history = {
            'rounds': [],
            'avg_loss': [],
            'avg_accuracy': [],
            'per_domain_accuracy': [],
            'per_client_accuracy': [],
        }

    def aggregate_delta_w(self, client_delta_ws, client_weights):
        """
        Weighted average in delta_W space.

        All delta_W = B @ A have identical shape (out_features x in_features)
        regardless of the rank used to produce them.

        Args:
            client_delta_ws: list of delta_W dicts
            client_weights: list of weights (typically n_samples)

        Returns:
            averaged delta_W dict
        """
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        agg_delta_w = {}
        for layer_name in client_delta_ws[0]:
            agg_delta_w[layer_name] = torch.zeros_like(
                client_delta_ws[0][layer_name]
            )

        for delta_w, weight in zip(client_delta_ws, normalized_weights):
            for layer_name in agg_delta_w:
                agg_delta_w[layer_name] += weight * delta_w[layer_name]

        return agg_delta_w

    def run(self, n_rounds=50, verbose=True):
        """
        Run heterogeneous FedAvg for n_rounds.

        Returns:
            history dict with per-round metrics
        """
        iterator = tqdm(range(n_rounds), desc="HeteroFedAvg") if verbose else range(n_rounds)

        for round_idx in iterator:
            # 1. Local training
            client_states = []
            client_weights = []

            for client in self.clients:
                metrics = client.train()
                client_states.append(client.get_lora_state())
                client_weights.append(metrics['n_samples'])

            # 2. Convert to delta_W space
            client_delta_ws = [
                merge_lora_to_delta_w(state) for state in client_states
            ]

            # 3. Aggregate in delta_W space
            avg_delta_w = self.aggregate_delta_w(client_delta_ws, client_weights)

            # 4. Decompose back to each client's rank and distribute
            for client in self.clients:
                target_rank = self.rank_assignments[client.client_id]
                client_lora = decompose_delta_w(
                    avg_delta_w, target_rank, alpha=self.alpha
                )
                client.set_lora_state(client_lora)

            # 5. Evaluate
            eval_results = self._evaluate_all_clients()

            self.history['rounds'].append(round_idx)
            self.history['avg_loss'].append(eval_results['avg_loss'])
            self.history['avg_accuracy'].append(eval_results['avg_accuracy'])
            self.history['per_domain_accuracy'].append(eval_results['per_domain'])
            self.history['per_client_accuracy'].append(eval_results['per_client'])

            if verbose:
                domain_str = " | ".join(
                    f"D{d}:{acc:.3f}"
                    for d, acc in sorted(eval_results['per_domain'].items())
                )
                iterator.set_postfix({
                    'acc': f"{eval_results['avg_accuracy']:.3f}",
                })
                if (round_idx + 1) % 10 == 0:
                    print(f"\n  Round {round_idx+1} | {domain_str}")

        return self.history

    def _evaluate_all_clients(self):
        """Evaluate each client with its current LoRA state."""
        per_client = {}
        per_domain = {}
        domain_counts = {}

        for client in self.clients:
            metrics = client.evaluate()
            per_client[client.client_id] = metrics['accuracy']

            did = client.domain_id
            if did not in per_domain:
                per_domain[did] = 0.0
                domain_counts[did] = 0
            per_domain[did] += metrics['accuracy']
            domain_counts[did] += 1

        for did in per_domain:
            per_domain[did] /= domain_counts[did]

        avg_accuracy = sum(per_client.values()) / len(per_client)

        return {
            'avg_accuracy': avg_accuracy,
            'avg_loss': 0.0,
            'per_domain': per_domain,
            'per_client': per_client,
        }
