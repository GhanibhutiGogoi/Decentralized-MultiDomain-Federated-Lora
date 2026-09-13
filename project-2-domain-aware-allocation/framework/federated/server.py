"""
Heterogeneous FedAvg: Federated Averaging with mixed LoRA ranks.

Unlike standard FedAvg which requires all clients to have the same LoRA rank,
this server aggregates in effective delta_W space ((alpha / rank) * B @ A).
After averaging, it SVD-decomposes back to each client's assigned rank.

Aggregation pipeline:
    1. Each client trains locally and sends LoRA params
    2. Convert each client's LoRA to delta_W = (alpha / rank) * B @ A
    3. Weighted average of delta_W matrices (all same shape)
    4. For each client, SVD-decompose avg_delta_W to client's rank
    5. Send rank-specific LoRA params back to each client
"""

import copy
import math
import torch
from tqdm import tqdm

from framework.models.lora_resnet import (
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
            alpha: shared client LoRA alpha (for source and destination scaling)
            device: compute device
        """
        self.clients = list(clients)
        self.rank_assignments = rank_assignments
        try:
            self.alpha = float(alpha)
        except (TypeError, ValueError) as exc:
            raise ValueError("alpha must be finite and positive") from exc
        self.device = device
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self._validate_shared_alpha()
        self.history = {
            'rounds': [],
            'avg_loss': [],
            'avg_accuracy': [],
            'per_domain_accuracy': [],
            'per_client_accuracy': [],
        }

    def aggregate_delta_w(self, client_delta_ws, client_weights):
        """Weighted average in effective ``delta_W`` space.

        Inputs are validated explicitly because malformed client payloads or
        weights otherwise produce silent NaNs or biased updates.
        """
        if client_delta_ws is None or len(client_delta_ws) == 0:
            raise ValueError("at least one client update is required")
        if client_weights is None or len(client_weights) != len(client_delta_ws):
            raise ValueError("client updates and weights must have matching lengths")

        weights = []
        for idx, weight in enumerate(client_weights):
            try:
                weight = float(weight)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"client weight at index {idx} is not numeric") from exc
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"client weight at index {idx} must be finite and nonnegative")
            weights.append(weight)
        total_weight = sum(weights)
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise ValueError("client weights must have a finite, positive total")

        reference = client_delta_ws[0]
        if not hasattr(reference, "keys"):
            raise ValueError("client updates must be mappings of layer names to tensors")
        layer_names = set(reference.keys())
        for idx, delta_w in enumerate(client_delta_ws):
            if not hasattr(delta_w, "keys") or set(delta_w.keys()) != layer_names:
                raise ValueError(f"client update at index {idx} has incompatible layer keys")
            for layer_name in layer_names:
                tensor = delta_w[layer_name]
                ref_tensor = reference[layer_name]
                if not isinstance(ref_tensor, torch.Tensor) or not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"client update for layer {layer_name!r} must be a tensor")
                if tensor.shape != ref_tensor.shape:
                    raise ValueError(f"incompatible tensor shape for layer {layer_name!r}")
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"non-finite update for layer {layer_name!r}")

        normalized_weights = [w / total_weight for w in weights]
        agg_delta_w = {
            layer_name: torch.zeros_like(reference[layer_name])
            for layer_name in layer_names
        }
        for delta_w, weight in zip(client_delta_ws, normalized_weights):
            for layer_name in layer_names:
                agg_delta_w[layer_name] += weight * delta_w[layer_name]
        return agg_delta_w

    @staticmethod
    def _client_alphas(client):
        """Discover alpha values exposed by a client or its LoRA modules."""
        values = []
        for obj in (client, getattr(client, "model", None)):
            if obj is None:
                continue
            if hasattr(obj, "alpha"):
                try:
                    values.append(float(obj.alpha))
                except (TypeError, ValueError) as exc:
                    raise ValueError("client alpha must be numeric") from exc
            named_modules = getattr(obj, "named_modules", None)
            if named_modules is not None:
                for _, module in named_modules():
                    if hasattr(module, "alpha"):
                        try:
                            values.append(float(module.alpha))
                        except (TypeError, ValueError) as exc:
                            raise ValueError("client alpha must be numeric") from exc
        return values

    def _validate_shared_alpha(self):
        discovered = [self._client_alphas(client) for client in self.clients]
        present = [value for values in discovered for value in values]
        if any(not values for values in discovered):
            raise ValueError("all clients must expose a shared alpha value")
        if any(not math.isfinite(value) or value <= 0 for value in present):
            raise ValueError("client alpha values must be finite and positive")
        for client, values in zip(self.clients, discovered):
            if any(not math.isclose(value, self.alpha, rel_tol=1e-6, abs_tol=1e-8) for value in values):
                raise ValueError(
                    f"client {getattr(client, 'client_id', '?')} alpha does not match server alpha {self.alpha}"
                )

    def run(self, n_rounds=50, verbose=True):
        """
        Run heterogeneous FedAvg for n_rounds.

        Returns:
            history dict with per-round metrics
        """
        self._validate_shared_alpha()
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
                merge_lora_to_delta_w(state, alpha=self.alpha) for state in client_states
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
