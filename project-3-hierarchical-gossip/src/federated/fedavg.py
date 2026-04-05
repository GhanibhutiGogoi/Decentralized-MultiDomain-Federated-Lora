"""
Federated Averaging (FedAvg) with LoRA - centralized baseline.

This serves as the upper-bound baseline: a central server coordinates
aggregation, which is the best-case scenario for convergence.
Our gossip-based approach aims to approach this performance without a server.
"""

import copy
import torch
from tqdm import tqdm

from src.models.lora_resnet import get_lora_state, set_lora_state


class FedAvgServer:
    """
    Central server for Federated Averaging with LoRA adapters.

    Protocol per round:
    1. Broadcast global LoRA params to all clients
    2. Each client trains locally for E epochs
    3. Server collects updated LoRA params
    4. Server aggregates via weighted average (weighted by n_samples)
    """

    def __init__(self, clients, global_model, device='cpu'):
        self.clients = clients
        self.global_model = global_model
        self.device = device
        self.history = {
            'rounds': [],
            'avg_loss': [],
            'avg_accuracy': [],
            'per_domain_accuracy': [],
            'per_client_accuracy': [],
        }

    def aggregate(self, client_states, client_weights):
        """
        Weighted average of LoRA parameters.

        Args:
            client_states: list of LoRA state dicts
            client_weights: list of weights (typically n_samples)
        """
        total_weight = sum(client_weights)
        normalized_weights = [w / total_weight for w in client_weights]

        # Initialize aggregated state with zeros
        agg_state = {}
        first_state = client_states[0]
        for layer_name, params in first_state.items():
            agg_state[layer_name] = {
                'A': torch.zeros_like(params['A']),
                'B': torch.zeros_like(params['B']),
            }

        # Weighted sum
        for state, weight in zip(client_states, normalized_weights):
            for layer_name in agg_state:
                agg_state[layer_name]['A'] += weight * state[layer_name]['A']
                agg_state[layer_name]['B'] += weight * state[layer_name]['B']

        return agg_state

    def run(self, n_rounds=50, verbose=True):
        """
        Run FedAvg for n_rounds.

        Returns:
            history dict with per-round metrics
        """
        iterator = tqdm(range(n_rounds), desc="FedAvg") if verbose else range(n_rounds)

        for round_idx in iterator:
            # 1. Broadcast global LoRA to all clients
            global_state = get_lora_state(self.global_model)
            for client in self.clients:
                client.set_lora_state(copy.deepcopy(global_state))

            # 2. Local training
            client_states = []
            client_weights = []
            train_metrics = []

            for client in self.clients:
                metrics = client.train()
                train_metrics.append(metrics)
                client_states.append(client.get_lora_state())
                client_weights.append(metrics['n_samples'])

            # 3. Aggregate
            agg_state = self.aggregate(client_states, client_weights)
            set_lora_state(self.global_model, agg_state)

            # 4. Evaluate
            eval_results = self._evaluate_all_clients(agg_state)

            # Log
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
                    'loss': f"{eval_results['avg_loss']:.3f}",
                })
                if (round_idx + 1) % 10 == 0:
                    print(f"\n  Round {round_idx+1} | {domain_str}")

        return self.history

    def _evaluate_all_clients(self, lora_state):
        """Evaluate all clients with given LoRA state."""
        per_client = {}
        per_domain = {}
        domain_counts = {}

        for client in self.clients:
            client.set_lora_state(copy.deepcopy(lora_state))
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
        avg_loss = 0.0  # Simplified - we track accuracy as primary metric

        return {
            'avg_accuracy': avg_accuracy,
            'avg_loss': avg_loss,
            'per_domain': per_domain,
            'per_client': per_client,
        }
