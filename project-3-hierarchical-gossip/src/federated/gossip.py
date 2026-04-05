"""
Basic gossip protocol for decentralized federated learning.

This implements uniform gossip averaging as the decentralized baseline.
Each client communicates only with neighbors in the topology graph,
averaging LoRA parameters with uniform weights.

Later (Week 3-4), this will be extended to hierarchical domain-aware gossip.
"""

import copy
import random
import torch
from tqdm import tqdm

from src.models.lora_resnet import get_lora_state, set_lora_state


class GossipProtocol:
    """
    Decentralized gossip-based federated learning.

    Each round:
    1. All clients train locally
    2. Each client selects a random neighbor
    3. Pairs exchange and average their LoRA parameters
    """

    def __init__(self, clients, topology='ring', seed=42):
        self.clients = clients
        self.client_ids = [c.client_id for c in clients]
        self.client_map = {c.client_id: c for c in clients}
        self.seed = seed
        self.rng = random.Random(seed)

        # Build topology
        self.neighbors = self._build_topology(topology)

        self.history = {
            'rounds': [],
            'avg_accuracy': [],
            'per_domain_accuracy': [],
            'per_client_accuracy': [],
            'messages_per_round': [],
        }

    def _build_topology(self, topology):
        """
        Build communication graph.

        Args:
            topology: 'ring', 'fully_connected', or 'random_regular'

        Returns:
            dict mapping client_id -> list of neighbor client_ids
        """
        n = len(self.client_ids)
        ids = self.client_ids
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
            # Each node has exactly 4 neighbors (or n-1 if n < 5)
            k = min(4, n - 1)
            # Simple approach: random pairing
            rng = random.Random(self.seed)
            for cid in ids:
                possible = [other for other in ids if other != cid]
                selected = rng.sample(possible, k)
                neighbors[cid] = selected
            # Make symmetric
            for cid in ids:
                for neighbor in neighbors[cid]:
                    if cid not in neighbors[neighbor]:
                        neighbors[neighbor].append(cid)

        return neighbors

    def _pairwise_average(self, state1, state2, weight1=0.5, weight2=0.5):
        """Average two LoRA states with given weights."""
        averaged = {}
        for layer_name in state1:
            averaged[layer_name] = {
                'A': weight1 * state1[layer_name]['A'] + weight2 * state2[layer_name]['A'],
                'B': weight1 * state1[layer_name]['B'] + weight2 * state2[layer_name]['B'],
            }
        return averaged

    def gossip_round(self):
        """
        Execute one gossip round.

        Each client picks a random neighbor and they average their models.
        Uses synchronous push-pull gossip.

        Returns:
            n_messages: number of messages exchanged
        """
        # Collect current states
        current_states = {}
        for client in self.clients:
            current_states[client.client_id] = client.get_lora_state()

        # Each client picks a random neighbor and averages
        new_states = {}
        n_messages = 0

        for client in self.clients:
            cid = client.client_id
            neighbor_id = self.rng.choice(self.neighbors[cid])

            # Average with selected neighbor
            my_state = current_states[cid]
            neighbor_state = current_states[neighbor_id]
            averaged = self._pairwise_average(my_state, neighbor_state)

            new_states[cid] = averaged
            n_messages += 1  # Each exchange = 1 message

        # Apply new states
        for client in self.clients:
            client.set_lora_state(new_states[client.client_id])

        return n_messages

    def run(self, n_rounds=50, verbose=True):
        """
        Run gossip-based federated learning.

        Each round: local train -> gossip exchange.
        """
        iterator = tqdm(range(n_rounds), desc="Gossip") if verbose else range(n_rounds)

        for round_idx in iterator:
            # 1. Local training on all clients
            for client in self.clients:
                client.train()

            # 2. Gossip exchange
            n_messages = self.gossip_round()

            # 3. Evaluate
            eval_results = self._evaluate_all_clients()

            # Log
            self.history['rounds'].append(round_idx)
            self.history['avg_accuracy'].append(eval_results['avg_accuracy'])
            self.history['per_domain_accuracy'].append(eval_results['per_domain'])
            self.history['per_client_accuracy'].append(eval_results['per_client'])
            self.history['messages_per_round'].append(n_messages)

            if verbose:
                iterator.set_postfix({
                    'acc': f"{eval_results['avg_accuracy']:.3f}",
                    'msgs': n_messages,
                })
                if (round_idx + 1) % 10 == 0:
                    domain_str = " | ".join(
                        f"D{d}:{acc:.3f}"
                        for d, acc in sorted(eval_results['per_domain'].items())
                    )
                    print(f"\n  Round {round_idx+1} | {domain_str}")

        return self.history

    def _evaluate_all_clients(self):
        """Evaluate all clients with their current local models."""
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
            'per_domain': per_domain,
            'per_client': per_client,
        }

    def get_topology_info(self):
        """Return topology summary for logging."""
        n_edges = sum(len(v) for v in self.neighbors.values()) // 2
        degrees = [len(v) for v in self.neighbors.values()]
        return {
            'n_clients': len(self.clients),
            'n_edges': n_edges,
            'avg_degree': sum(degrees) / len(degrees),
            'min_degree': min(degrees),
            'max_degree': max(degrees),
        }
