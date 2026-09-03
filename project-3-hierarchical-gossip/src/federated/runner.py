"""Decentralized runner with a pluggable mixing matrix and delta-W merging.

Generalises `GossipProtocol` (which is left untouched as the recorded
baseline) in three ways:

- mixing is a matrix W supplied per round by `mixing_fn(round_idx)`, so flat
  Metropolis-Hastings, affinity-weighted and two-tier schedules all run through
  the same loop;
- merging happens in delta-W space via the kernels in `merge.py`, so it is
  gauge-invariant and tolerates a different rank on every client;
- the per-round truncation residual is measured and, optionally, fed back.

One round, for client i with target rank r_i:

    y_i  = sum_j W_ij * DeltaW_j  (+ m_i, if error feedback is on)
    x_i  = SVD_{r_i}(y_i)                    # what the client can store
    m_i  = y_i - x_i                         # residual; kept only with feedback

The residual's relative energy ||y_i - x_i||^2 / ||y_i||^2 is the tail-mass
term epsilon_r in docs/research/2026-09-03-project3-convergence-analysis.md
and is logged every round.

Error feedback stores one dense (d_out x d_in) residual per LoRA layer per
client. On the fc-only testbed that is 100 x 512 floats -- trivial. On a model
with LoRA in every block it would erase most of LoRA's memory advantage, which
is why it is an option and off by default.
"""

import numpy as np
import torch

from src.federated.hierarchical import is_doubly_stochastic
from src.federated.merge import _check_alpha, factorize_delta, lora_to_delta


class DecentralizedRunner:
    """Run local training + matrix-weighted delta-W gossip for a set of clients.

    Args:
        clients: objects exposing `client_id`, `domain_id`, `train()`,
            `evaluate() -> {'accuracy': float}`, `get_lora_state()` and
            `set_lora_state(state)`. Their order fixes the row/column order
            of every mixing matrix.
        mixing_fn: `round_idx -> (N, N)` doubly stochastic array.
        target_ranks: {client_id: rank} -- the rank each client refactorises to.
        alpha: LoRA alpha shared by every client's forward pass.
        error_feedback: keep and re-inject the truncation residual.
    """

    def __init__(self, clients, mixing_fn, target_ranks, alpha, error_feedback=False):
        if not clients:
            raise ValueError("DecentralizedRunner needs at least one client")
        _check_alpha(alpha)
        self.clients = list(clients)
        self.client_ids = [c.client_id for c in self.clients]
        if len(self.client_ids) != len(set(self.client_ids)):
            raise ValueError("client ids must be unique")
        missing = [cid for cid in self.client_ids if cid not in target_ranks]
        if missing:
            raise ValueError(f"target_ranks missing entries for clients {missing}")
        for cid in self.client_ids:
            if int(target_ranks[cid]) < 1:
                raise ValueError(f"target rank for client {cid!r} must be >= 1")
        self.mixing_fn = mixing_fn
        self.target_ranks = {cid: int(target_ranks[cid]) for cid in self.client_ids}
        self.alpha = float(alpha)
        self.error_feedback = bool(error_feedback)
        self._memory = {cid: None for cid in self.client_ids}
        self.history = {
            'rounds': [],
            'avg_accuracy': [],
            'per_domain_accuracy': [],
            'per_client_accuracy': [],
            'messages_per_round': [],
            'floats_per_round': [],
            'mean_tail_mass': [],
            'max_tail_mass': [],
            'consensus_distance': [],
        }

    # ------------------------------------------------------------------
    def _mixing_matrix(self, round_idx):
        w = np.asarray(self.mixing_fn(round_idx), dtype=float)
        n = len(self.clients)
        if w.shape != (n, n):
            raise ValueError(f"mixing matrix must be ({n}, {n}), got {w.shape}")
        if not np.all(np.isfinite(w)):
            raise ValueError(f"mixing matrix for round {round_idx} contains non-finite entries")
        if not is_doubly_stochastic(w):
            raise ValueError(
                f"mixing matrix for round {round_idx} is not doubly stochastic; "
                "mean preservation and the spectral-gap argument both need it"
            )
        return w

    @staticmethod
    def _factor_floats(state):
        """Floats needed to transmit a lora state: sum over layers of r (d_in + d_out)."""
        return sum(p['A'].shape[0] * (p['A'].shape[1] + p['B'].shape[0]) for p in state.values())

    def _communication(self, w, states):
        """Messages and floats sent in one round under mixing matrix w.

        Each nonzero off-diagonal W_ij is one transmission of client j's
        factors to client i: r_j * (d_in + d_out) floats, never a dense delta.
        """
        sizes = [self._factor_floats(s) for s in states]
        messages, floats = 0, 0
        n = len(states)
        for i in range(n):
            for j in range(n):
                if i != j and w[i, j] > 0.0:
                    messages += 1
                    floats += sizes[j]
        return messages, floats

    def gossip_round(self, round_idx, states):
        """Mix the given lora states under this round's matrix.

        Returns (new_states, diagnostics). Separated from `run` so the mixing
        step can be tested without a training loop.
        """
        w = self._mixing_matrix(round_idx)
        n = len(self.clients)
        deltas = [lora_to_delta(states[i], self.alpha) for i in range(n)]
        layers = list(deltas[0])
        for k, d in enumerate(deltas[1:], start=1):
            if list(d) != layers:
                raise ValueError(f"client {self.client_ids[k]!r} has a different layer set")

        new_states, tails = [], []
        for i, cid in enumerate(self.client_ids):
            mixed = {}
            for layer in layers:
                total = None
                for j in range(n):
                    if w[i, j] == 0.0:
                        continue
                    contribution = w[i, j] * deltas[j][layer]
                    total = contribution if total is None else total + contribution
                if self.error_feedback and self._memory[cid] is not None:
                    total = total + self._memory[cid][layer]
                mixed[layer] = total

            state, residual, tail = {}, {}, []
            for layer, y in mixed.items():
                # Return factors in the client's own parameter dtype (y is in the
                # float32/float64 working dtype); reconstruct x in the working
                # dtype so the logged tail mass is truncation error, not rounding.
                out_dtype = states[i][layer]['A'].dtype
                factors = factorize_delta(y, self.target_ranks[cid], self.alpha, dtype=out_dtype)
                r = factors['A'].shape[0]
                x = (self.alpha / r) * (factors['B'].to(y.dtype) @ factors['A'].to(y.dtype))
                state[layer] = factors
                residual[layer] = y - x
                y_energy = float(torch.sum(y * y))
                tail.append(float(torch.sum(residual[layer] ** 2)) / y_energy if y_energy > 0 else 0.0)
            if self.error_feedback:
                self._memory[cid] = residual
            new_states.append(state)
            tails.append(float(np.mean(tail)) if tail else 0.0)

        messages, floats = self._communication(w, states)
        diagnostics = {
            'messages': messages,
            'floats': floats,
            'mean_tail_mass': float(np.mean(tails)),
            'max_tail_mass': float(np.max(tails)),
            'consensus_distance': self._consensus_distance(new_states),
        }
        return new_states, diagnostics

    def _consensus_distance(self, states):
        """(1/N) sum_i ||DeltaW_i - mean||_F^2 over all layers -- the Xi^t of the analysis."""
        deltas = [lora_to_delta(s, self.alpha) for s in states]
        total = 0.0
        for layer in deltas[0]:
            stack = torch.stack([d[layer] for d in deltas])
            mean = stack.mean(dim=0, keepdim=True)
            total += float(torch.sum((stack - mean) ** 2)) / len(deltas)
        return total

    # ------------------------------------------------------------------
    def run(self, n_rounds, verbose=False):
        for round_idx in range(int(n_rounds)):
            for c in self.clients:
                c.train()
            states = [c.get_lora_state() for c in self.clients]
            new_states, diag = self.gossip_round(round_idx, states)
            for c, s in zip(self.clients, new_states):
                c.set_lora_state(s)

            ev = self._evaluate()
            h = self.history
            h['rounds'].append(round_idx)
            h['avg_accuracy'].append(ev['avg_accuracy'])
            h['per_domain_accuracy'].append(ev['per_domain'])
            h['per_client_accuracy'].append(ev['per_client'])
            h['messages_per_round'].append(diag['messages'])
            h['floats_per_round'].append(diag['floats'])
            h['mean_tail_mass'].append(diag['mean_tail_mass'])
            h['max_tail_mass'].append(diag['max_tail_mass'])
            h['consensus_distance'].append(diag['consensus_distance'])
            if verbose:
                print(f"round {round_idx + 1}/{n_rounds}  acc={ev['avg_accuracy']:.4f}  "
                      f"msgs={diag['messages']}  tail={diag['mean_tail_mass']:.3e}  "
                      f"consensus={diag['consensus_distance']:.3e}")
        return self.history

    def _evaluate(self):
        per_client, per_domain, counts = {}, {}, {}
        for c in self.clients:
            acc = float(c.evaluate()['accuracy'])
            per_client[c.client_id] = acc
            per_domain[c.domain_id] = per_domain.get(c.domain_id, 0.0) + acc
            counts[c.domain_id] = counts.get(c.domain_id, 0) + 1
        for d in per_domain:
            per_domain[d] /= counts[d]
        return {
            'avg_accuracy': float(np.mean(list(per_client.values()))),
            'per_domain': per_domain,
            'per_client': per_client,
        }

    def consensus_state(self, target_rank):
        """Uniform merge of every client's current adapter at `target_rank`.

        This is the object the consensus evaluation protocol scores on the full
        test set, as opposed to the personalized protocol which scores each
        client's own adapter on its own shard.
        """
        states = [c.get_lora_state() for c in self.clients]
        deltas = [lora_to_delta(s, self.alpha) for s in states]
        merged = {}
        for layer in deltas[0]:
            mean = torch.stack([d[layer] for d in deltas]).mean(dim=0)
            out_dtype = states[0][layer]['A'].dtype
            merged[layer] = factorize_delta(mean, int(target_rank), self.alpha, dtype=out_dtype)
        return merged
