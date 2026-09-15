"""Auditable paired pooled-versus-peer LoRA experiment.

One-process simulation of neighbor message flow, not a multi-host deployment.
Test labels are only read by evaluators. P1 probes and P2 weights use training
data. Dense P2 control traffic and final peer assembly are explicitly charged.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.feature_cache import load_features, tensor_digest, write_json
from experiments.protocol_benchmark import (
    FeatureClient, initial_parameters, make_splits, package_versions,
    source_provenance, topology_order,
)
from src.federated.adaptive_rank import make_controller, POLICY_PROVENANCE
from src.federated.domain_weights import (
    DOMAIN_POLICY_PROVENANCE, conservative_domain_factors,
    features_from_updates, weighted_metropolis,
)
from src.federated.merge import lora_to_delta
from src.federated.mixing import build_topology, metropolis_hastings
from src.federated.peer_assembly import tree_allgather, tree_weighted_assembly
from src.federated.runner import DecentralizedRunner

METHODS = ('pooled', 'pooled_reset', 'fedavg16', 'mh16', 'mh_sample',
           'fixed_quality', 'fixed_domain', 'adaptive_quality', 'adaptive_domain')


class WeightedRunner(DecentralizedRunner):
    """Separate weighted-objective runner; no symmetric/DS theorem is claimed."""
    def set_weights(self, matrix, weights):
        self.matrix = np.asarray(matrix, dtype=float)
        self.stationary = np.asarray(weights, dtype=float)
        self.stationary /= self.stationary.sum()

    def _mixing_matrix(self, round_idx):
        p, pi = self.matrix, self.stationary
        n = len(self.clients)
        if p.shape != (n, n) or not np.isfinite(p).all() or (p < 0).any():
            raise ValueError('invalid weighted mixing matrix')
        if not np.allclose(p.sum(1), 1, atol=1e-12, rtol=0):
            raise ValueError('weighted mixer must be row stochastic')
        if not np.allclose(pi @ p, pi, atol=1e-12, rtol=0):
            raise ValueError('weighted mixer must preserve its stated objective')
        if not np.allclose(pi[:, None] * p, pi[None, :] * p.T, atol=1e-12, rtol=0):
            raise ValueError('weighted mixer must satisfy detailed balance')
        return p


@torch.no_grad()
def score_state(state, initial, test, indices, alpha, batch_size):
    delta = lora_to_delta(state, alpha)['fc'].to(test['features'].device)
    weight = initial['weight'].to(delta.device) + delta
    bias = initial['bias'].to(delta.device)
    correct = 0
    for start in range(0, len(indices), batch_size):
        index = indices[start:start + batch_size]
        prediction = nn.functional.linear(test['features'][index], weight, bias).argmax(1)
        correct += int((prediction == test['labels'][index]).sum())
    return {'accuracy': correct / len(indices), 'correct': correct, 'n_test': len(indices)}


def persistent_train(client, optimizer):
    """Normal pooled Adam baseline, preserving moments across complete epochs."""
    client.model.train()
    total, count = 0., 0
    for _ in range(client.config.local_epochs):
        order = torch.randperm(len(client.train_indices), generator=client.rng).to(client.device)
        for start in range(0, len(order), client.config.batch_size):
            index = client.train_indices[order[start:start + client.config.batch_size]]
            loss = nn.functional.cross_entropy(client.model(client.train_data['features'][index]),
                                               client.train_data['labels'][index])
            if not torch.isfinite(loss):
                raise ValueError('non-finite pooled loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(index)
            count += len(index)
    return {'loss': total / count, 'n_samples': count}


def communicated_weights(clients, before, after, quality_values, splits, neighbors,
                         root_id, domain, alpha):
    """Every peer derives P2 weights from its tree-delivered training metadata."""
    payloads = {}
    for c, old, new, quality_value in zip(clients, before, after, quality_values):
        packet = {'n': len(splits[c.client_id]['train_indices']),
                  'q': quality_value}
        if domain:
            old_delta, new_delta = lora_to_delta(old, alpha), lora_to_delta(new, alpha)
            packet['update'] = torch.cat([(new_delta[k] - old_delta[k]).flatten()
                                          for k in sorted(old_delta)]).numpy()
            packet['histogram'] = splits[c.client_id]['train_class_counts']
        payloads[c.client_id] = packet
    views, transport = tree_allgather(payloads, neighbors, root_id)
    ids = [c.client_id for c in clients]
    all_weights, factors, features = [], None, None
    for cid in ids:
        records = [views[cid][j] for j in ids]
        counts = np.asarray([p['n'] for p in records], dtype=float)
        quality = np.asarray([p['q'] for p in records], dtype=float)
        features = features_from_updates([p['update'] for p in records],
                                         [p['histogram'] for p in records]) if domain else None
        factors = conservative_domain_factors(counts, quality, features)
        weights = counts * quality * factors
        all_weights.append(weights / weights.sum())
    if not all(np.array_equal(all_weights[0], w) for w in all_weights):
        raise ValueError('peers disagree on domain allocation')
    return all_weights[0], {
        'quality': [payloads[cid]['q'] for cid in ids],
        'domain_factors': factors.tolist(),
        'domain_features': {k: np.asarray(v).tolist() for k, v in (features or {}).items()},
        'control_transport': transport,
    }


def run_one(config, seed, method, train, test, metadata, output):
    start = time.perf_counter()
    config.current_seed = seed
    splits, assignments = make_splits(train['labels'].cpu().numpy(),
                                     test['labels'].cpu().numpy(), config, seed)
    ids = sorted(splits)
    split_hash = hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()
    write_json(output / f'splits_seed{seed}.json', {'sha256': split_hash, 'clients': splits})
    initial = initial_parameters(train['features'].shape[1], config.reference_rank, seed)
    train_indices = sorted(i for s in splits.values() for i in s['train_indices'])
    test_indices = sorted(i for s in splits.values() for i in s['test_indices'])
    if len(set(train_indices)) != len(train_indices) or len(set(test_indices)) != len(test_indices):
        raise ValueError('client data partitions must not overlap')
    if not config.max_train_per_client and len(train_indices) != len(train['labels']):
        raise ValueError('full-data experiment must use all training examples')
    if not config.max_test_per_domain and len(test_indices) != len(test['labels']):
        raise ValueError('full-data experiment must evaluate all test examples')
    eval_indices = torch.tensor(test_indices, device=config.device)
    pooled = method.startswith('pooled')
    uniform = method in {'fedavg16', 'mh16'}
    adaptive = method.startswith('adaptive_')
    domain = method.endswith('_domain')
    quality = method.endswith(('_quality', '_domain'))
    capacities = {cid: config.reference_rank if uniform else (4, 8, 16)[cid % 3] for cid in ids}
    neighbors = build_topology(topology_order(assignments, seed), config.topology)
    root_id = next(cid for cid in ids if capacities[cid] == config.reference_rank)
    proposal = metropolis_hastings(neighbors, ids)
    if pooled:
        client = FeatureClient(0, 0, initial, config.reference_rank, config, train, test,
                               {'train_indices': train_indices, 'test_indices': test_indices},
                               torch.device(config.device))
        clients = [client]
        optimizer = torch.optim.Adam([client.model.lora_A, client.model.lora_B],
                                     lr=config.lr, weight_decay=config.weight_decay)
    else:
        clients = [FeatureClient(cid, assignments[cid], initial, capacities[cid], config,
                                 train, test, splits[cid], torch.device(config.device)) for cid in ids]
        runner = WeightedRunner(clients, lambda _: proposal, capacities, config.alpha)
    controllers = {cid: make_controller(capacities[cid]) for cid in ids} if adaptive else {}
    record = {
        'schema_version': 2, 'status': 'running', 'method': method, 'seed': seed,
        'alpha': config.alpha, 'reference_rank': config.reference_rank,
        'capacity_ceilings': {0: config.reference_rank} if pooled else capacities,
        'initial_state_sha256': tensor_digest(*initial.values()),
        'split_sha256': split_hash, 'n_train': len(train_indices), 'n_test': len(test_indices),
        'feature_cache_identity_sha256': metadata['cache_identity_sha256'],
        'optimizer': {'name': 'Adam', 'lr': config.lr, 'weight_decay': config.weight_decay,
                      'batch_size': config.batch_size, 'local_epochs': config.local_epochs,
                      'reset_each_round': method != 'pooled'},
        'topology': {} if pooled else neighbors, 'assembly_root': None if pooled or method == 'fedavg16' else root_id,
        'objective': 'sample*quality*domain' if domain else 'sample*quality' if quality else 'sample',
        'rank_policy': POLICY_PROVENANCE if adaptive else None,
        'domain_policy': DOMAIN_POLICY_PROVENANCE if domain else None,
        'execution': 'one-process simulation; tree-delivered metadata and peer assembly',
        'quality_definition': 'post-local-training cross entropy on fixed seeded local training minibatch; 1/(1+loss)',
        'rounds': [],
    }
    path = output / f'seed{seed}_{method}.json'
    write_json(path, record)
    counts = np.asarray([len(splits[cid]['train_indices']) for cid in ids], dtype=float)
    samples = counts / counts.sum()
    for round_idx in range(config.rounds):
        tick = time.perf_counter()
        row = {'round': round_idx + 1}
        if pooled:
            result = persistent_train(client, optimizer) if method == 'pooled' else client.train()
            row['train_seconds'] = time.perf_counter() - tick
            state = client.get_lora_state()
            row.update({'train_loss': result['loss'], 'train_sample_exposures': result['n_samples'],
                        'train_rank_sample_products': config.reference_rank * result['n_samples'],
                        'training_factor_floats': 0, 'training_messages': 0,
                        'control_floats': 0, 'control_bytes': 0, 'control_messages': 0,
                        'optimizer_steps': config.local_epochs * int(np.ceil(len(train_indices) / config.batch_size))})
        else:
            if adaptive:
                probe_start = time.perf_counter()
                stable_ranks = {}
                for c in clients:
                    stable = c.probe_gradient_stable_rank(capacities[c.client_id])
                    stable_ranks[c.client_id] = stable
                    c.resize_rank(controllers[c.client_id].update(stable))
                runner.set_target_ranks({c.client_id: c.rank for c in clients})
                row['gradient_probe_seconds'] = time.perf_counter() - probe_start
                row['gradient_probe_examples'] = sum(len(c.probe_indices) for c in clients)
                row['gradient_probe_rank_sample_products'] = sum(
                    len(c.probe_indices) * capacities[c.client_id] for c in clients)
                row['stable_ranks'] = stable_ranks
            before = [c.get_lora_state() for c in clients]
            train_start = time.perf_counter()
            training = [c.train() for c in clients]
            row['train_seconds'] = time.perf_counter() - train_start
            if quality or adaptive:
                quality_start = time.perf_counter()
                quality_values = [c.probe_quality() for c in clients]
                if adaptive:
                    for c, value in zip(clients, quality_values):
                        controllers[c.client_id].observe_quality(value)
                row['quality_probe_seconds'] = time.perf_counter() - quality_start
                row['quality_probe_examples'] = sum(len(c.probe_indices) for c in clients)
                row['controller_diagnostics'] = {cid: ctrl.diagnostics() for cid, ctrl in controllers.items()}
            states = [c.get_lora_state() for c in clients]
            if quality:
                control_start = time.perf_counter()
                weights, weight_record = communicated_weights(clients, before, states, quality_values,
                                                              splits, neighbors, root_id, domain, config.alpha)
                row['control_seconds'] = time.perf_counter() - control_start
                row.update(weight_record)
                transport = weight_record['control_transport']
                row.update({'control_floats': transport['numeric_values'],
                            'control_bytes': transport['bytes'], 'control_messages': transport['messages']})
            else:
                weights = samples.copy()
                row.update({'control_floats': 0, 'control_bytes': 0, 'control_messages': 0})
            matrix = np.tile(weights, (len(ids), 1)) if method == 'fedavg16' else weighted_metropolis(proposal, weights)
            runner.set_weights(matrix, weights)
            gossip_start = time.perf_counter()
            new_states, diagnostics = runner.gossip_round(round_idx, states)
            row['gossip_seconds'] = time.perf_counter() - gossip_start
            for c, new in zip(clients, new_states):
                c.set_lora_state(new)
            if method == 'fedavg16':
                # Centralized diagnostic: gather factors, return each rank-16 model.
                messages = 2 * len(ids)
                floats = sum(runner._factor_floats(s) for s in states + new_states)
            else:
                messages, floats = diagnostics['messages'], diagnostics['floats']
            assembly_start = time.perf_counter()
            if method == 'fedavg16':
                state, assembly = new_states[0], None
            else:
                state, assembly = tree_weighted_assembly(
                    {c.client_id: s for c, s in zip(clients, new_states)},
                    dict(zip(ids, weights)), neighbors, root_id, config.reference_rank, config.alpha)
            row['assembly_seconds'] = time.perf_counter() - assembly_start
            row.update({'ranks': {c.client_id: c.rank for c in clients},
                        'weights': weights.tolist(), 'mixing_matrix': matrix.tolist(),
                        'max_mixer_change_from_sample': float(np.max(np.abs(matrix - weighted_metropolis(proposal, samples)))),
                        'training_factor_floats': floats, 'training_messages': messages,
                        'merge_diagnostics': diagnostics, 'assembly': assembly,
                        'train_sample_exposures': sum(t['n_samples'] for t in training),
                        'optimizer_steps': config.local_epochs * sum(int(np.ceil(n / config.batch_size)) for n in counts),
                        'train_rank_sample_products': sum(c.rank * t['n_samples'] for c, t in zip(clients, training)),
                        'train_loss': float(np.average([t['loss'] for t in training], weights=counts)),
                        'personalized_accuracy': float(np.mean([c.evaluate()['accuracy'] for c in clients]))})
        evaluation = score_state(state, initial, test, eval_indices, config.alpha, config.eval_batch_size)
        row.update({'full_test_accuracy': evaluation['accuracy'], 'full_test_correct': evaluation['correct'],
                    'wall_seconds': time.perf_counter() - tick})
        record['rounds'].append(row)
        write_json(path, record)
        print(f"seed={seed} method={method} round={round_idx + 1}/{config.rounds} "
              f"full_test={row['full_test_accuracy']:.4f} seconds={row['wall_seconds']:.2f}", flush=True)
    record['status'] = 'complete'
    record['wall_seconds'] = time.perf_counter() - start
    record['final_full_test_accuracy'] = record['rounds'][-1]['full_test_accuracy']
    record['training_sample_exposures'] = sum(r['train_sample_exposures'] for r in record['rounds'])
    record['total_training_factor_floats'] = sum(r['training_factor_floats'] for r in record['rounds'])
    record['total_control_bytes'] = sum(r['control_bytes'] for r in record['rounds'])
    record['final_assembly'] = record['rounds'][-1].get('assembly')
    record['total_deployment_payload_bytes'] = (4 * record['total_training_factor_floats']
        + record['total_control_bytes'] + (record['final_assembly'] or {}).get('bytes', 0))
    record['evaluation_only_assembly_bytes'] = sum(
        (r.get('assembly') or {}).get('bytes', 0) for r in record['rounds'][:-1])
    record['total_optimizer_steps'] = sum(r['optimizer_steps'] for r in record['rounds'])
    record['communication_note'] = ('Training + all weight-control traffic + one final assembly are deployment costs. '
                                    'Earlier per-round assemblies produce evaluation curves and are counted separately. '
                                    'Static sample-count exchange/setup, packet framing and transport headers excluded.')
    write_json(path, record)
    torch.save({'state': state, 'initial': initial, 'alpha': config.alpha,
                'seed': seed, 'method': method}, output / f'seed{seed}_{method}_adapter.pt')
    return {k: record[k] for k in ('seed', 'method', 'final_full_test_accuracy', 'wall_seconds',
                                   'training_sample_exposures', 'total_training_factor_floats', 'total_control_bytes')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--feature-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    p.add_argument('--rounds', type=int, default=30)
    p.add_argument('--alpha', type=float, default=32.)
    p.add_argument('--reference-rank', type=int, choices=[16], default=16)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--weight-decay', type=float, default=.0001)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--eval-batch-size', type=int, default=2048)
    p.add_argument('--local-epochs', type=int, default=1)
    p.add_argument('--topology', choices=['ring', 'fully_connected', 'star', 'path'], default='ring')
    p.add_argument('--clients-per-domain', type=int, default=3)
    p.add_argument('--max-train-per-client', type=int, default=0)
    p.add_argument('--max-test-per-domain', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    args.n_domains, args.dirichlet_alpha = 5, .5
    if args.rounds < 1 or args.clients_per_domain != 3:
        p.error('positive rounds and exactly three capability tiers per domain are required')
    if args.output.exists() and any(args.output.iterdir()):
        p.error('use a fresh output directory to preserve every attempted run')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    metadata_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest = {'schema_version': 2, 'status': 'running', 'config': metadata_config,
                'source': source_provenance(), 'p1_policy': POLICY_PROVENANCE,
                'p2_policy': DOMAIN_POLICY_PROVENANCE, 'packages': package_versions(),
                'primary_endpoint': 'final assembled rank16 full-test accuracy minus pooled persistent-Adam accuracy',
                'noninferiority_margin': 'none prespecified; no formal equivalence claim',
                'results': []}
    write_json(args.output / 'manifest.json', manifest)
    train_cpu, test_cpu, metadata = load_features(args.data_dir, args.feature_cache, torch.device(args.device))
    manifest['feature_cache'] = metadata
    train = {k: v.to(args.device) for k, v in train_cpu.items()}
    test = {k: v.to(args.device) for k, v in test_cpu.items()}
    for seed in args.seeds:
        for method in args.methods:
            try:
                result = run_one(args, seed, method, train, test, metadata, args.output)
            except Exception as error:
                manifest.update({'status': 'failed', 'failed_seed': seed, 'failed_method': method,
                                 'error': f'{type(error).__name__}: {error}'})
                write_json(args.output / 'manifest.json', manifest)
                record_path = args.output / f'seed{seed}_{method}.json'
                if record_path.exists():
                    failed_record = json.loads(record_path.read_text())
                    failed_record.update({'status': 'failed', 'error': manifest['error']})
                    write_json(record_path, failed_record)
                raise
            manifest['results'].append(result)
            write_json(args.output / 'manifest.json', manifest)
    manifest['status'] = 'complete'
    write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
