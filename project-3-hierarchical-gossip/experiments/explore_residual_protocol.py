"""Train/validation-only exploration of retained-base low-rank residual updates.

This is an explicitly changed protocol, not an alternative implementation of
the original MH model-state averaging. It retains a dense frozen global head
at each peer and uses exact tree reduction/broadcast at each local epoch.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from experiments import integrated_benchmark as benchmark
from experiments.feature_cache import tensor_digest, write_json, cache_identity
from experiments.protocol_benchmark import FeatureClient, initial_parameters, make_splits, source_provenance, package_versions, topology_order
from src.federated.adaptive_rank import make_controller, POLICY_PROVENANCE
from src.federated.domain_weights import DOMAIN_POLICY_PROVENANCE
from src.federated.merge import factorize_delta, lora_to_delta
from src.federated.mixing import build_topology
from src.federated.peer_assembly import tree_weighted_assembly


# Values are (rank policy, weighting policy, aggregation gain).
RESIDUAL_ARMS = {
    'residual_uniform_sample_g1': ('uniform', 'sample', 1.),
    'residual_fixed_sample_g1': ('fixed', 'sample', 1.),
    'residual_adaptive_domain_g1': ('adaptive', 'domain', 1.),
    'residual_adaptive_domain_g5': ('adaptive', 'domain', 5.),
    'residual_adaptive_domain_g15': ('adaptive', 'domain', 15.),
    'residual_adaptive_sample_g5': ('adaptive', 'sample', 5.),
    'residual_adaptive_quality_g5': ('adaptive', 'quality', 5.),
    'residual_fixed_domain_g5': ('fixed', 'domain', 5.),
}
BASE_ARMS = ('pooled', 'fixed_domain', 'adaptive_domain')


def stratified_holdout(labels, per_class=50, seed=20260915):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    training, validation = [], []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) <= per_class:
            raise ValueError('every class must retain training examples')
        validation.extend(indices[:per_class].tolist())
        training.extend(indices[per_class:].tolist())
    return sorted(training), sorted(validation)


def load_training_holdout(cache_dir, device):
    """Do not open the official test tensor during candidate exploration."""
    cache = cache_dir / 'cifar100-resnet18-imagenet1k-v1-resize224-v1'
    metadata = json.loads((cache / 'manifest.json').read_text())
    if cache_identity(metadata) != metadata['cache_identity_sha256']:
        raise ValueError('cache metadata identity mismatch')
    all_training = torch.load(cache / 'train.pt', map_location='cpu', weights_only=True)
    if tensor_digest(all_training['features'], all_training['labels']) != metadata['train']['sha256']:
        raise ValueError('training tensor identity mismatch')
    fit, validation = stratified_holdout(all_training['labels'].numpy())
    split = {'seed': 20260915, 'training_original_indices': fit,
             'validation_original_indices': validation, 'official_test_opened': False}
    split['sha256'] = hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest()
    train = {k: v[fit].to(device) for k, v in all_training.items()}
    val = {k: v[validation].to(device) for k, v in all_training.items()}
    metadata = {**metadata, 'evaluation_split': '5000 held-out ORIGINAL TRAINING examples; official test not opened',
                'holdout_sha256': split['sha256']}
    return train, val, metadata, split


def advance_global(global_delta, residual_delta, gain, alpha=32., rank=16):
    combined = global_delta + float(gain) * residual_delta
    combined = combined - combined.mean(0, keepdim=True)
    state = {'fc': factorize_delta(combined, rank, alpha)}
    represented = lora_to_delta(state, alpha)['fc']
    error = float((combined - represented).square().sum())
    return state, represented, error / max(float(combined.square().sum()), 1e-30)


def tree_broadcast_head(head, neighbors, root):
    """Explicit peer-edge broadcasts; returns per-peer frozen-head replicas."""
    views = {root: head.clone()}
    order, edges = [root], []
    for sender in order:
        for receiver in neighbors[sender]:
            if receiver in views:
                continue
            views[receiver] = views[sender].clone()
            order.append(receiver)
            edges.append({'sender': sender, 'receiver': receiver,
                          'bytes': head.numel() * head.element_size()})
    if len(views) != len(neighbors):
        raise ValueError('broadcast requires connected graph')
    return views, {'messages': len(edges), 'bytes': sum(e['bytes'] for e in edges), 'edges': edges}


def run_residual(config, seed, arm, train, val, metadata, output):
    started = time.perf_counter()
    rank_policy, weighting, gain = RESIDUAL_ARMS[arm]
    config.current_seed = seed
    splits, assignments = make_splits(train['labels'].cpu().numpy(), val['labels'].cpu().numpy(), config, seed)
    ids = sorted(splits)
    ceilings = {cid: 16 if rank_policy == 'uniform' else (4, 8, 16)[cid % 3] for cid in ids}
    initial = initial_parameters(train['features'].shape[1], 16, seed)
    clients = [FeatureClient(cid, assignments[cid], initial, ceilings[cid], config, train, val,
                             splits[cid], torch.device(config.device)) for cid in ids]
    controllers = {cid: make_controller(ceilings[cid]) for cid in ids} if rank_policy == 'adaptive' else {}
    neighbors = build_topology(topology_order(assignments, seed), 'ring')
    counts = np.asarray([len(splits[cid]['train_indices']) for cid in ids], dtype=float)
    global_delta = torch.zeros_like(initial['weight'])
    heads = {cid: initial['weight'].clone() for cid in ids}
    indices = torch.arange(len(val['labels']), device=config.device)
    path = output / f'seed{seed}_{arm}.json'
    record = {'status': 'running', 'arm': arm, 'seed': seed, 'exploratory': True,
              'evaluation_split': metadata['evaluation_split'], 'holdout_sha256': metadata['holdout_sha256'],
              'rank_policy': rank_policy, 'weighting': weighting, 'gain': gain,
              'rank_controller': POLICY_PROVENANCE if controllers else None,
              'domain_policy': DOMAIN_POLICY_PROVENANCE if weighting == 'domain' else None,
              'protocol': 'retained frozen global head + local residuals + rotating peer-tree collectives',
              'retained_global_head_bytes_per_peer': global_delta.numel() * global_delta.element_size(),
              'memory_note': 'retained delta is folded into already allocated frozen linear weight; an unfused implementation needs this extra buffer; transport/SVD workspace additional',
              'capacity_scope': 'ceilings constrain TRAINABLE residual ranks only; every peer can hold the full frozen head and perform aggregation SVDs when root',
              'aggregation_scope': 'root performs untruncated rank100 residual factorization and rank16 global projection; residual reduction is exact up to floating-point roundoff',
              'transient_functional_rank_bound': 'global rank16 plus local residual rank r, at most32; final deployment rank16',
              'topology': neighbors, 'capacity_ceilings': ceilings,
              'initial_state_sha256': tensor_digest(*initial.values()),
              'split_sha256': hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest(),
              'n_train': len(train['labels']), 'n_validation': len(val['labels']), 'rounds': []}
    write_json(output / f'splits_seed{seed}.json', splits)
    write_json(path, record)
    try:
        for epoch in range(config.rounds):
            tick = time.perf_counter()
            root = ids[epoch % len(ids)]
            # Same fresh nested A directions in paired residual arms; B=0.
            directions = initial_parameters(train['features'].shape[1], 16, seed + 1000003 * epoch)['A']
            for client in clients:
                with torch.no_grad():
                    client.model.linear.weight.copy_(heads[client.client_id].to(client.device))
                    client.model.lora_A.copy_(directions[:client.rank].to(client.device))
                    client.model.lora_B.zero_()
            stable_ranks = {}
            if controllers:
                for client in clients:
                    cid = client.client_id
                    stable_ranks[cid] = client.probe_gradient_stable_rank(ceilings[cid])
                    client.resize_rank(controllers[cid].update(stable_ranks[cid]))
                    with torch.no_grad():
                        client.model.lora_A.copy_(directions[:client.rank].to(client.device))
                        client.model.lora_B.zero_()
            before = [client.get_lora_state() for client in clients]
            results = [client.train() for client in clients]
            states = [client.get_lora_state() for client in clients]
            quality_values = [client.probe_quality() for client in clients] if weighting != 'sample' or controllers else []
            if controllers:
                for client, quality in zip(clients, quality_values):
                    controllers[client.client_id].observe_quality(quality)
            if weighting != 'sample':
                weights, weight_record = benchmark.communicated_weights(clients, before, states, quality_values,
                                                                         splits, neighbors, root, weighting == 'domain', config.alpha)
            else:
                weights = counts / counts.sum()
                weight_record = {'control_transport': {'bytes': 0, 'messages': 0}}
            # rank100 retains the exact head residual; projection only after adding
            # it to the retained global state, never at a low-capacity relay.
            residual_state, transport = tree_weighted_assembly(
                dict(zip(ids, states)), dict(zip(ids, weights)), neighbors, root, 100, config.alpha)
            residual = lora_to_delta(residual_state, config.alpha)['fc']
            state, global_delta, tail = advance_global(global_delta, residual, gain, config.alpha)
            heads, broadcast = tree_broadcast_head(initial['weight'] + global_delta, neighbors, root)
            evaluation = benchmark.score_state(state, initial, val, indices, config.alpha, config.eval_batch_size)
            row = {'round': epoch + 1, 'validation_accuracy': evaluation['accuracy'],
                   'validation_correct': evaluation['correct'], 'ranks': {c.client_id: c.rank for c in clients},
                   'stable_ranks': stable_ranks, 'weights': weights.tolist(), 'weight_record': weight_record,
                   'controller_diagnostics': {cid: controller.diagnostics() for cid, controller in controllers.items()},
                   'training_sample_exposures': sum(x['n_samples'] for x in results),
                   'training_rank_sample_products': sum(c.rank * x['n_samples'] for c, x in zip(clients, results)),
                   'train_loss': float(np.average([x['loss'] for x in results], weights=counts)),
                   'optimizer_steps': sum(int(np.ceil(n / config.batch_size)) for n in counts),
                   'rank_probe_examples': sum(len(c.probe_indices) for c in clients) if controllers else 0,
                   'rank_probe_rank_sample_products': sum(len(c.probe_indices) * ceilings[c.client_id] for c in clients) if controllers else 0,
                   'quality_probe_examples': sum(len(c.probe_indices) for c in clients) if quality_values else 0,
                   'residual_reduction': transport, 'broadcast': broadcast,
                   'global_projection_relative_tail': tail, 'wall_seconds': time.perf_counter() - tick}
            record['rounds'].append(row)
            write_json(path, record)
            print(f'seed={seed} arm={arm} epoch={epoch+1}/{config.rounds} validation={evaluation["accuracy"]:.4f}', flush=True)
        record['status'] = 'complete'
        record['final_validation_accuracy'] = record['rounds'][-1]['validation_accuracy']
        record['training_sample_exposures'] = sum(r['training_sample_exposures'] for r in record['rounds'])
        record['total_transport_bytes'] = sum(r['residual_reduction']['bytes'] + r['broadcast']['bytes'] + r['weight_record']['control_transport']['bytes'] for r in record['rounds'])
        torch.save({'state': state, 'initial': initial, 'alpha': config.alpha, 'seed': seed, 'arm': arm,
                    'evaluation_split': metadata['evaluation_split']}, output / f'seed{seed}_{arm}_adapter.pt')
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record['wall_seconds'] = time.perf_counter() - started
        write_json(path, record)
    return {k: record[k] for k in ('arm', 'seed', 'final_validation_accuracy', 'training_sample_exposures', 'total_transport_bytes')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', choices=BASE_ARMS + tuple(RESIDUAL_ARMS), default=list(BASE_ARMS) + list(RESIDUAL_ARMS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--rounds', type=int, default=30)
    parser.add_argument('--device', default='cuda:0')
    config = parser.parse_args()
    if config.output.exists() and any(config.output.iterdir()):
        parser.error('use a fresh output directory')
    if config.rounds < 1 or len(set(config.arms)) != len(config.arms) or len(set(config.seeds)) != len(config.seeds):
        parser.error('positive rounds and unique arms/seeds required')
    for name, value in dict(alpha=32., reference_rank=16, lr=.001, weight_decay=.0001,
                            batch_size=128, eval_batch_size=2048, local_epochs=1,
                            topology='ring', clients_per_domain=3, max_train_per_client=0,
                            max_test_per_domain=0, n_domains=5, dirichlet_alpha=.5).items():
        setattr(config, name, value)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    manifest = {'status': 'running', 'exploratory': True, 'source': source_provenance(),
                'packages': package_versions(), 'official_test_opened': False,
                'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(config).items()}, 'results': []}
    write_json(config.output / 'manifest.json', manifest)
    try:
        train, val, metadata, holdout = load_training_holdout(config.feature_cache, config.device)
        write_json(config.output / 'holdout.json', holdout)
        manifest['feature_cache'] = metadata
        for seed in config.seeds:
            for arm in config.arms:
                manifest['active'] = {'seed': seed, 'arm': arm}
                write_json(config.output / 'manifest.json', manifest)
                directory = config.output / arm
                directory.mkdir(parents=True, exist_ok=True)
                if arm in BASE_ARMS:
                    result = benchmark.run_one(copy.copy(config), seed, arm, train, val, metadata, directory)
                    result = {'arm': arm, 'seed': seed,
                              'final_validation_accuracy': result['final_full_test_accuracy'],
                              'training_sample_exposures': result['training_sample_exposures']}
                    record_path = directory / f'seed{seed}_{arm}.json'
                    record = json.loads(record_path.read_text())
                    record['evaluation_split'] = metadata['evaluation_split']
                    record['field_note'] = 'legacy full_test_* fields in this reused runner refer only to the 5000-example training holdout'
                    record['holdout_sha256'] = holdout['sha256']
                    write_json(record_path, record)
                else:
                    result = run_residual(copy.copy(config), seed, arm, train, val, metadata, directory)
                manifest['results'].append(result)
                write_json(config.output / 'manifest.json', manifest)
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(config.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
