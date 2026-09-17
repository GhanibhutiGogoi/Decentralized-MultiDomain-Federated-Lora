"""Validation-only shared-model, heterogeneous partial-gradient exploration."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from experiments.explore_residual_protocol import load_training_holdout
from experiments.diagnose_gradient_equivalence import tree_gradient_allreduce, install_gradient, spanning_tree
from experiments.feature_cache import write_json, tensor_digest
from experiments.protocol_benchmark import FeatureClient, initial_parameters, make_splits, topology_order, source_provenance, package_versions
from experiments.integrated_benchmark import score_state
from src.federated.adaptive_rank import make_controller, POLICY_PROVENANCE
from src.federated.domain_weights import conservative_domain_factors, features_from_updates, DOMAIN_POLICY_PROVENANCE
from src.federated.mixing import build_topology
from src.federated.peer_assembly import tree_allgather

ARMS = {'uniform_sample': (False, 'sample'), 'fixed_sample': (False, 'sample'), 'adaptive_sample': (True, 'sample'),
        'adaptive_quality': (True, 'quality'), 'fixed_domain': (False, 'domain'),
        'adaptive_domain': (True, 'domain')}


@torch.no_grad()
def partial_gradients(x, labels, a, b, base_weight, bias, selected, alpha=32.):
    """Only r selected factor gradients; full common model forward is retained."""
    scale = alpha / a.shape[0]
    logits = nn.functional.linear(x, base_weight, bias) + (x @ a.T @ b.T) * scale
    loss = nn.functional.cross_entropy(logits, labels)
    error = logits.softmax(1)
    error[torch.arange(len(labels), device=labels.device), labels] -= 1
    error /= len(labels)
    da = scale * (error @ b[:, selected]).T @ x
    db = scale * error.T @ (x @ a[selected].T)
    return da, db, float(loss)


def padded_gradient(a, b, selected, da, db, importance):
    grad_a, grad_b = torch.zeros_like(a), torch.zeros_like(b)
    grad_a[selected] = da * importance
    grad_b[:, selected] = db * importance
    return torch.cat([grad_a.reshape(-1), grad_b.reshape(-1)])


def stable_rank(gradients):
    values = []
    for gradient in gradients:
        squared = float(torch.sum(gradient.float() ** 2))
        spectral = float(torch.linalg.matrix_norm(gradient.float(), ord=2) ** 2)
        if spectral > 1e-12:
            values.append(squared / spectral)
    return float(np.median(values)) if values else 1.


def run_seed(args, seed, arm, train, val, metadata, output):
    adaptive, policy = ARMS[arm]
    args.current_seed = seed
    splits, assignments = make_splits(train['labels'].cpu().numpy(), val['labels'].cpu().numpy(), args, seed)
    ids = sorted(splits)
    ceilings = {cid: 16 if arm == 'uniform_sample' else (4, 8, 16)[cid % 3] for cid in ids}
    controllers = {cid: make_controller(ceilings[cid]) for cid in ids} if adaptive else {}
    ranks = dict(ceilings)
    initial = initial_parameters(train['features'].shape[1], 16, seed)
    whole = {'train_indices': list(range(len(train['labels']))), 'test_indices': list(range(len(val['labels'])))}
    client = FeatureClient(0, 0, initial, 16, args, train, val, whole, torch.device(args.device))
    model = client.model
    optimizer = torch.optim.Adam([model.lora_A, model.lora_B], lr=args.lr, weight_decay=args.weight_decay)
    neighbors = build_topology(topology_order(assignments, seed), 'ring')
    owners = torch.full((len(train['labels']),), -1, device=args.device, dtype=torch.long)
    probes, probe_coordinates = {}, {}
    for cid in ids:
        owned = torch.tensor(splits[cid]['train_indices'], device=args.device)
        assert torch.all(owners[owned] == -1)
        owners[owned] = cid
        generator = torch.Generator().manual_seed(seed + 15485863 * (cid + 1))
        probes[cid] = owned[torch.randperm(len(owned), generator=generator)[:args.batch_size].to(args.device)]
        probe_coordinates[cid] = torch.randperm(16, generator=torch.Generator().manual_seed(seed + 982451653 * (cid + 1)))[:ceilings[cid]].to(args.device)
    assert torch.all(owners >= 0)
    mask_rng = torch.Generator().manual_seed(seed + 32452843)
    counts_total = np.asarray([len(splits[cid]['train_indices']) for cid in ids], dtype=float)
    factor_values = model.lora_A.numel() + model.lora_B.numel()
    bytes_per_step = 2 * (len(ids) - 1) * (factor_values * 4 + 8)
    path = output / f'seed{seed}_{arm}.json'
    record = {'status': 'running', 'arm': arm, 'seed': seed, 'exploratory': True,
              'evaluation_split': metadata['evaluation_split'], 'holdout_sha256': metadata['holdout_sha256'],
              'protocol': 'common rank16 model/Adam; random partial factor gradients; importance correction; neighbor allreduce',
              'rank_policy': POLICY_PROVENANCE if adaptive else None, 'domain_policy': DOMAIN_POLICY_PROVENANCE if policy == 'domain' else None,
              'probe_integration': 'fixed ceiling-sized shared-coordinate subset; training-only gradient/quality probes; P2 signal is first-order effective probe update',
              'capacity_scope': 'r limits computed gradient coordinates; full rank16 frozen factors, padded transport, and shared optimizer state still required',
              'execution': 'single-process peer-flow simulation; identical optimizer/model replicas collapsed',
              'communication_exclusions': 'initial dissemination, global scheduling/setup, framing; dense domain-control exchange included',
              'topology': neighbors, 'ceilings': ceilings, 'n_train': len(train['labels']), 'n_validation': len(val['labels']),
              'initial_state_sha256': tensor_digest(*initial.values()),
              'ownership_sha256': tensor_digest(owners),
              'split_sha256': hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest(), 'rounds': []}
    write_json(path, record)
    try:
        for epoch in range(args.epochs):
            start = time.perf_counter()
            root = ids[epoch % len(ids)]
            a, b = model.lora_A.detach(), model.lora_B.detach()
            packet, signals, qualities = {}, {}, []
            domain_features, domain_factors = None, np.ones(len(ids))
            for cid in ids:
                index, selected = probes[cid], probe_coordinates[cid]
                da, db, loss = partial_gradients(train['features'][index], train['labels'][index], a, b, model.linear.weight, model.linear.bias, selected)
                signal = stable_rank((da, db))
                signals[cid] = signal
                quality = 1 / (1 + loss)
                qualities.append(quality)
                if adaptive:
                    ranks[cid] = controllers[cid].update(signal)
                packet[cid] = {'n': counts_total[cid], 'q': quality}
                if policy == 'domain':
                    # Negative gradient tangent in scaled effective-weight space.
                    tangent = -(args.alpha / 16) * (b[:, selected] @ da + db @ a[selected]) * (16 / ceilings[cid])
                    packet[cid]['update'] = tangent.detach().cpu().flatten().numpy()
                    packet[cid]['histogram'] = splits[cid]['train_class_counts']
            if policy == 'sample':
                multipliers = np.ones(len(ids))
                control = {'bytes': 0, 'messages': 0}
            else:
                views, control = tree_allgather(packet, neighbors, root)
                reference = None
                for receiver in ids:
                    received = [views[receiver][cid] for cid in ids]
                    qualities_remote = np.asarray([p['q'] for p in received])
                    counts_remote = np.asarray([p['n'] for p in received])
                    features = features_from_updates([p['update'] for p in received], [p['histogram'] for p in received]) if policy == 'domain' else None
                    factors = conservative_domain_factors(counts_remote, qualities_remote, features)
                    domain_features, domain_factors = features, factors
                    value = qualities_remote * factors
                    value /= np.average(value, weights=counts_remote)
                    if reference is not None:
                        assert np.array_equal(value, reference)
                    reference = value
                multipliers = reference
            permutation = torch.randperm(len(train['labels']), generator=client.rng).to(args.device)
            loss_sum, steps, rank_samples = 0., 0, 0
            for begin in range(0, len(permutation), args.batch_size):
                batch = permutation[begin:begin + args.batch_size]
                batch_owners = owners[batch]
                a, b = model.lora_A.detach(), model.lora_B.detach()
                local, counts = {}, {}
                for cid in ids:
                    index = batch[batch_owners == cid]
                    count = len(index)
                    counts[cid] = count
                    # Draw even for an empty peer to preserve paired RNG streams.
                    selected = torch.randperm(16, generator=mask_rng)[:ranks[cid]].to(args.device)
                    if count:
                        da, db, loss = partial_gradients(train['features'][index], train['labels'][index], a, b, model.linear.weight, model.linear.bias, selected)
                        local[cid] = padded_gradient(a, b, selected, da, db, (16 / ranks[cid]) * multipliers[cid])
                        loss_sum += loss * count
                        rank_samples += count * ranks[cid]
                    else:
                        local[cid] = a.new_zeros(factor_values)
                gradient, count = tree_gradient_allreduce(local, counts, neighbors, root)
                assert count == len(batch)
                assert torch.isfinite(gradient).all()
                optimizer.zero_grad(set_to_none=True)
                install_gradient(model, gradient)
                optimizer.step()
                steps += 1
            if adaptive:
                # Quality guard observes the updated COMMON model on local train probes.
                for cid in ids:
                    index = probes[cid]
                    with torch.no_grad():
                        loss = float(nn.functional.cross_entropy(model(train['features'][index]), train['labels'][index]))
                    controllers[cid].observe_quality(1 / (1 + loss))
            state = client.get_lora_state()
            scored = score_state(state, initial, val, torch.arange(len(val['labels']), device=args.device), args.alpha, args.eval_batch_size)
            row = {'epoch': epoch + 1, 'validation_accuracy': scored['accuracy'], 'validation_correct': scored['correct'],
                   'train_loss': loss_sum / len(train['labels']), 'ranks': dict(ranks), 'stable_ranks': signals,
                   'controller_diagnostics': {cid: c.diagnostics() for cid, c in controllers.items()},
                   'objective_multipliers': multipliers.tolist(), 'control_transport': control,
                   'pre_epoch_probe_qualities': qualities,
                   'domain_factors': domain_factors.tolist(),
                   'domain_features': {k: np.asarray(v).tolist() for k, v in (domain_features or {}).items()},
                   'training_sample_exposures': len(train['labels']), 'training_gradient_rank_sample_products': rank_samples,
                   'optimizer_steps': steps, 'peer_gradient_payload_bytes': steps * bytes_per_step,
                   'peer_gradient_messages': steps * 2 * (len(ids) - 1),
                   'ceiling_probe_examples': sum(len(x) for x in probes.values()),
                   'ceiling_probe_rank_sample_products': sum(len(probes[cid]) * ceilings[cid] for cid in ids),
                   'postepoch_quality_probe_examples': sum(len(x) for x in probes.values()) if adaptive else 0,
                   'wall_seconds': time.perf_counter() - start}
            record['rounds'].append(row)
            write_json(path, record)
            print(f'seed={seed} arm={arm} epoch={epoch+1}/{args.epochs} validation={scored["accuracy"]:.4f}', flush=True)
        record['status'] = 'complete'
        record['final_validation_accuracy'] = record['rounds'][-1]['validation_accuracy']
        torch.save({'state': state, 'initial': initial, 'alpha': args.alpha, 'seed': seed, 'arm': arm}, output / f'seed{seed}_{arm}_adapter.pt')
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(path, record)
    return {'seed': seed, 'arm': arm, 'final_validation_accuracy': record['final_validation_accuracy']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feature-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('use a fresh output directory')
    if args.epochs < 1 or len(set(args.arms)) != len(args.arms) or len(set(args.seeds)) != len(args.seeds):
        parser.error('positive epochs and unique arms/seeds required')
    for key, value in dict(alpha=32., reference_rank=16, lr=.001, weight_decay=.0001,
                            batch_size=128, eval_batch_size=2048, local_epochs=1,
                            topology='ring', clients_per_domain=3, max_train_per_client=0,
                            max_test_per_domain=0, n_domains=5, dirichlet_alpha=.5).items():
        setattr(args, key, value)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    manifest = {'status': 'running', 'exploratory': True, 'source': source_provenance(), 'packages': package_versions(),
                'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'results': []}
    write_json(args.output / 'manifest.json', manifest)
    try:
        train, val, metadata, split = load_training_holdout(args.feature_cache, args.device)
        write_json(args.output / 'holdout.json', split)
        manifest['feature_cache'] = metadata
        for seed in args.seeds:
            for arm in args.arms:
                manifest['active'] = {'seed': seed, 'arm': arm}
                write_json(args.output / 'manifest.json', manifest)
                directory = args.output / arm
                directory.mkdir(parents=True, exist_ok=True)
                result = run_seed(args, seed, arm, train, val, metadata, directory)
                manifest['results'].append(result)
                write_json(args.output / 'manifest.json', manifest)
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_json(args.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
