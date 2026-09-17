"""Paired exploratory controls around the unchanged integrated benchmark.

Only this entry point installs temporary experiment-specific overrides. The
original benchmark and its completed artifacts remain unchanged. All controls
use the original optimizer, initialization, evaluation, and sample exposures.
"""
import argparse
import copy
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from experiments import integrated_benchmark as benchmark
from experiments.feature_cache import load_features, write_json
from experiments.protocol_benchmark import FeatureClient, make_splits, source_provenance, package_versions
from src.federated.merge import factorize_delta, lora_to_delta


ARMS = {
    'pooled_reset': ('pooled_reset', False, False, False),
    'pooled_reset_svd': ('pooled_reset', True, False, False),
    'pooled_reset_svd_center': ('pooled_reset', True, True, False),
    'fedavg16': ('fedavg16', False, False, False),
    'fedavg16_center': ('fedavg16', False, True, False),
    'fedavg16_iid': ('fedavg16', False, False, True),
    'fedavg16_iid_center': ('fedavg16', False, True, True),
}


def centered_state(state):
    """Remove an identical logit shift from all classes without raising rank."""
    return {name: {'A': p['A'].clone(), 'B': p['B'] - p['B'].mean(dim=0, keepdim=True)}
            for name, p in state.items()}


def matrix_diagnostics(delta):
    delta = delta.double()
    common = delta.mean(0, keepdim=True).expand_as(delta)
    centered = delta - common
    energy = float(delta.square().sum())
    s = torch.linalg.svdvals(centered)
    return {'frobenius_energy': energy,
            'common_logit_energy_fraction': float(common.square().sum()) / max(energy, 1e-300),
            'centered_energy': float(centered.square().sum()),
            'centered_singular_values': s.tolist(),
            'centered_tail16_fraction': float(s[16:].square().sum()) / max(float(s.square().sum()), 1e-300)}


def iid_splits(train_labels, test_labels, config, seed):
    """Same data union and n_i, different allocation; official tests unchanged."""
    splits, assignments = make_splits(train_labels, test_labels, config, seed)
    indices = sorted(i for s in splits.values() for i in s['train_indices'])
    shuffled = np.random.default_rng(seed + 982451653).permutation(indices)
    start = 0
    for cid in sorted(splits):
        size = len(splits[cid]['train_indices'])
        part = shuffled[start:start + size]
        start += size
        splits[cid]['train_indices'] = part.tolist()
        splits[cid]['train_class_counts'] = np.bincount(train_labels[part], minlength=100).tolist()
    return splits, assignments


def run_control(config, seed, arm, train, test, metadata, output):
    base, use_svd, center, iid = ARMS[arm]
    diagnostic_rows = []

    class ControlClient(FeatureClient):
        def train(self):
            result = super().train()
            if use_svd:
                before_state = self.get_lora_state()
                original = lora_to_delta(before_state, self.alpha)['fc']
                target_state = centered_state(before_state) if center else before_state
                target = lora_to_delta(target_state, self.alpha)['fc']
                state = {'fc': factorize_delta(target, self.rank, self.alpha)}
                reconstructed = lora_to_delta(state, self.alpha)['fc']
                residual = reconstructed - target
                row = matrix_diagnostics(original)
                row['svd_relative_reconstruction_energy'] = float(residual.square().sum()) / max(float(target.square().sum()), 1e-30)
                # Compare centered logits on a fixed TRAINING batch, not test data.
                features = self.train_data['features'][self.probe_indices].detach().cpu()
                old_logits, new_logits = features @ original.T, features @ reconstructed.T
                old_logits -= old_logits.mean(1, keepdim=True)
                new_logits -= new_logits.mean(1, keepdim=True)
                row['max_centered_train_logit_change'] = float((old_logits - new_logits).abs().max())
                row['round'] = len(diagnostic_rows) + 1
                diagnostic_rows.append(row)
                self.set_lora_state(state)
            return result

    class ControlRunner(benchmark.WeightedRunner):
        def gossip_round(self, round_idx, states):
            weights = self.stationary
            average = sum(float(w) * lora_to_delta(s, self.alpha)['fc'] for w, s in zip(weights, states))
            row = matrix_diagnostics(average)
            row['round'] = round_idx + 1
            diagnostic_rows.append(row)
            if center:
                states = [centered_state(s) for s in states]
            return super().gossip_round(round_idx, states)

    output.mkdir(parents=True, exist_ok=True)
    specification = {'arm': arm, 'base_method': base, 'pooled_epoch_svd': use_svd,
                     'center_before_svd': center, 'iid_preserving_client_sizes': iid,
                     'exploratory': True, 'seed': seed}
    write_json(output / f'seed{seed}_intervention.json', specification)
    with patch.object(benchmark, 'FeatureClient', ControlClient), \
         patch.object(benchmark, 'WeightedRunner', ControlRunner), \
         patch.object(benchmark, 'make_splits', iid_splits if iid else make_splits):
        result = benchmark.run_one(config, seed, base, train, test, metadata, output)
    record_path = output / f'seed{seed}_{base}.json'
    record = json.loads(record_path.read_text())
    record['intervention'] = specification
    record['optimization_diagnostics'] = diagnostic_rows
    write_json(record_path, record)
    return {'arm': arm, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--feature-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--rounds', type=int, default=30)
    parser.add_argument('--device', default='cuda:0')
    config = parser.parse_args()
    if config.output.exists() and any(config.output.iterdir()):
        parser.error('use a fresh output directory to preserve all attempts')
    if config.rounds < 1:
        parser.error('rounds must be positive')
    for name, value in dict(alpha=32., reference_rank=16, lr=.001, weight_decay=.0001,
                            batch_size=128, eval_batch_size=2048, local_epochs=1,
                            topology='ring', clients_per_domain=3, max_train_per_client=0,
                            max_test_per_domain=0, n_domains=5, dirichlet_alpha=.5).items():
        setattr(config, name, value)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    manifest = {'status': 'running', 'exploratory': True, 'source': source_provenance(),
                'packages': package_versions(), 'started_unix': time.time(),
                'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(config).items()},
                'results': []}
    write_json(config.output / 'manifest.json', manifest)
    try:
        train, test, metadata = load_features(config.data_dir, config.feature_cache, torch.device(config.device))
        train = {k: v.to(config.device) for k, v in train.items()}
        test = {k: v.to(config.device) for k, v in test.items()}
        manifest['feature_cache'] = metadata
        for seed in config.seeds:
            for arm in config.arms:
                manifest['active'] = {'seed': seed, 'arm': arm}
                write_json(config.output / 'manifest.json', manifest)
                result = run_control(copy.copy(config), seed, arm, train, test, metadata, config.output / arm)
                manifest['results'].append(result)
                write_json(config.output / 'manifest.json', manifest)
        manifest['status'] = 'complete'
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        manifest['finished_unix'] = time.time()
        write_json(config.output / 'manifest.json', manifest)


if __name__ == '__main__':
    main()
