"""Independent full-test evaluation of the saved integrated LoRA checkpoints.

Run on gpu003 after all methods/seeds finish. Does not import experiment,
aggregation, feature-cache, or score_state code. It evaluates the factors
explicitly in float64 and also checks a directly reconstructed fp32 head.
"""
import argparse
import datetime
import hashlib
import json
import math
import platform
import socket
from pathlib import Path

import torch


def digest_tensors(*values):
    digest = hashlib.sha256()
    for tensor in values:
        value = tensor.detach().cpu().contiguous()
        digest.update(str((tuple(value.shape), str(value.dtype))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def prediction_digest(predictions):
    return hashlib.sha256(predictions.numpy().astype('<i8').tobytes()).hexdigest()


def evaluate(checkpoint, features, labels, batch_size=509):
    state, base = checkpoint['state'], checkpoint['initial']
    if set(state) != {'fc'}:
        raise ValueError('expected the documented head-only fc checkpoint')
    a, b = state['fc']['A'].cpu(), state['fc']['B'].cpu()
    weight, bias = base['weight'].cpu(), base['bias'].cpu()
    rank = int(a.shape[0])
    scale = float(checkpoint['alpha']) / rank
    if tuple(a.shape) != (16, 512) or tuple(b.shape) != (100, 16):
        raise ValueError('final deployment adapter must have rank 16 and head dimensions 100 x 512')
    for name, tensor in [('A', a), ('B', b), ('weight', weight), ('bias', bias)]:
        if not torch.isfinite(tensor).all():
            raise ValueError(f'non-finite {name}')
    a64, b64, weight64, bias64 = a.double(), b.double(), weight.double(), bias.double()
    delta64 = scale * (b64 @ a64)
    numeric_rank = int(torch.linalg.matrix_rank(delta64).item())
    singular = torch.linalg.svdvals(delta64)
    # fp32 path mirrors the represented dense head, but uses explicit matmul
    # and bias addition, independent batching, and no benchmark helper.
    dense32 = weight.float() + scale * (b.float() @ a.float())
    predictions64, predictions32, boundary_margins = [], [], []
    disagreements = []
    max_logit_error = 0.0
    with torch.inference_mode():
        for start in range(0, len(labels), batch_size):
            x32 = features[start:start + batch_size].float()
            x64 = x32.double()
            logits64 = x64 @ weight64.T + bias64
            logits64 = logits64 + scale * ((x64 @ a64.T) @ b64.T)
            logits32 = x32 @ dense32.T + bias.float()
            p64, p32 = logits64.argmax(1), logits32.argmax(1)
            predictions64.append(p64)
            predictions32.append(p32)
            best = logits64.topk(2, dim=1).values
            boundary_margins.append(best[:, 0] - best[:, 1])
            max_logit_error = max(max_logit_error, float((logits64 - logits32.double()).abs().max()))
            for offset in (p64 != p32).nonzero().flatten().tolist():
                pred32, pred64 = int(p32[offset]), int(p64[offset])
                disagreements.append({
                    'test_index': start + offset, 'true_class': int(labels[start + offset]),
                    'prediction_fp32': pred32, 'prediction_fp64': pred64,
                    'fp64_top_two_margin': float(best[offset, 0] - best[offset, 1]),
                    'fp32_logits_for_fp32_and_fp64_classes': [float(logits32[offset, pred32]), float(logits32[offset, pred64])],
                    'fp64_logits_for_fp32_and_fp64_classes': [float(logits64[offset, pred32]), float(logits64[offset, pred64])],
                    'maximum_absolute_logit_difference_on_example': float((logits64[offset] - logits32[offset].double()).abs().max()),
                })
    pred64, pred32 = torch.cat(predictions64), torch.cat(predictions32)
    margins = torch.cat(boundary_margins)
    return {
        'configured_rank': rank,
        'numeric_rank_fp64_factor_product': numeric_rank,
        'singular_values_fp64': singular.tolist(),
        'full_test_correct_fp64_factorized': int((pred64 == labels).sum()),
        'full_test_correct_fp32_dense': int((pred32 == labels).sum()),
        'full_test_accuracy_fp64_factorized': float((pred64 == labels).double().mean()),
        'prediction_sha256_fp64_factorized': prediction_digest(pred64),
        'prediction_sha256_fp32_dense': prediction_digest(pred32),
        'prediction_hash_encoding': 'test order, little-endian int64 class labels',
        'fp32_fp64_prediction_disagreements': int((pred64 != pred32).sum()),
        'precision_disagreement_details': disagreements,
        'maximum_absolute_fp32_fp64_logit_difference': max_logit_error,
        'minimum_top_two_logit_margin_fp64': float(margins.min()),
        'initial_state_sha256': digest_tensors(*base.values()),
        'finite_parameters': True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads((args.run_dir / 'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('independent final verification requires a completed run manifest')
    config = manifest['config']
    cache = Path(config['feature_cache']) / 'cifar100-resnet18-imagenet1k-v1-resize224-v1'
    metadata = json.loads((cache / 'manifest.json').read_text())
    test = torch.load(cache / 'test.pt', map_location='cpu', weights_only=True)
    features, labels = test['features'], test['labels'].long()
    test_digest = digest_tensors(features, test['labels'])
    if test_digest != metadata['test']['sha256'] or len(labels) != 10000:
        raise ValueError('full test cache identity/count mismatch')
    identity_fields = ('schema_version', 'dataset', 'representation', 'weights', 'weights_sha256', 'transform')
    identity = {key: metadata[key] for key in identity_fields}
    identity['feature_sha256'] = {split: metadata[split]['sha256'] for split in ('train', 'test')}
    identity_digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if identity_digest != metadata['cache_identity_sha256']:
        raise ValueError('cache manifest identity mismatch')
    results = []
    checks = []
    for seed in config['seeds']:
        split = json.loads((args.run_dir / f'splits_seed{seed}.json').read_text())
        indices = sorted(index for value in split['clients'].values() for index in value['test_indices'])
        if indices != list(range(len(labels))):
            raise ValueError(f'seed {seed} does not use each full-test example exactly once')
        for method in config['methods']:
            record = json.loads((args.run_dir / f'seed{seed}_{method}.json').read_text())
            if record['status'] != 'complete' or len(record['rounds']) != config['rounds']:
                raise ValueError(f'{seed}/{method} is incomplete')
            path = args.run_dir / f'seed{seed}_{method}_adapter.pt'
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            if checkpoint['seed'] != seed or checkpoint['method'] != method:
                raise ValueError(f'{path.name}: checkpoint identity mismatch')
            if float(checkpoint['alpha']) != float(config['alpha']):
                raise ValueError(f'{path.name}: alpha mismatch')
            evaluation = evaluate(checkpoint, features, labels)
            recorded_correct = int(record['rounds'][-1]['full_test_correct'])
            passed = (
                evaluation['full_test_correct_fp32_dense'] == recorded_correct
                and evaluation['numeric_rank_fp64_factor_product'] <= 16
                and evaluation['initial_state_sha256'] == record['initial_state_sha256']
                and record['feature_cache_identity_sha256'] == identity_digest
                and record['n_test'] == 10000
                and math.isclose(record['final_full_test_accuracy'], recorded_correct / 10000, abs_tol=1e-14)
            )
            results.append({
                'seed': seed, 'method': method, 'checkpoint': path.name,
                'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'recorded_full_test_correct': recorded_correct,
                'fp64_reference_correct_count_matches': evaluation['full_test_correct_fp64_factorized'] == recorded_correct,
                'passed': passed, **evaluation,
            })
            checks.append(passed)
            print(f'{seed}/{method}: recorded={recorded_correct} '
                  f'fp64={evaluation["full_test_correct_fp64_factorized"]} '
                  f'fp32={evaluation["full_test_correct_fp32_dense"]} '
                  f'rank={evaluation["numeric_rank_fp64_factor_product"]} passed={passed}', flush=True)
    expected = len(config['seeds']) * len(config['methods'])
    payload = {
        'schema_version': 2, 'status': 'fp32_reproduction_passed' if all(checks) and len(results) == expected else 'failed',
        'verification_criterion': 'Exact independent reproduction of the recorded fp32 model accuracy; fp64 factorized inference is a separately reported precision sensitivity analysis',
        'strict_both_precision_correct_count_agreement': all(row['fp64_reference_correct_count_matches'] for row in results) and all(checks),
        'total_fp32_fp64_prediction_disagreements': sum(row['fp32_fp64_prediction_disagreements'] for row in results),
        'precision_note': 'The initial stricter cross-precision report is preserved as independent_checkpoint_evaluation_strict_initial.json; no trained model or reported fp32 result is changed',
        'evaluated_checkpoints': len(results), 'expected_checkpoints': expected,
        'test_examples_per_checkpoint': len(labels), 'test_tensor_sha256': test_digest,
        'cache_identity_sha256': identity_digest,
        'run_source_sha256': manifest['source']['combined_sha256'],
        'evaluator_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'evaluator_scope': 'independent explicit factorized float64 and reconstructed dense fp32 forward passes; no project imports',
        'machine': socket.gethostname(), 'python': platform.python_version(), 'torch': torch.__version__,
        'device': 'CPU on the required remote test host',
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'results': results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    if payload['status'] != 'fp32_reproduction_passed':
        raise SystemExit('one or more saved-checkpoint checks disagreed; see JSON before drawing conclusions')


if __name__ == '__main__':
    main()
