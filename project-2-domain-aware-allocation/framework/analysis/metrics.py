"""
Metrics tracking for federated learning experiments.
"""

import json
import os
from datetime import datetime


class MetricsTracker:
    """Track and save experiment metrics across rounds."""

    def __init__(self, experiment_name, log_dir='./logs'):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = {
            'experiment': experiment_name,
            'start_time': self.start_time,
            'rounds': [],
        }

        os.makedirs(log_dir, exist_ok=True)

    def log_round(self, round_idx, **kwargs):
        """Log metrics for a single round."""
        entry = {'round': round_idx}
        entry.update(kwargs)
        self.metrics['rounds'].append(entry)

    def save(self):
        """Save metrics to JSON file."""
        filename = f"{self.experiment_name}_{self.start_time}.json"
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        print(f"Metrics saved to {filepath}")
        return filepath

    def get_best_accuracy(self):
        """Get the best average accuracy across all rounds."""
        if not self.metrics['rounds']:
            return 0.0
        return max(r.get('avg_accuracy', 0) for r in self.metrics['rounds'])

    def get_final_accuracy(self):
        """Get the final round accuracy."""
        if not self.metrics['rounds']:
            return 0.0
        return self.metrics['rounds'][-1].get('avg_accuracy', 0)


def compute_fairness_metrics(per_domain_accuracy):
    """
    Compute fairness metrics across domains.

    Args:
        per_domain_accuracy: dict mapping domain_id -> accuracy

    Returns:
        dict with fairness metrics
    """
    accuracies = list(per_domain_accuracy.values())
    if not accuracies:
        return {}

    max_acc = max(accuracies)
    min_acc = min(accuracies)
    avg_acc = sum(accuracies) / len(accuracies)
    variance = sum((a - avg_acc) ** 2 for a in accuracies) / len(accuracies)

    return {
        'accuracy_gap': max_acc - min_acc,
        'accuracy_variance': variance,
        'min_domain_accuracy': min_acc,
        'max_domain_accuracy': max_acc,
        'avg_domain_accuracy': avg_acc,
    }


def compute_communication_cost(lora_state, bits=32):
    """
    Compute communication cost of transmitting LoRA parameters.

    Args:
        lora_state: dict of LoRA params
        bits: bits per parameter (32 for float32)

    Returns:
        cost in bytes
    """
    total_params = 0
    for layer_name, params in lora_state.items():
        total_params += params['A'].numel() + params['B'].numel()
    return total_params * bits // 8
