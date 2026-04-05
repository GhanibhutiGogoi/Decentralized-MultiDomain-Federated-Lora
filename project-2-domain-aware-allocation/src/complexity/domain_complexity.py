"""
Domain Complexity Analysis for LoRA Rank Allocation.

This module computes a composite domain complexity score for each federated
client based on 5 sub-metrics derived from the client's local data and a
frozen pretrained model. The key hypothesis is that this complexity score
predicts the optimal LoRA rank: more complex domains need higher ranks.

Sub-metrics:
    1. Label entropy        (weight 0.3) — diversity of class distribution
    2. Feature diversity     (weight 0.2) — spread of pretrained representations
    3. Intrinsic dimension   (weight 0.2) — effective dimensionality of feature space
    4. Task difficulty       (weight 0.2) — how poorly the pretrained model performs
    5. Data imbalance        (weight 0.1) — skewness of class distribution
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances
from torchvision import models


DEFAULT_WEIGHTS = {
    'entropy': 0.3,
    'diversity': 0.2,
    'dimensionality': 0.2,
    'difficulty': 0.2,
    'imbalance': 0.1,
}


class DomainComplexityAnalyzer:
    """
    Computes domain complexity scores for federated learning clients.

    Uses a frozen pretrained ResNet-18 to extract features from each
    client's local data, then computes 5 sub-metrics that together
    predict optimal LoRA rank.
    """

    def __init__(self, device='cpu', n_feature_samples=500):
        """
        Args:
            device: compute device
            n_feature_samples: max samples to use for feature extraction
        """
        self.device = device
        self.n_feature_samples = n_feature_samples

        # Create frozen pretrained ResNet-18 for feature extraction
        self.feature_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_model.eval()
        self.feature_model = self.feature_model.to(device)

        # We'll hook into the avgpool layer to get 512-dim features
        self._features = None
        self.feature_model.avgpool.register_forward_hook(self._hook_features)

        # Freeze everything
        for param in self.feature_model.parameters():
            param.requires_grad = False

    def _hook_features(self, module, input, output):
        """Forward hook to capture avgpool features."""
        self._features = output.squeeze(-1).squeeze(-1)

    @torch.no_grad()
    def extract_features(self, data_loader):
        """
        Extract penultimate-layer (512-dim) features from frozen model.

        Args:
            data_loader: client's DataLoader

        Returns:
            numpy array of shape (n_samples, 512)
        """
        all_features = []
        n_collected = 0

        for batch_x, _ in data_loader:
            batch_x = batch_x.to(self.device)
            _ = self.feature_model(batch_x)
            features = self._features.cpu().numpy()
            all_features.append(features)
            n_collected += len(features)
            if n_collected >= self.n_feature_samples:
                break

        all_features = np.concatenate(all_features, axis=0)
        return all_features[:self.n_feature_samples]

    def compute_label_entropy(self, labels):
        """
        Shannon entropy of class distribution, normalized to [0, 1].

        High entropy = many classes with similar counts = more complex.
        Low entropy = dominated by few classes = simpler adaptation needed.

        Args:
            labels: numpy array of integer labels

        Returns:
            float in [0, 1]
        """
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))

        n_classes = len(counts)
        max_entropy = np.log2(n_classes) if n_classes > 1 else 1.0
        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def compute_feature_diversity(self, features):
        """
        Mean pairwise cosine distance between feature vectors.

        High distance = data is spread across diverse representations.
        Low distance = data clusters tightly = simpler patterns.

        Args:
            features: numpy array (n_samples, feature_dim)

        Returns:
            float in [0, 1] (normalized)
        """
        # Subsample for speed
        n = min(200, len(features))
        indices = np.random.choice(len(features), n, replace=False)
        subset = features[indices]

        distances = cosine_distances(subset)
        # Take upper triangle (exclude diagonal)
        triu_indices = np.triu_indices(n, k=1)
        mean_distance = float(np.mean(distances[triu_indices]))

        # Cosine distance ranges [0, 2], normalize to [0, 1]
        return min(mean_distance / 2.0, 1.0)

    def compute_intrinsic_dimensionality(self, features, variance_threshold=0.95):
        """
        Fraction of PCA components needed to explain 95% variance.

        High intrinsic dim = data lives in a high-dimensional subspace.
        Low intrinsic dim = data can be captured by few dimensions.

        Args:
            features: numpy array (n_samples, feature_dim)
            variance_threshold: fraction of variance to explain

        Returns:
            float in [0, 1]
        """
        n_components = min(features.shape[0], features.shape[1])
        if n_components < 2:
            return 0.0

        pca = PCA(n_components=n_components)
        pca.fit(features)

        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_needed = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)

        return float(n_needed / features.shape[1])

    @torch.no_grad()
    def compute_task_difficulty(self, data_loader, num_classes):
        """
        1 - accuracy of frozen pretrained model on client data.

        Uses a random linear head on top of frozen features.
        High difficulty = pretrained features poorly match this domain.

        Args:
            data_loader: client's DataLoader
            num_classes: number of classes for this client

        Returns:
            float in [0, 1]
        """
        # Create a simple linear probe
        probe = nn.Linear(512, num_classes).to(self.device)

        # Collect features and labels
        all_features = []
        all_labels = []
        n_collected = 0

        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(self.device)
            _ = self.feature_model(batch_x)
            all_features.append(self._features.cpu())
            all_labels.append(batch_y)
            n_collected += len(batch_y)
            if n_collected >= self.n_feature_samples:
                break

        features_tensor = torch.cat(all_features)[:self.n_feature_samples]
        labels_tensor = torch.cat(all_labels)[:self.n_feature_samples]

        # Map labels to 0..num_classes-1 range
        unique_labels = torch.unique(labels_tensor)
        label_map = {int(l): i for i, l in enumerate(unique_labels)}
        mapped_labels = torch.tensor([label_map[int(l)] for l in labels_tensor])

        # Quick linear probe training (just to measure alignment, not accuracy)
        probe = nn.Linear(512, len(unique_labels)).to(self.device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        features_tensor = features_tensor.to(self.device)
        mapped_labels = mapped_labels.to(self.device)

        # Train for a few steps
        probe.train()
        for _ in range(50):
            logits = probe(features_tensor)
            loss = criterion(logits, mapped_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate
        probe.eval()
        with torch.no_grad():
            logits = probe(features_tensor)
            preds = logits.argmax(dim=1)
            accuracy = float((preds == mapped_labels).float().mean())

        # Scale by number of classes (more classes = harder)
        difficulty = (1.0 - accuracy) * np.log(len(unique_labels)) / np.log(10)
        return float(np.clip(difficulty, 0.0, 1.0))

    def compute_data_imbalance(self, labels):
        """
        Gini coefficient of class distribution.

        High Gini = heavily imbalanced distribution = harder optimization.
        Low Gini = balanced classes = easier.

        Args:
            labels: numpy array of integer labels

        Returns:
            float in [0, 1]
        """
        _, counts = np.unique(labels, return_counts=True)
        counts = np.sort(counts)
        n = len(counts)
        if n <= 1:
            return 0.0

        index = np.arange(1, n + 1)
        gini = (2 * np.sum(index * counts) - (n + 1) * np.sum(counts)) / (n * np.sum(counts))
        return float(np.clip(gini, 0.0, 1.0))

    def compute_complexity_score(
        self,
        data_loader,
        labels,
        num_classes=20,
        weights=None,
    ):
        """
        Compute the full composite complexity score for one client.

        Args:
            data_loader: client's training DataLoader
            labels: numpy array of client's labels
            num_classes: number of classes in client's domain
            weights: dict of sub-metric weights

        Returns:
            dict with 'score' (float in [0,1]) and all sub-metrics
        """
        if weights is None:
            weights = DEFAULT_WEIGHTS

        # Extract features once
        features = self.extract_features(data_loader)

        # Compute all 5 sub-metrics
        entropy = self.compute_label_entropy(labels)
        diversity = self.compute_feature_diversity(features)
        intrinsic_dim = self.compute_intrinsic_dimensionality(features)
        difficulty = self.compute_task_difficulty(data_loader, num_classes)
        imbalance = self.compute_data_imbalance(labels)

        # Composite score
        score = (
            weights['entropy'] * entropy +
            weights['diversity'] * diversity +
            weights['dimensionality'] * intrinsic_dim +
            weights['difficulty'] * difficulty +
            weights['imbalance'] * imbalance
        )

        # Imbalance penalty: very imbalanced data needs more capacity
        if imbalance > 0.8:
            score *= 1.2

        score = float(np.clip(score, 0.0, 1.0))

        return {
            'score': score,
            'entropy': entropy,
            'diversity': diversity,
            'intrinsic_dim': intrinsic_dim,
            'task_difficulty': difficulty,
            'data_imbalance': imbalance,
        }

    def analyze_all_clients(self, clients_data, train_dataset):
        """
        Compute complexity scores for all clients.

        Args:
            clients_data: dict from create_federated_datasets
            train_dataset: CIFAR-100 training dataset

        Returns:
            dict mapping client_id -> complexity result dict
        """
        from src.data.cifar100_domains import get_client_labels

        results = {}
        for client_id, client_info in clients_data.items():
            print(f"  Analyzing client {client_id} "
                  f"(Domain {client_info['domain_id']}: {client_info['domain_name']})...")

            labels = get_client_labels(client_info, train_dataset)
            num_classes = len(np.unique(labels))

            result = self.compute_complexity_score(
                data_loader=client_info['train_loader'],
                labels=labels,
                num_classes=num_classes,
            )
            result['domain_id'] = client_info['domain_id']
            result['domain_name'] = client_info['domain_name']
            result['n_samples'] = client_info['n_samples']
            result['n_classes'] = num_classes

            results[client_id] = result

            print(f"    Score: {result['score']:.4f} | "
                  f"Entropy: {result['entropy']:.3f} | "
                  f"Diversity: {result['diversity']:.3f} | "
                  f"IntrDim: {result['intrinsic_dim']:.3f} | "
                  f"Difficulty: {result['task_difficulty']:.3f} | "
                  f"Imbalance: {result['data_imbalance']:.3f}")

        return results
