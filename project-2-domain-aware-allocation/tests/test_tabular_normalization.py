"""Regression tests for UCI Heart Disease tabular preprocessing."""

from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT2_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT2_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT2_ROOT))

from framework.datasets.factory import DatasetConfig, DatasetFactory
from framework.datasets.tabular import (
    TabularDataset,
    _deterministic_split_indices,
    _training_normalization_stats,
    get_tabular,
)


def _write_heart_fixture(path: Path, n_rows: int = 297) -> np.ndarray:
    """Write a processed.cleveland-shaped fixture and return raw features."""
    rows = []
    for row_id in range(n_rows):
        features = np.asarray(
            [row_id + (feature_id * 0.25) for feature_id in range(13)],
            dtype=np.float32,
        )
        rows.append([*features.tolist(), int(row_id % 2)])

    train_idx, test_idx = _deterministic_split_indices(n_rows)
    for position in test_idx:
        rows[int(position)][0] += 10000.0
        rows[int(position)][5] -= 5000.0

    pd.DataFrame(rows).to_csv(path / "heart.csv", header=False, index=False)
    return np.asarray([row[:-1] for row in rows], dtype=np.float32)


class TabularNormalizationTest(unittest.TestCase):
    def test_real_loader_uses_train_fitted_scaler_for_train_and_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            raw_X = _write_heart_fixture(data_root)
            y = (np.arange(len(raw_X)) % 2).astype(np.int64)
            train_idx, test_idx = _deterministic_split_indices(len(raw_X))
            expected_mean, expected_std = _training_normalization_stats(raw_X, train_idx)
            full_mean = raw_X.mean(0)
            test_mean = raw_X[test_idx].mean(0)

            train = TabularDataset("train", data_root=data_root)
            test = TabularDataset("test", data_root=data_root)

            self.assertEqual(len(train), 237)
            self.assertEqual(len(test), 60)
            self.assertFalse(np.allclose(expected_mean, full_mean))
            self.assertFalse(np.allclose(expected_mean, test_mean))
            np.testing.assert_allclose(train.normalization_mean, expected_mean)
            np.testing.assert_allclose(test.normalization_mean, expected_mean)
            np.testing.assert_allclose(train.normalization_std, expected_std)
            np.testing.assert_allclose(test.normalization_std, expected_std)
            np.testing.assert_allclose(
                train.X.numpy(),
                ((raw_X - expected_mean) / expected_std)[train_idx],
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                test.X.numpy(),
                ((raw_X - expected_mean) / expected_std)[test_idx],
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_array_equal(train.y.numpy(), y[train_idx])
            np.testing.assert_array_equal(test.y.numpy(), y[test_idx])

    def test_held_out_values_do_not_change_training_statistics(self):
        raw_X = np.arange(297 * 13, dtype=np.float32).reshape(297, 13)
        train_idx, test_idx = _deterministic_split_indices(len(raw_X))
        mean_before, std_before = _training_normalization_stats(raw_X, train_idx)

        perturbed = raw_X.copy()
        perturbed[test_idx] += 1_000_000.0
        mean_after, std_after = _training_normalization_stats(perturbed, train_idx)

        np.testing.assert_allclose(mean_after, mean_before)
        np.testing.assert_allclose(std_after, std_before)

    def test_get_tabular_public_interface_and_factory_expectations(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            _write_heart_fixture(data_root)

            train, test, test_loader = get_tabular(data_root=data_root)
            self.assertIsInstance(train, TabularDataset)
            self.assertIsInstance(test, TabularDataset)
            self.assertIsInstance(test_loader, DataLoader)
            self.assertEqual(len(train), 237)
            self.assertEqual(len(test), 60)
            self.assertEqual(train.in_dim, 13)
            self.assertEqual(test.in_dim, 13)
            self.assertEqual(train.num_classes, 2)
            self.assertEqual(test.num_classes, 2)

            bundle = DatasetFactory(
                DatasetConfig(data_root=data_root)
            ).load("uci_heart_disease")
            self.assertEqual(bundle.metadata.train_sample_count, 237)
            self.assertEqual(bundle.metadata.test_sample_count, 60)
            self.assertEqual(bundle.metadata.class_count, 2)
            self.assertFalse(bundle.metadata.synthetic)

    def test_synthetic_path_counts_and_classes_are_preserved(self):
        train, test, test_loader = get_tabular(synthetic=True)
        self.assertIsInstance(test_loader, DataLoader)
        self.assertEqual(len(train), 3200)
        self.assertEqual(len(test), 160)
        self.assertEqual(train.in_dim, 20)
        self.assertEqual(test.in_dim, 20)
        self.assertEqual(train.num_classes, 4)
        self.assertEqual(test.num_classes, 4)
        self.assertIsInstance(train[0][0], torch.Tensor)


if __name__ == "__main__":
    unittest.main()
