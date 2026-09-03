"""Regression tests for AG News train-fitted vocabulary sharing."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader


PROJECT2_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT2_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT2_ROOT))

from framework.datasets.factory import DatasetConfig, DatasetFactory
from framework.datasets.text import AGNewsDataset, get_agnews


TRAIN_RECORDS = [
    (1, "alpha common trainonly"),
    (2, "beta common"),
]
TEST_RECORDS = [
    (3, "common testonly alien"),
    (4, "alpha alien"),
]


class FakeVocab:
    def __init__(self, token_iterable, specials, max_tokens):
        self.itos = []
        self.stoi = {}
        for token in specials:
            self._add(token, max_tokens)
        for tokens in token_iterable:
            for token in tokens:
                self._add(token, max_tokens)
        self.default_index = None

    def _add(self, token, max_tokens):
        if token in self.stoi:
            return
        if max_tokens is not None and len(self.itos) >= max_tokens:
            return
        self.stoi[token] = len(self.itos)
        self.itos.append(token)

    def __getitem__(self, token):
        if token in self.stoi:
            return self.stoi[token]
        if self.default_index is not None:
            return self.default_index
        raise KeyError(token)

    def __call__(self, tokens):
        return [self[token] for token in tokens]

    def __len__(self):
        return len(self.itos)

    def set_default_index(self, index):
        self.default_index = int(index)


def fake_tokenizer(text):
    return text.lower().split()


def fake_ag_news(*, root, split):
    del root
    return {"train": TRAIN_RECORDS, "test": TEST_RECORDS}[split]


def fake_build_vocab_from_iterator(token_iterable, specials, max_tokens=None):
    return FakeVocab(token_iterable, specials, max_tokens)


def patched_torchtext_modules():
    torchtext = types.ModuleType("torchtext")
    datasets = types.ModuleType("torchtext.datasets")
    data = types.ModuleType("torchtext.data")
    utils = types.ModuleType("torchtext.data.utils")
    vocab = types.ModuleType("torchtext.vocab")
    datasets.AG_NEWS = fake_ag_news
    utils.get_tokenizer = lambda name: fake_tokenizer
    vocab.build_vocab_from_iterator = fake_build_vocab_from_iterator
    torchtext.datasets = datasets
    torchtext.data = data
    torchtext.vocab = vocab
    data.utils = utils
    return {
        "torchtext": torchtext,
        "torchtext.datasets": datasets,
        "torchtext.data": data,
        "torchtext.data.utils": utils,
        "torchtext.vocab": vocab,
    }


def write_cache_markers(data_root: Path) -> None:
    cache = data_root / "datasets" / "AG_NEWS"
    cache.mkdir(parents=True)
    (cache / "train.csv").write_text("cached fixture marker\n", encoding="utf-8")
    (cache / "test.csv").write_text("cached fixture marker\n", encoding="utf-8")


class AGNewsVocabularyTest(unittest.TestCase):
    def test_get_agnews_uses_training_vocabulary_for_test_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            write_cache_markers(data_root)

            with patch.dict(sys.modules, patched_torchtext_modules()):
                train, test, test_loader = get_agnews(
                    data_root=data_root,
                    batch_size=2,
                    download=False,
                )

        self.assertIsInstance(train, AGNewsDataset)
        self.assertIsInstance(test, AGNewsDataset)
        self.assertIsInstance(test_loader, DataLoader)
        self.assertIs(train.vocab, test.vocab)
        self.assertIs(train.tokenizer, test.tokenizer)
        self.assertEqual(train.vocab.stoi, test.vocab.stoi)

        self.assertEqual(train.vocab["<pad>"], 0)
        self.assertEqual(train.vocab["<unk>"], 1)
        self.assertEqual(train.vocab.default_index, train.vocab["<unk>"])
        self.assertIn("trainonly", train.vocab.stoi)
        self.assertNotIn("testonly", train.vocab.stoi)
        self.assertNotIn("alien", train.vocab.stoi)
        self.assertEqual(train.vocab["alien"], train.vocab["<unk>"])
        self.assertEqual(len(train.vocab), 6)

        test_only_id = int(test[0][0][1])
        self.assertEqual(test_only_id, train.vocab["<unk>"])
        self.assertEqual(len(test.vocab), len(train.vocab))
        self.assertEqual(test.vocab["alpha"], train.vocab["alpha"])

        provenance = train.vocab_provenance
        self.assertEqual(provenance["dataset"], "AG News")
        self.assertEqual(provenance["source_split"], "train")
        self.assertEqual(provenance["tokenizer"], "basic_english")
        self.assertEqual(provenance["pad_index"], 0)
        self.assertEqual(provenance["unk_index"], 1)
        self.assertEqual(provenance["vocab_size"], len(train.vocab))
        self.assertIs(test.vocab_provenance, train.vocab_provenance)

    def test_independent_test_vocabulary_construction_would_change_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            write_cache_markers(data_root)

            with patch.dict(sys.modules, patched_torchtext_modules()):
                train, test, _ = get_agnews(data_root=data_root, download=False)
                independent_test_vocab = FakeVocab(
                    (fake_tokenizer(text) for _, text in TEST_RECORDS),
                    specials=["<pad>", "<unk>"],
                    max_tokens=AGNewsDataset.VOCAB_SIZE,
                )

        self.assertEqual(test.vocab["alien"], train.vocab["<unk>"])
        self.assertNotEqual(independent_test_vocab.stoi, train.vocab.stoi)
        self.assertIn("testonly", independent_test_vocab.stoi)
        self.assertIn("alien", independent_test_vocab.stoi)
        self.assertNotIn("trainonly", independent_test_vocab.stoi)

    def test_dataset_factory_ag_news_contract_with_mocked_real_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            write_cache_markers(data_root)
            registry = dict(DatasetFactory.REGISTRY)
            registry["ag_news"] = replace(
                registry["ag_news"],
                expected_train_samples=len(TRAIN_RECORDS),
                expected_test_samples=len(TEST_RECORDS),
                expected_classes=2,
            )

            with patch.dict(sys.modules, patched_torchtext_modules()):
                with patch.object(DatasetFactory, "REGISTRY", registry):
                    bundle = DatasetFactory(
                        DatasetConfig(data_root=data_root, batch_size=2)
                    ).load("ag_news")

        self.assertIsInstance(bundle.train, AGNewsDataset)
        self.assertIsInstance(bundle.test, AGNewsDataset)
        self.assertIsInstance(bundle.test_loader, DataLoader)
        self.assertEqual(bundle.metadata.train_sample_count, len(TRAIN_RECORDS))
        self.assertEqual(bundle.metadata.test_sample_count, len(TEST_RECORDS))
        self.assertEqual(bundle.metadata.vocabulary_provenance["source_split"], "train")
        self.assertFalse(bundle.metadata.synthetic)

    def test_synthetic_ag_news_opt_in_contract_is_preserved(self):
        train, test, test_loader = get_agnews(synthetic=True, batch_size=8)

        self.assertIsInstance(test_loader, DataLoader)
        self.assertEqual(len(train), 5000)
        self.assertEqual(len(test), 1000)
        self.assertTrue(train.is_synthetic)
        self.assertTrue(test.is_synthetic)
        self.assertIsNone(train.vocab_provenance)
        self.assertIsInstance(train[0][0], torch.Tensor)
        self.assertEqual(train[0][0].shape[0], AGNewsDataset.MAX_LEN)


if __name__ == "__main__":
    unittest.main()
