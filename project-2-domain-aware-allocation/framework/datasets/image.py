"""Image dataset adapters used by the centralized dataset factory."""

from pathlib import Path

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[2] / "experiment" / "data"


def get_cifar10(
    data_root=None,
    batch_size=64,
    synthetic=False,
    download=False,
    num_workers=0,
    pin_memory=False,
):
    """Returns (train_dataset, test_dataset, test_loader) for CIFAR-10."""
    if synthetic:
        raise ValueError("CIFAR-10 does not support synthetic mode.")
    data_root = str(data_root or DEFAULT_DATA_ROOT)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    train = torchvision.datasets.CIFAR10(
        data_root, train=True, download=download, transform=transform)
    test = torchvision.datasets.CIFAR10(
        data_root, train=False, download=download, transform=transform)
    train.is_synthetic = False
    test.is_synthetic = False
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train, test, test_loader


def get_fashion_mnist(
    data_root=None,
    batch_size=64,
    synthetic=False,
    download=False,
    num_workers=0,
    pin_memory=False,
):
    """Returns (train_dataset, test_dataset, test_loader) for FashionMNIST."""
    if synthetic:
        raise ValueError("FashionMNIST does not support synthetic mode.")
    data_root = str(data_root or DEFAULT_DATA_ROOT)
    transform = transforms.Compose([transforms.ToTensor()])
    train = torchvision.datasets.FashionMNIST(
        data_root, train=True, download=download, transform=transform)
    test = torchvision.datasets.FashionMNIST(
        data_root, train=False, download=download, transform=transform)
    train.is_synthetic = False
    test.is_synthetic = False
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train, test, test_loader


def get_cifar100(
    data_root=None,
    batch_size=64,
    synthetic=False,
    download=False,
    num_workers=0,
    pin_memory=False,
    train_transform=None,
    test_transform=None,
):
    """Returns (train_dataset, test_dataset, test_loader) for CIFAR-100."""
    if synthetic:
        raise ValueError("CIFAR-100 does not support synthetic mode.")
    data_root = str(data_root or DEFAULT_DATA_ROOT)
    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761],
            ),
        ])
    if test_transform is None:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761],
            ),
        ])
    train = torchvision.datasets.CIFAR100(
        data_root, train=True, download=download, transform=train_transform)
    test = torchvision.datasets.CIFAR100(
        data_root, train=False, download=download, transform=test_transform)
    train.is_synthetic = False
    test.is_synthetic = False
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train, test, test_loader
