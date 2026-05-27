# image dataset loaders

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_cifar10(data_root="./data", batch_size=64):
    """Returns (train_dataset, test_dataset, test_loader) for CIFAR-10."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    train = torchvision.datasets.CIFAR10(
        data_root, train=True,  download=True, transform=transform)
    test  = torchvision.datasets.CIFAR10(
        data_root, train=False, download=True, transform=transform)
    test_loader = DataLoader(test, batch_size=batch_size)
    return train, test, test_loader


def get_fashion_mnist(data_root="./data", batch_size=64):
    """Returns (train_dataset, test_dataset, test_loader) for FashionMNIST."""
    transform = transforms.Compose([transforms.ToTensor()])
    train = torchvision.datasets.FashionMNIST(
        data_root, train=True,  download=True, transform=transform)
    test  = torchvision.datasets.FashionMNIST(
        data_root, train=False, download=True, transform=transform)
    test_loader = DataLoader(test, batch_size=batch_size)
    return train, test, test_loader
