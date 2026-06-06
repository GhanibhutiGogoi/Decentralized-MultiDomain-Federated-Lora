from .image   import get_cifar10, get_fashion_mnist
from .text    import AGNewsDataset, get_agnews
from .tabular import TabularDataset, get_tabular
from .audio   import AudioDataset, get_audio

__all__ = [
    "get_cifar10", "get_fashion_mnist",
    "AGNewsDataset", "get_agnews",
    "TabularDataset", "get_tabular",
    "AudioDataset", "get_audio",
]
