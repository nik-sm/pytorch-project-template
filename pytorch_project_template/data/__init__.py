from .base import BaseDataModule
from .iterable_dataset import IterableStyleMNIST
from .map_style_dataset import MapStyleMNIST

__all__ = [
    "BaseDataModule",
    "IterableStyleMNIST",
    "MapStyleMNIST",
]
