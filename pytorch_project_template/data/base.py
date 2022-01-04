from abc import ABC, abstractmethod
from typing import Union

from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset


class BaseDataModule(ABC):
    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_set, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_set, shuffle=True)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_set, shuffle=True)

    @abstractmethod
    def _loader(self, dataset: Union[Dataset, IterableDataset], shuffle: bool) -> DataLoader:
        pass

    @property
    @abstractmethod
    def train_set(self) -> Dataset:
        ...

    @property
    @abstractmethod
    def val_set(self) -> Dataset:
        ...

    @property
    @abstractmethod
    def test_set(self) -> Dataset:
        ...

    @property
    @abstractmethod
    def n_classes(self) -> int:
        ...

    @property
    @abstractmethod
    def train_labels(self) -> Tensor:
        ...

    @property
    def class_weights(self):
        """Inverse class frequencies for weighted cross-entropy"""
        counts = self.train_labels.bincount()
        return counts.sum() / counts
