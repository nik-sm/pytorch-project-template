import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from pytorch_project_template.utils import get_project_paths

from .base import BaseDataModule
from .utils import seed_worker


class MapStyleMNIST(BaseDataModule):
    def __init__(self):
        project_path, _, _ = get_project_paths()
        data_dir = project_path / "datasets"
        data_dir.mkdir(exist_ok=True, parents=True)
        self._train_set = MNIST(data_dir, train=True, download=True, transform=ToTensor())

        test_set = MNIST(data_dir, train=False, download=True, transform=ToTensor())
        all_test_indices = torch.randperm(len(test_set))
        val_len = int(0.1 * len(test_set))
        self._val_set = Subset(test_set, all_test_indices[:val_len])
        self._test_set = Subset(test_set, all_test_indices[val_len:])

    @property
    def train_labels(self):
        return self.train_set.targets

    @property
    def n_classes(self):
        return 10

    @property
    def train_set(self):
        return self._train_set

    @property
    def val_set(self):
        return self._val_set

    @property
    def test_set(self):
        return self._test_set

    def _loader(self, dataset: Dataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=32,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=4,
            worker_init_fn=seed_worker,
        )
