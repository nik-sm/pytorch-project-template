import math
import random

import torch
from torch.utils.data import DataLoader, IterableDataset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from pytorch_project_template.utils import get_project_paths

from .base import BaseDataModule
from .utils import seed_worker


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


class MNISTIterableDataset(IterableDataset):
    """
    Iterable datasets, when custom batching logic is required.
    To motivate this use case, here we load batches of only a single digit at a time.
    """

    def __init__(self, data, labels):
        super().__init__()
        self.data = data
        self.labels = labels
        self.batch_size = 32
        self.label_to_indices = {}
        for label in self.labels.unique(sorted=True).tolist():
            self.label_to_indices[label] = torch.where(self.labels == label)[0]
        self._shuffle_work_list()  # Run once at beginning, so we can determine len(self) for progress bars

    def _shuffle_work_list(self):
        """
        Make a list of batches. Each batch contains a single digit type.
        We'll keep re-creating this work list for each epoch.
        For multi-worker loading, we'll just divide this list into chunks after shuffling.
        """
        self.work_list = []
        for indices in self.label_to_indices.values():
            indices = indices[torch.randperm(len(indices))].tolist()  # shuffle indices
            for chunk in chunker(indices, self.batch_size):
                self.work_list.append(chunk)
        random.shuffle(self.work_list)

    def __iter__(self):
        self._shuffle_work_list()
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading
            for data_idx in self.work_list:
                yield self.data[data_idx].unsqueeze(1), self.labels[data_idx]

        else:  # multi-process data loading
            batches_per_worker = int(math.ceil(len(self) / float(worker_info.num_workers)))
            assigned_each_worker = list(chunker(range(len(self)), batches_per_worker))
            this_worker_batch_list = assigned_each_worker[worker_info.id]
            for which_batches in this_worker_batch_list:  # several indices chosen for a single batch
                data_idx = self.work_list[which_batches]
                yield self.data[data_idx].unsqueeze(1), self.labels[data_idx]

    def __len__(self):
        """Number of batches"""
        return len(self.work_list)


class IterableStyleMNIST(BaseDataModule):
    def __init__(self):
        project_path, _, _ = get_project_paths()
        data_dir = project_path / "datasets"
        data_dir.mkdir(exist_ok=True, parents=True)

        # Setup train set
        mnist = MNIST(data_dir, train=True, download=True, transform=ToTensor())
        data = []
        for d, _ in mnist:  # These loops are just to apply ToTensor() across the data
            data.append(d)
        train_data, train_labels = torch.cat(data), mnist.targets
        self._train_set = MNISTIterableDataset(train_data, train_labels)

        # Setup val and test sets
        mnist_test = MNIST(data_dir, train=False, download=True, transform=ToTensor())
        data = []
        for d, _ in mnist_test:
            data.append(d)
        data = torch.cat(data)

        all_test_indices = torch.randperm(len(mnist_test))
        val_len = int(0.1 * len(mnist_test))

        # val
        val_idx = all_test_indices[:val_len]
        val_data, val_labels = data[val_idx], mnist_test.targets[val_idx]
        self._val_set = MNISTIterableDataset(val_data, val_labels)

        # test
        test_idx = all_test_indices[val_len:]
        test_data, test_labels = data[test_idx], mnist_test.targets[test_idx]
        self._test_set = MNISTIterableDataset(test_data, test_labels)

    @property
    def train_labels(self):
        return self.train_set.labels

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

    def _loader(self, dataset: IterableDataset, shuffle: bool):
        return DataLoader(
            dataset,
            batch_size=None,  # This is controlled within the dataset
            shuffle=False,  # Also controlled within dataset
            pin_memory=True,
            num_workers=4,
            worker_init_fn=seed_worker,
        )
