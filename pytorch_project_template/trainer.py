"""
Example trainer for training a classifier using weighted cross-entropy.
"""
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import Accuracy
from tqdm import tqdm, trange

from pytorch_project_template.data import BaseDataModule


class Trainer:
    def __init__(
        self,
        model,
        datamodule: BaseDataModule,
        experimental_setting: Any,  # FIXME: A dummy attribute that would change between runs
        results_dir: Path,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        tqdm_pos=0,
        lr=1e-3,
    ):
        self.device = device
        self.tqdm_pos = tqdm_pos
        self.experimental_setting = experimental_setting

        # model
        self.model = model.to(device)
        self.optim = AdamW(self.model.parameters(), lr=lr)
        self.sched = LambdaLR(self.optim, lr_lambda=lambda epoch: 1 / ((epoch + 1) ** 2))

        # data
        self.n_classes = datamodule.n_classes
        self.train_loader = datamodule.train_dataloader()
        self.val_loader = datamodule.val_dataloader()
        self.test_loader = datamodule.test_dataloader()
        class_weights = datamodule.class_weights.to(self.device)

        # loss functions
        self.train_criterion = lambda log_probs, labels: F.nll_loss(log_probs, labels, weight=class_weights)
        self.test_criterion = lambda log_probs, labels: F.nll_loss(log_probs, labels, weight=class_weights)

        # bookkeeping
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(self.results_dir)

        # Store key configuration here for collecting results across runs
        self.metrics = {"experimental_setting": self.experimental_setting}

    def __call__(self, epochs: int = 50, tiny=False):
        self.global_step = 0  # batches seen
        self.epoch = 0
        for _ in trange(epochs, desc="Epochs", leave=False, position=self.tqdm_pos):
            self.writer.add_scalar("lr", self.sched.get_last_lr()[0], global_step=self.global_step)
            self._train(tiny=tiny)
            self._val(tiny=tiny)
            self.epoch += 1
            self.checkpoint()
            self.sched.step()
        self._test(tiny=tiny)
        return self.metrics

    def _train(self, tiny: bool):
        """Training loop"""
        self.model.train()
        pbar = tqdm(self.train_loader, desc="train", leave=False, position=self.tqdm_pos + 1)
        acc_metric = Accuracy(num_classes=self.n_classes).to(self.device)
        bal_acc_metric = Accuracy(num_classes=self.n_classes, average="macro").to(self.device)

        for data, labels in pbar:
            data, labels = data.to(self.device), labels.to(self.device)
            log_probs = self.model(data)
            loss = self.train_criterion(log_probs, labels)

            # backward pass
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # tracking
            acc = acc_metric(log_probs.argmax(-1), labels)
            bal_acc = bal_acc_metric(log_probs.argmax(-1), labels)
            results = {"train/loss": float(loss), "train/acc": acc, "train/bal_acc": bal_acc}
            pbar.set_postfix({k: f"{v:.3f}" for k, v in results.items()})
            for key, val in results.items():
                self.writer.add_scalar(key, val, self.global_step)
            self.global_step += 1
            if tiny:
                break

        results.update({"epoch": self.epoch})
        self.metrics.update(results)

    @torch.no_grad()
    def __val_test(self, desc: str, loader: DataLoader, tiny: bool):
        """Validation and Test loop"""
        self.model.eval()
        avg_loss, count = 0, 0
        pbar = tqdm(loader, desc=desc, leave=False, position=self.tqdm_pos + 1)
        acc_metric = Accuracy(num_classes=self.n_classes).to(self.device)
        bal_acc_metric = Accuracy(num_classes=self.n_classes, average="macro").to(self.device)
        for data, labels in pbar:
            data, labels = data.to(self.device), labels.to(self.device)
            log_probs = self.model(data)

            # incremental average loss
            count += 1
            avg_loss = avg_loss + (self.test_criterion(log_probs, labels) - avg_loss) / count

            acc = acc_metric(log_probs.argmax(-1), labels)
            bal_acc = bal_acc_metric(log_probs.argmax(-1), labels)
            results = {f"{desc}/loss": avg_loss, f"{desc}/bal_acc": bal_acc, f"{desc}/acc": acc}
            results = {k: float(v) for k, v in results.items()}
            pbar.set_postfix({k: f"{v:.3f}" for k, v in results.items()})
            if tiny:
                break

        self.metrics.update(results)

        for key, val in results.items():
            self.writer.add_scalar(key, val, self.global_step)

    def _val(self, tiny: bool):
        return self.__val_test("val", self.val_loader, tiny=tiny)

    def _test(self, tiny: bool):
        return self.__val_test("test", self.test_loader, tiny=tiny)

    def checkpoint(self):
        ckpt = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optim_state_dict": self.optim.state_dict(),
            "experimental_setting": self.experimental_setting,
        }
        torch.save(ckpt, self.results_dir / "checkpoint.pt")
