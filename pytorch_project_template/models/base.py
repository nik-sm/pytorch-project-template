from abc import ABC, abstractmethod

from torch import Tensor, nn


class BaseModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, data) -> Tensor:
        ...
