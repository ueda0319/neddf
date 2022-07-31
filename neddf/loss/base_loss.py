from abc import ABC, abstractmethod
from typing import Dict

from torch import Tensor, nn


class BaseLoss(ABC, nn.Module):
    def __init__(
        self,
        weight: float = 1.0,
        weight_coarse: float = 0.1,
    ) -> None:
        super().__init__()
        self.weight: float = weight
        self.weight_coarse: float = weight_coarse

    @abstractmethod
    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        raise NotImplementedError()
