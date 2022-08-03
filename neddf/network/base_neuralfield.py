from abc import ABC, abstractmethod
from typing import Dict

from neddf.ray import Sampling
from torch import Tensor, nn


class BaseNeuralField(ABC, nn.Module):
    def set_iter(self, iter: int) -> None:
        """Set iteration

        This methods set iteration number for configure warm ups

        Args:
            iter (int): current iteration. Set -1 for evaluation.
        """
        pass

    @abstractmethod
    def forward(
        self,
        sampling: Sampling,
    ) -> Dict[str, Tensor]:
        """Forward propagation

        This methods should return density and color at minimum

        Args:
            sampling (Sampling[batch_size, sampling, 3])
        """
        raise NotImplementedError()
