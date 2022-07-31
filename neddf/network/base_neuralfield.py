from abc import ABC, abstractmethod
from typing import Dict

from torch import Tensor, nn


class BaseNeuralField(ABC, nn.Module):
    @abstractmethod
    def forward(
        self,
        input_pos: Tensor,
        input_dir: Tensor,
    ) -> Dict[str, Tensor]:
        """Forward propagation

        This methods should return density and color at minimum

        Args:
            input_pos (Tensor[batch_size, sampling, 3, float32]):
                input point positions
                If you need to use PE, please enter the tensor you have already applied PE.
            input_dir (Tensor[batch_size, sampling, 3, float32]):
                input point positions
                If you need to use PE, please enter the tensor you have already applied PE.
        """
        raise NotImplementedError()