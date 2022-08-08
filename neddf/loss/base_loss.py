from abc import ABC, abstractmethod
from typing import Dict

from torch import Tensor, nn


class BaseLoss(ABC, nn.Module):
    """Abstract base class for loss function.

    Attributes:
        weight (float): weight of this loss function
        weight_coarse (float): weight of this loss function in coarse model
        key_output (str): dictionary key in output
        key_target (str): dictionary key in target
        key_loss (str): dictionary key in return
    """

    def __init__(
        self,
        key_output: str,
        key_target: str,
        key_loss: str,
        weight: float = 1.0,
        weight_coarse: float = 0.1,
    ) -> None:
        """Initializer

        This method initialize common attributes.

        Args:
            weight (float): weight of this loss function
            weight_coarse (float): weight of this loss function in coarse model
            key_output (str): dictionary key in output
            key_target (str): dictionary key in target
            key_loss (str): dictionary key in return

        """
        super().__init__()
        self.weight: float = weight
        self.weight_coarse: float = weight_coarse
        self.key_output: str = key_output
        self.key_target: str = key_target
        self.key_loss: str = key_loss

    def forward(
        self,
        outputs: Dict[str, Tensor],
        targets: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """Forward

        Wrapper of calculate loss value in the objective function
        This method is inheritance implementation of nn.Module, called on self()

        Args:
            outputs (Dict[str, Tensor]):
                Inference results of network.
                This should include key_output as key.
            targets (Dict[str, Tensor]):
                The target values outputs should have taken.
                This should include key_target as key.

        """
        assert self.key_output in outputs
        assert self.key_target in targets
        loss_dict: Dict[str, Tensor] = {}
        loss_dict[self.key_loss] = self.weight * self.loss(
            outputs[self.key_output], targets[self.key_target]
        )
        if self.weight_coarse > 0.0:
            key_output_coarse: str = "{}_coarse".format(self.key_output)
            key_loss_coarse: str = "{}_coarse".format(self.key_loss)
            assert key_output_coarse in outputs
            loss_dict[key_loss_coarse] = self.weight_coarse * self.loss(
                outputs[key_output_coarse], targets[self.key_target]
            )

        return loss_dict

    @abstractmethod
    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        """Loss

        Calculate loss value in the objective function
        This method must be inherited for execute forward method

        Args:
            output (Tensor): Inference result of network
            target (Tensor): The target values output should have taken

        """
        raise NotImplementedError()
