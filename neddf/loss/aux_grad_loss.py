import torch
from torch import Tensor

from neddf.loss.base_loss import BaseLoss


class AuxGradLoss(BaseLoss):
    """AuxGradLoss.

    This class inheriting base_loss calculate penalty for shape of auxiliary gradients.
    The formulation is from neddf paper.

    Attributes:
        weight (float): weight for color loss
        weight_coarse (float): weight for color loss in coarse model
    """

    def __init__(
        self,
        weight: float = 1.0,
        weight_coarse: float = 0.1,
    ) -> None:
        super().__init__(
            key_output="aux_grad_penalty",
            key_target="aux_grad_penalty",
            key_loss="aux_grad_penalty",
            weight=weight,
            weight_coarse=weight_coarse,
        )

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        res = torch.mean(torch.square(output))
        return res
